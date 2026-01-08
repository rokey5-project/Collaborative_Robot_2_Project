import os
import tempfile
import subprocess
import time
import threading

from queue import Queue
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from dotenv import load_dotenv
from openai import OpenAI
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class FaceAgeNode(Node):
    def __init__(self):
        super().__init__('face_age_node')

        openai_api_key = os.getenv("OPENAI_API_KEY")

        # API Key 체크
        if not openai_api_key:
            self.get_logger().error(f"❌ API KEY 로드 실패! 시도한 경로")
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

        # ROS2 설정
        self.sub = self.create_subscription(Bool, '/need_face_check', self.start_callback, 10)
        self.done_pub = self.create_publisher(String, '/task_done', 10)

        # 모델 및 클라이언트 초기화
        self.client = OpenAI(api_key=openai_api_key)
        model_name = "prithivMLmods/facial-age-detection"
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.id2label = {
            0: "age 01-10", 1: "age 11-20", 2: "age 21-30", 3: "age 31-40",
            4: "age 41-55", 5: "age 56-65", 6: "age 66-80", 7: "age 80+"
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # 상태 관리 변수
        self.next_person_id = 0
        self.tracks = {}
        self.TRACK_IOU_TH = 0.4
        self.TRACK_TTL = 2.0
        self.is_active = False

        # TTS 큐 및 스레드
        self.tts_queue = Queue()
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()

        self.get_logger().info("--- FaceAgeNode Online: Waiting for Signal ---")

    def start_callback(self, msg: Bool):
        if msg.data and not self.is_active:
            self.get_logger().info("🔞 성인 인증 프로세스 시작")
            self.is_active = True
            threading.Thread(target=self.run_logic, daemon=True).start()

    def _ensure_resources_cleaned(self):
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)
        temp_cap = cv2.VideoCapture(8)
        if temp_cap.isOpened():
            temp_cap.release()
            time.sleep(0.5)

    def tts_worker(self):
        while True:
            text = self.tts_queue.get()
            if text is None: break
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    audio_path = f.name
                with self.client.audio.speech.with_streaming_response.create(
                    model="gpt-4o-mini-tts", voice="alloy", input=text
                ) as response:
                    response.stream_to_file(audio_path)
                subprocess.run(["ffplay", "-nodisp", "-autoexit", audio_path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(audio_path)
            except Exception as e:
                self.get_logger().error(f"TTS Error: {e}")
            finally:
                self.tts_queue.task_done()

    def classify_image(self, image):
        image = Image.fromarray(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()
        return self.id2label[int(torch.argmax(probs))]

    def assign_id(self, box):
        now = time.time()
        best_iou, best_id = 0.0, None
        for pid, info in self.tracks.items():
            score = self.iou(box, info["box"])
            if score > best_iou:
                best_iou, best_id = score, pid
        if best_iou >= self.TRACK_IOU_TH:
            self.tracks[best_id]["box"] = box
            self.tracks[best_id]["last_seen"] = now
            return best_id
        pid = self.next_person_id
        self.next_person_id += 1
        self.tracks[pid] = {"box": box, "warned": False, "last_seen": now}
        return pid

    def cleanup_tracks(self):
        now = time.time()
        self.tracks = {pid: info for pid, info in self.tracks.items()
                       if now - info["last_seen"] <= self.TRACK_TTL}

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea

        return interArea / union if union > 0 else 0.0

    def run_logic(self):
        self._ensure_resources_cleaned()
        cap = cv2.VideoCapture(8)

        if not cap.isOpened():
            self.get_logger().error("웹캠 오픈 실패!")
            self.is_active = False
            return

        start_time = time.time()
        # 15초 동안 감지 수행
        while rclpy.ok() and (time.time() - start_time < 15.0):
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 10, minSize=(30, 30))

            for (x, y, w, h) in faces:
                pid = self.assign_id((x, y, w, h))
                face = frame[y:y+h, x:x+w]
                if face.size == 0: continue

                age_group = self.classify_image(face)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {pid} | {age_group}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if age_group in ("age 01-10", "age 11-20"):
                    if not self.tracks[pid]["warned"]:
                        self.tts_queue.put("미성년자는 구매할 수 없습니다.")
                        self.tracks[pid]["warned"] = True

            self.cleanup_tracks()
            cv2.imshow("Age Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

        cap.release()
        cv2.destroyAllWindows()

        # 완료 신호 송신
        done_msg = String()
        done_msg.data = "CALC_DONE"
        self.done_pub.publish(done_msg)

        self.is_active = False
        self.get_logger().info("--- 성인 인증 종료 및 리셋 신호 송신 ---")

def main():
    rclpy.init()
    try:
        node = FaceAgeNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 중복 셧다운 에러 방지
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()