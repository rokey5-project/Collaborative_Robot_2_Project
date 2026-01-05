import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import time
import cv2
import torch
import threading
from queue import Queue
from collections import deque
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import tempfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String   # 🔥 (추가) String
from openai import OpenAI


# ===============================
# ENV
# ===============================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")


# ===============================
# IoU
# ===============================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter / union if union > 0 else 0.0


class FaceAgeNode(Node):
    def __init__(self):
        super().__init__("face_age_node")

        # ItemCheckNode → FaceAgeNode
        self.sub = self.create_subscription(
            Bool, "/need_face_check", self.start_cb, 10
        )

        self.active = False

        # 🔥 (추가) calc 종료 신호 퍼블리셔
        self.calc_done_pub = self.create_publisher(
            String,
            "/task_done",
            10
        )

        # OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        # Age model
        model_name = "prithivMLmods/facial-age-detection"
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

        self.id2label = {
            0: "age 01-10",
            1: "age 11-20",
            2: "age 21-30",
            3: "age 31-40",
            4: "age 41-55",
            5: "age 56-65",
            6: "age 66-80",
            7: "age 80+"
        }

        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Tracking
        self.tracks = {}   # pid: {box, history, decided}
        self.next_pid = 0
        self.IOU_TH = 0.4

        # TTS
        self.tts_queue = Queue()
        threading.Thread(target=self.tts_worker, daemon=True).start()

        self.get_logger().info("FaceAge node ready (waiting /need_face_check)")

    # ===============================
    # ROS callback
    # ===============================
    def start_cb(self, msg: Bool):
        if msg.data and not self.active:
            self.active = True
            self.tracks.clear()
            self.get_logger().info("need_face_check received → FaceAge start")
            threading.Thread(target=self.run, daemon=True).start()

    # ===============================
    # TTS
    # ===============================
    def tts_worker(self):
        while True:
            text = self.tts_queue.get()
            if text is None:
                break

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                path = f.name

            with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text
            ) as r:
                r.stream_to_file(path)

            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            os.remove(path)

    # ===============================
    # Vision utils
    # ===============================
    def classify(self, face):
        image = Image.fromarray(face).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return self.id2label[int(torch.argmax(logits))]

    def assign_id(self, box):
        best_iou, best_id = 0.0, None
        for pid, info in self.tracks.items():
            score = iou(box, info["box"])
            if score > best_iou:
                best_iou, best_id = score, pid

        if best_iou >= self.IOU_TH:
            self.tracks[best_id]["box"] = box
            return best_id

        pid = self.next_pid
        self.next_pid += 1
        self.tracks[pid] = {
            "box": box,
            "history": deque(maxlen=3),
            "decided": False
        }
        return pid

    # ===============================
    # Main loop
    # ===============================
    def run(self):
        cap = cv2.VideoCapture(6)

        while rclpy.ok() and self.active:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 10)

            for (x, y, w, h) in faces:
                pid = self.assign_id((x, y, w, h))
                track = self.tracks[pid]

                face = frame[y:y+h, x:x+w]
                age = self.classify(face)

                # 기록
                if age in ("age 01-10", "age 11-20"):
                    track["history"].append("minor")
                else:
                    track["history"].append("adult")

                # ===============================
                # 판정
                # ===============================
                if len(track["history"]) == 3 and not track["decided"]:

                    # ❌ 미성년자
                    if track["history"].count("minor") >= 2:
                        self.tts_queue.put("미성년자는 구매할 수 없습니다.")

                        done_msg = String()
                        done_msg.data = "CALC_DONE"
                        self.calc_done_pub.publish(done_msg)
                        self.get_logger().info("→ /task_done published: CALC_DONE (minor)")

                        track["decided"] = True
                        self.active = False
                        break

                    # ✅ 성인
                    if track["history"].count("adult") >= 2:
                        self.tts_queue.put("감사합니다.")

                        done_msg = String()
                        done_msg.data = "CALC_DONE"
                        self.calc_done_pub.publish(done_msg)
                        self.get_logger().info("→ /task_done published: CALC_DONE (adult)")

                        track["decided"] = True
                        self.active = False
                        break

                # 시각화
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"ID {pid} | {age} | {list(track['history'])}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

            cv2.imshow("FaceAge", frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.active = False


def main():
    rclpy.init()
    node = FaceAgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
