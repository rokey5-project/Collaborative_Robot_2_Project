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


# =====================================================
# 1. .env 로딩 (안전하게)
# =====================================================
ENV_PATH = Path(__file__).resolve().parent / ".env"
print("ENV_PATH =", ENV_PATH, "exists =", ENV_PATH.exists())

load_dotenv(ENV_PATH, override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError(f"OPENAI_API_KEY not loaded. Checked: {ENV_PATH}")


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


class FaceAgeTTS:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)

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

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.cap = cv2.VideoCapture(8)
        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다")

        self.next_person_id = 0
        self.tracks = {}
        self.TRACK_IOU_TH = 0.4
        self.TRACK_TTL = 2.0

        self.tts_queue = Queue()
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()

    def tts_worker(self):
        while True:
            text = self.tts_queue.get()
            if text is None:
                break

            # 로그 찍기
            print(f"TTS 멘트: {text}")  # TTS 멘트 로그 출력

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                audio_path = f.name

            # 최신 스트리밍 방식 (경고 제거)
            with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text
            ) as response:
                response.stream_to_file(audio_path)

            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            os.remove(audio_path)
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
            score = iou(box, info["box"])
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
        self.tracks = {
            pid: info for pid, info in self.tracks.items()
            if now - info["last_seen"] <= self.TRACK_TTL
        }

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 10, minSize=(30, 30))

            for (x, y, w, h) in faces:
                pid = self.assign_id((x, y, w, h))
                face = frame[y:y+h, x:x+w]
                age_group = self.classify_image(face)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {pid} | {age_group}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if age_group in ("age 01-10", "age 11-20") and not self.tracks[pid]["warned"]:
                    self.tts_queue.put("미성년자는 구매할 수 없습니다.")
                    self.tracks[pid]["warned"] = True

            self.cleanup_tracks()

            # GUI 방어: 실패 시 원인 출력
            try:
                cv2.imshow("Age Detection", frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as e:
                print("OpenCV GUI error:", e)
                key = 255

            if key == 27:
                break

        self.tts_queue.put(None)
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = FaceAgeTTS(openai_api_key)
    app.run()
