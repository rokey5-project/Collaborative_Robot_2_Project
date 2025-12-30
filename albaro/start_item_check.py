import cv2
import os
import tempfile
import subprocess
import time
from pathlib import Path

from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from dotenv import load_dotenv
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
# ROS + Item Check Node
# ===============================
class ItemCheckNode(Node):
    def __init__(self):
        super().__init__('item_check_node')

        self.sub_start = self.create_subscription(
            Bool,
            '/start_item_check',
            self.start_cb,
            10
        )

        self.pub_face = self.create_publisher(
            Bool,
            '/need_face_check',
            10
        )

        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.active = False
        self.cass_active = False

        self.start_time = None
        self.TIMEOUT_SEC = 7.0

        self.get_logger().info("ItemCheck node initialized")

    def start_cb(self, msg: Bool):
        if msg.data and not self.active:
            self.active = True
            self.cass_active = False
            self.start_time = time.time()
            self.tts("물품을 올려주세요.")

    def tts(self, text: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            audio_path = f.name

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

    def cass_detected_once(self):
        self.tts("인증이 필요한 상품입니다.")

        msg = Bool()
        msg.data = True
        self.pub_face.publish(msg)

        self.active = False

    def timeout_no_cass(self):
        self.tts("감사합니다.")
        self.active = False


# ===============================
# Main
# ===============================
def main():
    rclpy.init()
    ros_node = ItemCheckNode()

    model = YOLO('best.pt')

    cap = cv2.VideoCapture(6)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다")

    print("ItemCheck node running")

    while rclpy.ok():
        rclpy.spin_once(ros_node, timeout_sec=0.01)

        ret, frame = cap.read()
        if not ret:
            break

        # -----------------------------
        # 🔥 active일 때만 감지 로직
        # -----------------------------
        #if ros_node.active:
        # if time.time() - ros_node.start_time >= ros_node.TIMEOUT_SEC:
        #     ros_node.timeout_no_cass()

        results = model.predict(source=frame, conf=0.7, verbose=False)
        result = results[0]
        boxes = result.boxes
        classes = result.names

    
        cass_found = False

        for box in boxes:
            cls_id = int(box.cls[0])
            label_name = classes[cls_id]

            if label_name == 'cass':
                cass_found = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0]) * 100
            label = f"{label_name} {conf:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        if cass_found and not ros_node.cass_active:
            ros_node.cass_active = True
            ros_node.cass_detected_once()

        if not cass_found:
            ros_node.cass_active = False

        # -----------------------------
        # ✅ imshow는 항상 실행
        # -----------------------------
        cv2.imshow("Item Check", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
