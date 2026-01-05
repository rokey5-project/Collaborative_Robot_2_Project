import cv2
import os
import tempfile
import subprocess
import time
from pathlib import Path

from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from dotenv import load_dotenv
from openai import OpenAI


# env 파일 로드
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")


class ItemCheckNode(Node):
    def __init__(self):
        super().__init__('item_check_node')

        self.item_check_sub = self.create_subscription(
            Bool,
            '/start_item_check',
            self.start_check_item,
            10
        )

        self.pub_face = self.create_publisher(
            Bool,
            '/need_face_check',
            10
        )

        # 🔥 calc 완료 신호
        self.calc_done_pub = self.create_publisher(
            String,
            '/task_done',
            10
        )

        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.active = False
        self.cass_active = False

        self.start_time = None
        self.TIMEOUT_SEC = 7.0

        # 🔥 FaceAge로 넘어갈 때 카메라 종료용 플래그 (추가)
        self.shutdown_camera = False

        self.get_logger().info("아이템 확인 진행중...")

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

    def start_check_item(self, msg: Bool):
        if msg.data and not self.active:
            self.active = True
            self.cass_active = False
            self.shutdown_camera = False
            self.start_time = time.time()
            self.tts("물품을 올려주세요.")

    def cass_detected_once(self):
        self.tts("인증이 필요한 상품입니다.")

        msg = Bool()
        msg.data = True
        self.pub_face.publish(msg)

        self.active = False

        # 🔥 FaceAge로 넘기고 ItemCheck 카메라 종료
        self.shutdown_camera = True

    def timeout_no_cass(self):
        self.tts("감사합니다.")
        self.active = False

        done_msg = String()
        done_msg.data = "CALC_DONE"
        self.calc_done_pub.publish(done_msg)
        self.get_logger().info("→ /task_done published: CALC_DONE")


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

        # 🔥 FaceAge로 넘어가는 순간 즉시 카메라 종료
        if ros_node.shutdown_camera:
            break

        ret, frame = cap.read()
        if not ret:
            break

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

        if not cass_found and ros_node.active:
            if time.time() - ros_node.start_time >= ros_node.TIMEOUT_SEC:
                ros_node.timeout_no_cass()

        cv2.imshow("Item Check", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
