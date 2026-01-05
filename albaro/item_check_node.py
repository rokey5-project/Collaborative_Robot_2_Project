import cv2
import os
import tempfile
import subprocess
import time
import threading
from pathlib import Path

from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from rclpy.executors import MultiThreadedExecutor  # MultiThreadedExecutor 임포트

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
        self.cass_found = False  # `cass_found` 상태를 따로 기록
        self.start_time = time.time()  # start_time을 0으로 초기화하거나 현재 시간으로 설정
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

        # TTS 멘트 로그 찍기
        self.get_logger().info(f"안내: {text}")

        # subprocess 실행 전에 audio_path가 제대로 생성됐는지 확인
        if os.path.exists(audio_path):
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            os.remove(audio_path)
        else:
            self.get_logger().error("TTS audio file creation failed!")

    def start_check_item(self, msg: Bool):
        if msg.data and not self.active:
            self.active = True
            self.cass_active = False
            self.shutdown_camera = False
            self.start_time = time.time()  # start_time을 여기서 초기화

    def cass_detected_once(self):
        # Record that cass was found
        self.cass_found = True

        # 카메라 종료
        self.shutdown_camera = True  # 카메라 종료 플래그를 True로 설정

        # `imshow` 창이 종료되었는지 확인하고 그 후에 FaceAge로 토픽을 발행
        if self.shutdown_camera:
            self.get_logger().info("카메라 종료 후, FaceAge로 토픽을 발행하였습니다.")
            # FaceAge로 넘기기 전에 Face 확인 요청 토픽 발행
            msg = Bool()
            msg.data = True
            self.pub_face.publish(msg)

        # 카메라는 이미 종료된 상태이므로 이후 프로세스를 멈추고 종료
        self.active = False


    def timeout_no_cass(self):
        # If no item is found, record it and process
        self.cass_found = False
        self.active = False

        # 발행 시 cass가 감지되지 않았을 때만 CALC_DONE을 발행
        done_msg = String()
        done_msg.data = "CALC_DONE"
        self.calc_done_pub.publish(done_msg)
        self.get_logger().info("→ /task_done published: CALC_DONE")


# ===============================
# Main
# ===============================
def display_frame(cap, ros_node, model):
    while rclpy.ok():
        # 5초 동안은 아무 판단 없이 계속 화면을 보여줍니다
        ret, frame = cap.read()

        # 프레임 읽기 실패 시 에러 로그
        if not ret:
            ros_node.get_logger().error("Failed to capture frame")
            break

        # YOLO 모델을 실행하여 물체를 감지합니다
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

        # cass를 감지한 경우 `cass_found`를 True로 기록
        if cass_found and not ros_node.cass_active:
            ros_node.cass_active = True
            ros_node.cass_detected_once()

        if not cass_found and ros_node.active:
            if time.time() - ros_node.start_time >= ros_node.TIMEOUT_SEC:
                ros_node.timeout_no_cass()

        # Show the frame for 5 seconds
        cv2.imshow("Item Check", frame)

        # 5초가 지나면 자동으로 imshow 종료
        if time.time() - ros_node.start_time >= 5.0:
            # cass의 감지 여부에 따라 TTS를 실행
            if ros_node.cass_found:
                ros_node.tts("인증이 필요한 상품입니다.")
            else:
                ros_node.tts("감사합니다.")
            
            # 2초 대기 후 화면 종료
            time.sleep(2)  # 2초 후에 강제 종료

            # 강제 종료 및 카메라 리소스 해제
            ros_node.timeout_no_cass()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    rclpy.init()
    ros_node = ItemCheckNode()

    model = YOLO('best.pt')

    cap = cv2.VideoCapture(8)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다")

    print("ItemCheck node running")

    # Start the frame display in a separate thread using MultiThreadedExecutor
    display_thread = threading.Thread(target=display_frame, args=(cap, ros_node, model))
    display_thread.start()

    # Create an executor to manage multiple threads
    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)

    try:
        # Execute the ROS node in multiple threads
        executor.spin()
    finally:
        # Wait for the display thread to finish
        display_thread.join()

        # Ensure that the display window is closed and resources are released
        cap.release()
        cv2.destroyAllWindows()
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
