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

from openai import OpenAI

class ItemCheckNode(Node):
    def __init__(self):
        super().__init__('item_check_node')

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # API KEY 체크
        if not OPENAI_API_KEY:
            self.get_logger().error(f"❌ API KEY 로드 실패!")
            raise RuntimeError("OPENAI_API_KEY를 .env 파일에서 찾을 수 없습니다.")

        # 구독자: /start_item_check 토픽 수신 시 동작
        self.item_check_sub = self.create_subscription(
            Bool, '/start_item_check', self.start_check_callback, 10
        )

        # 발행자: FaceAge 노드 호출 또는 작업 완료 보고
        self.pub_face = self.create_publisher(Bool, '/need_face_check', 10)
        self.calc_done_pub = self.create_publisher(String, '/task_done', 10)

        self.client = OpenAI(api_key=OPENAI_API_KEY)

        # 상태 제어 변수
        self.trigger_received = False
        self.cass_found = False
        self.TIMEOUT_SEC = 5.0

        self.get_logger().info("--- ItemCheckNode Online: Waiting for Signal ---")

    def start_check_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("🚀 검사 시작 신호 수신!")
            self.trigger_received = True

    def tts(self, text: str):
        try:
            self.get_logger().info(f"🔊 TTS 안내: {text}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                audio_path = f.name

            with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts", voice="alloy", input=text
            ) as response:
                response.stream_to_file(audio_path)

            if os.path.exists(audio_path):
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", audio_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                os.remove(audio_path)
        except Exception as e:
            self.get_logger().error(f"❌ TTS 오류: {e}")

    def send_final_topic(self):
        if self.cass_found:
            msg = Bool()
            msg.data = True
            self.pub_face.publish(msg)
            self.get_logger().info("✅ [결과] 성인 상품 감지 -> FaceAge로 바통 터치")
        else:
            done_msg = String()
            done_msg.data = "CALC_DONE"
            self.calc_done_pub.publish(done_msg)
            self.get_logger().info("✅ [결과] 일반 상품 -> 시스템 리셋")

def main():
    rclpy.init()
    ros_node = ItemCheckNode()

    # YOLO 모델 로드 (경로 에러 방지를 위해 절대 경로 사용 추천)
    model_path = Path("/home/rokey/albaro/src/albaro/albaro/best.pt")
    if not model_path.exists():
        model_path = "best.pt" # 못 찾으면 현재 위치 시도

    model = YOLO(str(model_path))

    # ROS 통신용 스레드 분리
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    try:
        while rclpy.ok():
            if not ros_node.trigger_received:
                time.sleep(0.1)
                continue

            cap = cv2.VideoCapture(8)
            if not cap.isOpened():
                ros_node.get_logger().error("❌ 카메라 8번 점유 실패!")
                ros_node.trigger_received = False
                continue

            ros_node.get_logger().info("📸 감지 시작 (5초간 유지)")
            ros_node.cass_found = False
            start_time = time.time()

            while rclpy.ok():
                ret, frame = cap.read()
                if not ret: break

                elapsed = time.time() - start_time
                results = model.predict(source=frame, conf=0.5, verbose=False)

                for box in results[0].boxes:
                    label = results[0].names[int(box.cls[0])]
                    if label.lower() == 'cass':
                        ros_node.cass_found = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                remaining = max(0, ros_node.TIMEOUT_SEC - elapsed)
                cv2.putText(frame, f"Checking... {remaining:.1f}s", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow("Item Check", frame)
                cv2.waitKey(1)

                if elapsed >= ros_node.TIMEOUT_SEC:
                    break

            cap.release()
            cv2.destroyAllWindows()

            for _ in range(10): cv2.waitKey(1)
            time.sleep(0.5)

            if ros_node.cass_found:
                ros_node.tts("인증이 필요한 상품입니다.")
            else:
                ros_node.tts("감사합니다.")

            ros_node.send_final_topic()
            ros_node.trigger_received = False

    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            ros_node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()