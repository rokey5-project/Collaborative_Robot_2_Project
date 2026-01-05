import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from dotenv import load_dotenv

from STT import STT
from keyword_extraction import ExtractKeyword

# .env 파일 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not loaded. Check .env path and content.")


class OrderOrchestrator(Node):
    """
    역할:
    - /start_orc 수신
    - STT → LLM 키워드 추출
    - /order_item publish
    - 로봇 동작 시작 신호를 StateManager에 전달
    """

    def __init__(self):
        super().__init__("order_orchestrator")

        # STT / Keyword Extractor
        self.stt = STT(OPENAI_API_KEY)
        self.extractor = ExtractKeyword()

        # Publisher
        self.order_pub = self.create_publisher(String, "/order_item", 10)
        self.robot_start_pub = self.create_publisher(Bool, "/robot_start", 10)

        # Subscriber
        self.wakeup_sub = self.create_subscription(
            Bool, "/start_orc", self.orc_callback, 10
        )

        self.get_logger().info("OrderOrchestrator node started")

    def run_once(self):
        """
        음성 1회 → 주문 1회 publish
        """
        text, wav_path = self.stt.speech2text()

        try:
            if not text:
                self.get_logger().warn("STT 실패 또는 무음")
                return

            result = self.extractor.extract_keyword(text)
            if not result:
                self.get_logger().warn("키워드 추출 실패")
                return

            items, counts = result
            order_dict = dict(zip(items, counts))

            # 주문 publish
            order_msg = String()
            order_msg.data = str(order_dict)
            self.order_pub.publish(order_msg)
            self.get_logger().info(f"주문 전송: {order_msg.data}")

            # 🔥 로봇 시작 신호 (StateManager → ROBOT_BUSY)
            self.robot_start_pub.publish(Bool(data=True))
            self.get_logger().info("robot_start signal sent")

        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
                self.get_logger().info(f"임시 wav 삭제: {wav_path}")

    def orc_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("start_orc received")
            self.run_once()


def main(args=None):
    rclpy.init(args=args)
    node = OrderOrchestrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
