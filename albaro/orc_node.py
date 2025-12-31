import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dotenv import load_dotenv

from STT import STT
from keyword_extraction import ExtractKeyword
from std_msgs.msg import Bool

# .env 파일 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not loaded. Check .env path and content.")

class OrderOrchestrator(Node):
    """
    역할:
    - STT로 음성 입력 수집
    - LLM 키워드 추출
    - {'cocacola': 2, 'sprite': 1} 형태로
      /order_item 토픽 publish
    """

    def __init__(self):
        super().__init__("order_orchestrator")

        # STT / Keyword Extractor
        self.stt = STT(OPENAI_API_KEY)  # openai_api_key를 STT에 전달
        self.extractor = ExtractKeyword()

        # pick_place_depth.py 쪽에서 받을 토픽
        self.order_pub = self.create_publisher(String, "/order_item", 10)
        self.wakeup_sub = self.create_subscription(Bool, "/start_orc", self.orc_callback, 10)

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

            # SmartStoreNode / OrderListenerNode가 기대하는 형태
            # 예: {'cocacola': 2, 'sprite': 1}
            order_dict = dict(zip(items, counts))

            msg = String()
            msg.data = str(order_dict)   # ast.literal_eval 대응
            self.order_pub.publish(msg)

            self.get_logger().info(f"주문 전송: {msg.data}")

        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
                self.get_logger().info(f"임시 wav 삭제: {wav_path}")

    def orc_callback(self, msg):
        if msg.data:
            self.run_once()



def main(args=None):
    rclpy.init(args=args)
    node = OrderOrchestrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
