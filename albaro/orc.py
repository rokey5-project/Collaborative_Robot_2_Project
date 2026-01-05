import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dotenv import load_dotenv

from STT import STT
from keyword_extraction import ExtractKeyword

# =====================================================
# .env 로딩 (OPENAI API KEY)
# =====================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not loaded. Check .env path and content.")


# =====================================================
# Orchestrator Node (DB 제거 버전)
# =====================================================
class Orchestrator(Node):
    """
    역할:
    - STT 1회 실행
    - 키워드 추출
    - /order_item 토픽으로 결과 전달
    """

    def __init__(self):
        super().__init__("orc")

        self.stt = STT(OPENAI_API_KEY)
        self.extractor = ExtractKeyword()

        self.order_pub = self.create_publisher(
            String,
            "/order_item",
            10
        )

        self.get_logger().info("ORC node ready")

    def run_once(self):
        """
        음성 → 키워드 → /order_item publish
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

            msg = String()
            msg.data = str(order_dict)

            self.order_pub.publish(msg)
            self.get_logger().info(f"/order_item publish: {msg.data}")

        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
                self.get_logger().info(f"임시 wav 삭제: {wav_path}")


# =====================================================
# 실행
# =====================================================
def main():
    rclpy.init()
    node = Orchestrator()

    # orc는 1회 실행 후 끝나는 구조
    node.run_once()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
