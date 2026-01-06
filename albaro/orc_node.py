import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from pathlib import Path

# 사용자가 작성한 클래스 임포트
from .STT import STT
from .keyword_extraction import ExtractKeyword

class Orchestrator(Node):
    def __init__(self):
        super().__init__("orc_node")

        self.api_key = os.getenv("OPENAI_API_KEY")

        self.get_logger().info(f"✅ API Key 강제 설정 완료")

        # ---------------------------------------------------------
        # 2. 부품 클래스 초기화 (api_key를 인자로 전달)
        # ---------------------------------------------------------
        # STT와 ExtractKeyword 클래스의 __init__이 api_key를 받도록 수정되어 있어야 합니다.
        self.stt = STT(self.api_key)
        self.extractor = ExtractKeyword(self.api_key)

        # ---------------------------------------------------------
        # 3. ROS2 통신 설정
        # ---------------------------------------------------------
        self.trigger_sub = self.create_subscription(Bool, "/start_orc", self.trigger_cb, 10)
        self.order_pub = self.create_publisher(String, "/order_item", 10)

        self.get_logger().info("✅ ORC Node Ready: Waiting for /start_orc...")

    def trigger_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().info("🎙️ 마이크 준비 대기 중 (5초)...")
            time.sleep(5) 

            try:
                self.get_logger().info("🔴 [녹음 시작] 주문을 말씀해 주세요!")
                # STT 클래스 내부에 정의된 speech2text() 호출
                text, wav_path = self.stt.speech2text()

                if text:
                    self.get_logger().info(f"🗣️ STT 인식 성공: {text}")
                    
                    # LLM 키워드 추출
                    result = self.extractor.extract_keyword(text)
                    
                    if result:
                        items, counts = result
                        order_dict = dict(zip(items, counts))
                        
                        # 결과 발행
                        msg_out = String()
                        msg_out.data = str(order_dict)
                        self.order_pub.publish(msg_out)
                        
                        self.get_logger().info(f"🚀 [발행 완료] 피킹 노드로 주문 전송: {msg_out.data}")
                    else:
                        self.get_logger().warn("⚠️ 주문 키워드 추출 실패 (사물/개수 없음)")
                else:
                    self.get_logger().warn("⚠️ 음성 인식 데이터 없음")

            except Exception as e:
                self.get_logger().error(f"❌ ORC 노드 실행 중 오류 발생: {e}")

            finally:
                # 임시 파일 삭제
                if 'wav_path' in locals() and wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except:
                        pass
                
                self.get_logger().info("⌛ ORC 작업 세션 종료. 다음 신호를 기다립니다.")

def main(args=None):
    rclpy.init(args=args)
    node = Orchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()