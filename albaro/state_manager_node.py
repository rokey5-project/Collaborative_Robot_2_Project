import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import time

class StateManager(Node):
    def __init__(self):
        super().__init__("state_manager")

        self.is_busy = False
        self.current_task = None

        self.intent_sub = self.create_subscription(
            String, "/wakeup_intent", self.intent_cb, 10
        )

        # 각 작업 노드들로부터 완료 신호 수신
        self.task_done_sub = self.create_subscription(
            String, "/task_done", self.task_done_cb, 10
        )

        self.start_item_pub = self.create_publisher(Bool, "/start_item_check", 10)
        self.start_orc_pub = self.create_publisher(Bool, "/start_orc", 10)

        # WakeupNode를 강제 종료시켜 마이크 자원을 회수하는 토픽
        self.kill_wakeup_pub = self.create_publisher(Bool, "/kill_wakeup", 10)

        self.get_logger().info("========================================")
        self.get_logger().info("   StateManager Online | System IDLE    ")
        self.get_logger().info("========================================")

    def intent_cb(self, msg: String):
        # 대소문자 구분 없이 처리하기 위해 소문자로 통일
        intent = msg.data.strip().lower()

        if self.is_busy:
            self.get_logger().warn(f"🚨 시스템 바쁨: {self.current_task} 수행 중. '{intent}' 명령 무시.")
            return

        # 1. 계산(CALC) 모드
        if intent == "calc":
            self.is_busy = True
            self.current_task = "CALC"
            self.get_logger().info("🛒 [CALC 감지] 계산 프로세스 신호를 보냅니다.")
            self.start_item_pub.publish(Bool(data=True))

        # 2. 주문(PICK) 모드
        elif intent == "pick":
            self.is_busy = True
            self.current_task = "PICK"

            self.get_logger().warn("💀 [PICK 감지] WakeupNode 종료 명령 송신 (마이크 해제)")
            # WakeupNode에게 즉시 종료 신호 전송
            self.kill_wakeup_pub.publish(Bool(data=True))

            # 마이크 장치가 OS에 완전히 반환될 수 있도록 대기 (중요)
            time.sleep(2.0)

            self.get_logger().info("🤖 Orchestrator(STT) 가동 신호를 보냅니다.")
            self.start_orc_pub.publish(Bool(data=True))

        else:
            self.get_logger().info(f"❓ 정의되지 않은 명령: {intent}")

    def task_done_cb(self, msg: String):
        task_status = msg.data.strip().upper()

        if task_status in ["CALC_DONE", "PICK_DONE"]:
            self.get_logger().info("----------------------------------------")
            self.get_logger().info(f"✅ [작업 완료 보고] {task_status}")
            self.get_logger().info(f"🔄 시스템 리셋: {self.current_task} → IDLE")
            self.get_logger().info("----------------------------------------")

            self.is_busy = False
            self.current_task = None
        else:
            self.get_logger().error(f"❌ 알 수 없는 완료 신호: {task_status}")

def main(args=None):
    rclpy.init(args=args)
    node = StateManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("StateManager 종료 중...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()