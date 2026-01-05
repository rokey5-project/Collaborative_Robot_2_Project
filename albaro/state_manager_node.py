import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


class StateManager(Node):
    def __init__(self):
        super().__init__("state_manager")

        # -------------------------
        # 상태 (실제 제어용)
        # -------------------------
        self.calc_active = False   # item_check / face_age
        self.pick_active = False   # orc / robot

        # -------------------------
        # Subscriber
        # -------------------------
        self.intent_sub = self.create_subscription(
            String,
            "/wakeup_intent",
            self.intent_cb,
            10
        )

        self.task_done_sub = self.create_subscription(
            String,
            "/task_done",
            self.task_done_cb,
            10
        )

        # -------------------------
        # Publisher (🔥 트리거용)
        # -------------------------
        self.start_item_pub = self.create_publisher(
            Bool,
            "/start_item_check",
            10
        )

        self.start_orc_pub = self.create_publisher(
            Bool,
            "/start_orc",
            10
        )

        self.get_logger().info(
            "StateManager started | calc=IDLE, pick=IDLE"
        )

    # -------------------------
    # Wakeup Intent 처리
    # -------------------------
    def intent_cb(self, msg: String):
        intent = msg.data.strip().lower()

        self.get_logger().info(
            f"[INTENT] {intent} | "
            f"calc_active={self.calc_active}, pick_active={self.pick_active}"
        )

        # -------------------------
        # CALC (item_check)
        # -------------------------
        if intent == "calc":
            if self.calc_active:
                self.get_logger().warn("calc already running → ignored")
                return

            # 🔥 트리거 1회 발행
            self.start_item_pub.publish(Bool(data=True))
            self.calc_active = True

            self.get_logger().info("→ ITEM_CHECK triggered")
            return

        # -------------------------
        # PICK (orc / robot)
        # -------------------------
        if intent == "pick":
            if self.pick_active:
                self.get_logger().warn("pick already running → ignored")
                return

            # 🔥 트리거 1회 발행
            self.start_orc_pub.publish(Bool(data=True))
            self.pick_active = True

            self.get_logger().info("→ ORC triggered")
            return

        self.get_logger().warn(f"Unknown intent: {intent}")

    # -------------------------
    # 작업 완료 신호
    # -------------------------
    def task_done_cb(self, msg: String):
        task = msg.data.strip().upper()
        self.get_logger().info(f"[TASK DONE] {task}")

        if task == "CALC_DONE":
            self.calc_active = False
            self.get_logger().info("→ calc reset to IDLE")

        elif task == "PICK_DONE":
            self.pick_active = False
            self.get_logger().info("→ pick reset to IDLE")

        else:
            self.get_logger().warn(f"Unknown task_done: {task}")


def main():
    rclpy.init()
    node = StateManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
