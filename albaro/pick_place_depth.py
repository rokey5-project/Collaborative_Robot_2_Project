import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import threading
import numpy as np
import os
import json
import ast
import time
from ultralytics import YOLO
from scipy.spatial.transform import Rotation

# DSR 초기화용 모듈만 먼저 임포트
import DR_init

class SmartStoreNode(Node):
    def __init__(self, dsr_functions):
        super().__init__('smart_store_node')
        self.bridge = CvBridge()

        # 1. DSR 함수 바인딩
        self.get_current_posx = dsr_functions['get_current_posx']
        self.movel = dsr_functions['movel']
        self.movej = dsr_functions['movej']
        self.wait = dsr_functions['wait']
        self.posx = dsr_functions['posx']
        self.posj = dsr_functions['posj']
        self.set_do = dsr_functions['set_digital_output']

        # 2. YOLO 및 카메라 설정
        self.model = YOLO(os.path.join(os.getcwd(), "best.pt"))
        self.intrinsics = {"fx": 606.33, "fy": 605.55, "ppx": 323.07, "ppy": 238.41}
        file_path = "/home/rokey/albaro/T_gripper2camera.npy"
        try:
            self.gripper2cam = np.load(file_path)
        except:
            self.get_logger().error("T_gripper2camera.npy 파일을 찾을 수 없습니다!")
            self.gripper2cam = np.eye(4)

        # 3. 상태 관리
        self.order_queue = []
        self.is_robot_busy = False
        self.last_frame = None
        self.last_depth_frame = None

        # 4. 품목별 경로 설정 (2단계 approach + 매대 위치)
        self.routes = {
            "cocacola": {
                "approach": [
                    self.posj(-27.28, 19.27, 106.93, -40.64, -44.78, 120.46),
                    self.posj(-27.31, 42.43, 112.49, -29.58, -68.55, 100.82)
                ],
                "stand": self.stand1
            },
            "fanta": {
                "approach": [
                    self.posj(-11.15, 14.67, 112.70, -19.26, -38.81, 105.83),
                    self.posj(-11.18, 39.73, 118.82, -12.84, -68.88, 95.27)
                ],
                "stand": self.stand2
            },
            "sprite": {
                "approach": [
                    self.posj(6.94, 14.47, 113.12, 9.29, -37.74, 83.75),
                    self.posj(6.94, 39.72, 119.20, 6.08, -68.81, 88.89)
                ],
                "stand": self.stand3
            },
            "cass": {
                "approach": [
                    self.posj(24.95, 18.12, 108.48, 37.93, -42.36, 60.25),
                    self.posj(24.97, 41.74, 114.15, 26.66, -67.40, 79.24)
                ],
                "stand": self.stand4
            }
        }

        # 5. ROS2 구독 및 퍼블리셔
        self.inventory_sub = self.create_subscription(String, '/detect_info', self.inventory_callback, 10)
        self.order_sub = self.create_subscription(String, '/order_item', self.order_callback, 10)
        self.img_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.vision_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)


    # --- 콜백 함수부 ---
    def depth_callback(self, msg):
        self.last_depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    def inventory_callback(self, msg):
        if self.is_robot_busy:
            return
        elif len(self.order_queue) == 0:
            data_str = msg.data.replace("'", '"')
            inventory = json.loads(data_str)

            for item, count in inventory.items():
                if count == 0 and item in self.routes:
                    self.get_logger().info(f"재고 부족 감지! {item}을 3개 보충합니다.")
                    for _ in range(3):
                        self.order_queue.append(item)
        else:
            threading.Thread(target=self.delivery_process, args=(self.order_queue[0],), daemon=True).start()

    def order_callback(self, msg):
        try:
            order_dict = ast.literal_eval(msg.data)
            print("order callback")
            for item, count in order_dict.items():
                for _ in range(count):
                    self.order_queue.append(item)
                print(f"{item, count}")

            threading.Thread(target=self.delivery_process, args=(self.order_queue[0],), daemon=True).start()

        except Exception as e:
            self.get_logger().error(f"주문 파싱 오류: {e}")

    def vision_callback(self, msg):
        # print("get topic")
        self.last_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 큐에 주문이 있는지 확인
        if not self.order_queue:
            return # 주문이 없으면 아무것도 안 함

        # 로봇이 동작 중인지 확인
        if self.is_robot_busy:
            # self.get_logger().info("로봇이 바쁩니다...") # 너무 자주 찍히면 주석 처리
            return

        # Depth 데이터가 들어왔는지 확인
        if self.last_depth_frame is None:
            self.get_logger().warn("주문은 있으나 Depth 프레임이 아직 없습니다!")
            return

    def delivery_process(self, label):
        self.get_logger().info(f"--- [스레드 시작] 타겟: {label} ---")
        try:
            self.is_robot_busy = True

            # 1. 경로 존재 여부 확인
            if label not in self.routes:
                self.get_logger().error(f"경로 데이터에 '{label}'이 없습니다!")
                return

            # 2. 창고 접근 동작
            self.get_logger().info("창고 접근 중...")
            for pos in self.routes[label]['approach']:
                self.movej(pos, vel=30, acc=30, ra=1)
            self.wait(1.0)

            # 3. 객체 인식
            self.get_logger().info(f"YOLO 인식 시도: {label}")
            results = self.model.predict(source=self.last_frame, conf=0.75, verbose=False)

            target_box = None
            for box in results[0].boxes:
                if results[0].names[int(box.cls[0])] == label:
                    target_box = box
                    break

            if target_box is not None:
                x1, y1, x2, y2 = map(int, target_box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 박스 그리기
                cv2.rectangle(self.last_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 클래스 및 확률 표시
                cv2.putText(self.last_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                self.get_logger().info(f"좌표 감지 ({cx}, {cy}), 피킹 실행")
                self.get_logger().info(f"")
                self.execute_vision_pick(cx, cy)

                self.get_logger().info("매대로 이동 중...")
                self.routes[label]['stand']()

                # 4. 완료 후 큐에서 제거
                self.order_queue.pop(0)
                self.get_logger().info(f"--- [스레드 종료] {label} 처리 완료 ---")
            else:
                self.get_logger().warn(f"화면에서 {label}을 찾지 못했습니다.")
                self.order_queue = [word for word in self.order_queue if word != label]

                self.movel(self.posx(0, 0, 300, 0, 0, 0),vel=60, acc=60, mod=1)
                self.movej(self.posj(0, 0, 90, 0, 90, 0), vel=30, acc=30,  ra=1)

        except Exception as e:
            self.get_logger().error(f"delivery_process 실행 중 치명적 오류: {e}")
        finally:
            self.is_robot_busy = False # 에러가 나더라도 반드시 False로 복구


    def execute_vision_pick(self, cx, cy):
        """Depth 기반 정밀 피킹"""
        try:
            z_mm = self.last_depth_frame[cy, cx]
            if z_mm > 0:
                self.get_logger().error("Depth 값이 0입니다. 피킹을 취소합니다.")
                # 카메라 좌표 계산
                cam_x = (cx - self.intrinsics["ppx"]) * z_mm / self.intrinsics["fx"]
                cam_y = (cy - self.intrinsics["ppy"]) * z_mm / self.intrinsics["fy"]
                cam_coords = np.array([cam_x, cam_y, z_mm, 1.0])

                # 베이스 좌표 변환
                curr_pos = self.get_current_posx()[0]
                R = Rotation.from_euler("ZYZ", curr_pos[3:], degrees=True).as_matrix()
                T = np.eye(4); T[:3,:3]=R; T[:3,3]=curr_pos[:3]
                target_pos = (T @ self.gripper2cam @ cam_coords)[:3]


                pick = self.posx([target_pos[0]+70, curr_pos[1], curr_pos[2], curr_pos[3], curr_pos[4], curr_pos[5]])

                self.get_logger().info(f"상세 좌표: {target_pos}")



                self.get_logger().info("2. Pick 지점(L) 이동")
                self.movel(pick, vel=20, acc=20)


            self.wait(0.5)
            self.get_logger().info("3. 그리퍼 닫기")
            self.set_do(1, 1); self.set_do(2, 0); self.wait(1.0)

            self.get_logger().info("4. 물체 들어올리기(L)")
            # self.movel(approach, vel=40, acc=40)
            self.movel(self.posx(0, 0, 100, 0, 0, 0),vel=60, acc=60, mod=1)

            self.get_logger().info("5. 중간 안전지대 이동(J)")
            # 관절 포즈는 movej로 이동
            self.movej(self.posj(-21.00, -7.23, 102.58, 1.29, -6.27, 88.93), vel=30, acc=30, ra=1)
            self.movej(self.posj(-34.18, -14.68, 94.26, 1.29, 8.00, 88.93), vel=30, acc=30, ra=1)

            self.get_logger().info("피킹 및 중간 동작 완료")

        except Exception as e:
            self.get_logger().error(f"execute_vision_pick 내부 오류: {e}")
            raise e # 상위 함수로 에러 전달


    # --- 매대별 내려놓기 동작 ---
    def stand1(self):
        self.movej(self.posj(-30.25, 38.45, 23.09, -11.04, 110.93, 53.81), vel=60, acc=60, ra=1)
        self.movej(self.posj(-32.28, 36.26, 40.68, -10.03, 95.72, 54.79), vel=60, acc=60, ra=1)
        self.set_do(1, 0); self.set_do(2, 1); self.wait(1.0) # 그리퍼 열기
        self.movel(self.posx(0, 0, 30, 0, 0, 0),vel=60, acc=60, mod=1)
        self.movej(self.posj(-9.08, 13.64, 71.30, -3.78, 93.99, 80.38),vel=60, acc=60, ra=1)


    def stand2(self):
        self.movej(self.posj(-41.43, 13.20, 59.31, -8.71, 98.39, 45.49), vel=60, acc=60, ra=1)
        self.movej(self.posj(-43.94, 13.08, 74.78, -8.45, 83.20, 45.33), vel=60, acc=60, ra=1)
        self.set_do(1, 0); self.set_do(2, 1); self.wait(1.0) # 그리퍼 열기
        self.movej(self.posj(-41.43, 13.20, 59.31, -8.71, 98.39, 45.49), vel=60, acc=60, ra=1)
        self.movej(self.posj(-9.08, 13.64, 71.30, -3.78, 93.99, 80.38),vel=60, acc=60, ra=1)

    def stand3(self):
        self.movej(self.posj(-61.71, -2.94, 75.61, -5.01, 95.74, 26.26), vel=60, acc=60, ra=1)
        self.movej(self.posj(-64.04, -2.29, 89.98, -4.85, 81.00, 25.28), vel=60, acc=60, ra=1)
        self.set_do(1, 0); self.set_do(2, 1); self.wait(1.0) # 그리퍼 열기
        self.movej(self.posj(-61.71, -2.94, 75.61, -5.01, 95.74, 26.26), vel=60, acc=60, ra=1)
        self.movej(self.posj(-9.08, 13.64, 71.30, -3.78, 93.99, 80.38),vel=60, acc=60, ra=1)

    def stand4(self):
        self.movej(self.posj(-92.90, -8.97, 82.16, 1.34, 94.19, -3.99), vel=60, acc=60, ra=1)
        self.movej(self.posj(-94.38, -6.91, 95.10, 1.50, 79.83, -5.67), vel=60, acc=60, ra=1)
        self.set_do(1, 0); self.set_do(2, 1); self.wait(1.0) # 그리퍼 열기
        self.movej(self.posj(-92.90, -8.97, 82.16, 1.34, 94.19, -3.99), vel=60, acc=60, ra=1)
        self.movej(self.posj(-9.08, 13.64, 71.30, -3.78, 93.99, 80.38),vel=60, acc=60, ra=1)

# --- 메인 함수 (에러 해결용 초기화 구조) ---
def main(args=None):
    rclpy.init(args=args)

    # 1. 로봇 노드 생성
    robot_node = rclpy.create_node("dsr_vision_node", namespace="dsr01")

    # 2. DSR 라이브러리 초기화 (중요: 임포트 전 설정)
    DR_init.__dsr__node = robot_node
    DR_init.__dsr__id = "dsr01"
    DR_init.__dsr__model = "m0609"

    # 3. 노드 생성 '후'에 라이브러리 임포트 (NoneType 에러 방지)
    try:
        import DSR_ROBOT2 as dsr
        from DR_common2 import posx, posj

        dsr_funcs = {
            'get_current_posx': dsr.get_current_posx,
            'movej': dsr.movej,
            'movel': dsr.movel,
            'wait': dsr.wait,
            'posx': posx,
            'posj': posj,
            'set_digital_output': dsr.set_digital_output
        }
    except ImportError as e:
        print(f"DSR Import Error: {e}")
        return

    # 4. 스마트 스토어 노드 생성
    node = SmartStoreNode(dsr_funcs)

    # 5. 멀티스레드 실행기 설정
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(robot_node)
    executor.add_node(node)

    # 스핀 스레드 시작
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    node.get_logger().info("--- 노드가 실행되었습니다. 토픽 대기 중 ---")

    # 6. OpenCV 메인 루프
    try:
        while rclpy.ok():
            if node.last_frame is not None:
                cv2.imshow('Smart Store Picking System', node.last_frame)
                if cv2.waitKey(1) == 27: break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()