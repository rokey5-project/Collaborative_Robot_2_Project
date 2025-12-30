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
        self.model = YOLO(os.path.join(os.getcwd(), 'src', 'Collaborative_Robot_2_Project', "best.pt"))
        self.intrinsics = {"fx": 606.33, "fy": 605.55, "ppx": 323.07, "ppy": 238.41}
        file_path = "/home/suan/rokey_ws/src/Collaborative_Robot_2_Project/T_gripper2camera.npy"
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
        """재고 정보(JSON)를 파싱하여 0개인 품목을 주문 큐에 추가"""
        if self.is_robot_busy or len(self.order_queue) > 0:
            return
        try:
            # JSON 파싱 (작은따옴표 문제 해결을 위해 replace 사용)
            data_str = msg.data.replace("'", '"')
            inventory = json.loads(data_str)
            for item, count in inventory.items():
                if count == 0 and item in self.routes:
                    self.get_logger().info(f"재고 부족 감지! {item}을 3개 보충합니다.")
                    for _ in range(3):
                        self.order_queue.append(item)
                    break 
        except Exception as e:
            pass

    def order_callback(self, msg):
        try:
            order_dict = ast.literal_eval(msg.data)
            print("order callback")
            for item, count in order_dict.items():
                for _ in range(count):
                    self.order_queue.append(item)
                
        except Exception as e:
            self.get_logger().error(f"주문 파싱 오류: {e}")

    # def vision_callback(self, msg):
    #     self.last_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    #     if not self.order_queue or self.is_robot_busy or self.last_depth_frame is None:
    #         return
        
    #     # 현재 처리해야 할 타겟
    #     target_label = self.order_queue[0]
        
    #     # 이전 코드의 안정적인 스레딩 방식 사용
    #     self.is_robot_busy = True
    #     threading.Thread(target=self.delivery_process, args=(target_label,)).start()


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

        # 여기까지 왔다면 조건 만족
        target_label = self.order_queue[0]
        self.get_logger().info(f"조건 만족! {target_label} 배송 스레드를 시작합니다.")
        
        self.is_robot_busy = True # 스레드 시작 전 미리 선점
        threading.Thread(target=self.delivery_process, args=(target_label,), daemon=True).start()


    # --- 메인 실행 로직 ---
    # def delivery_process(self, label):
    #     self.get_logger().info(f"--- {label} 보충 프로세스 시작 ---")
    #     try:
    #         # 1. 창고 접근 (2단계 movej)
    #         for pos in self.routes[label]['approach']:
    #             self.movej(pos, vel=60, acc=60, ra=1)
    #         self.wait(1.5)

    #         # 2. YOLO 인식 (정확한 타겟 조준)
    #         results = self.model.predict(source=self.last_frame, conf=0.5, verbose=False)
    #         target_box = None
    #         for box in results[0].boxes:
    #             if results[0].names[int(box.cls[0])] == label:
    #                 target_box = box
    #                 break

    #         if target_box is None:
    #             self.get_logger().warn(f"{label}을 찾을 수 없습니다.")
    #         else:
    #             # 3. 실시간 피킹 실행
    #             x1, y1, x2, y2 = map(int, target_box.xyxy[0])
    #             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    #             self.execute_vision_pick(cx, cy)
                
    #             # 4. 매대 이동 및 내려놓기
    #             self.routes[label]['stand']()
    #             self.get_logger().info(f"{label} 보충 완료")

    #         self.order_queue.pop(0)

    #     except Exception as e:
    #         self.get_logger().error(f"실행 중 오류: {e}")
    #     finally:
    #         self.is_robot_busy = False

    def delivery_process(self, label):
        self.get_logger().info(f"--- [스레드 시작] 타겟: {label} ---")
        try:
            # 1. 경로 존재 여부 확인
            if label not in self.routes:
                self.get_logger().error(f"경로 데이터에 '{label}'이 없습니다!")
                return

            # 2. 창고 접근 동작
            self.get_logger().info("창고 접근 중...")
            for pos in self.routes[label]['approach']:
                self.movej(pos, vel=60, acc=60, ra=1)
            self.wait(1.0)

            # 3. 객체 인식
            self.get_logger().info(f"YOLO 인식 시도: {label}")
            results = self.model.predict(source=self.last_frame, conf=0.5, verbose=False)
            
            target_box = None
            for box in results[0].boxes:
                if results[0].names[int(box.cls[0])] == label:
                    target_box = box
                    break

            if target_box is not None:
                x1, y1, x2, y2 = map(int, target_box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.get_logger().info(f"좌표 감지 ({cx}, {cy}), 피킹 실행")
                self.execute_vision_pick(cx, cy)
                
                self.get_logger().info("매대로 이동 중...")
                self.routes[label]['stand']()
            else:
                self.get_logger().warn(f"화면에서 {label}을 찾지 못했습니다.")

            # 4. 완료 후 큐에서 제거
            self.order_queue.pop(0)
            self.get_logger().info(f"--- [스레드 종료] {label} 처리 완료 ---")

        except Exception as e:
            self.get_logger().error(f"delivery_process 실행 중 치명적 오류: {e}")
        finally:
            self.is_robot_busy = False # 에러가 나더라도 반드시 False로 복구



    def execute_vision_pick(self, cx, cy):
        """Depth 기반 정밀 피킹"""
        z_mm = self.last_depth_frame[cy, cx]
        if z_mm <= 0: return

        # 카메라 좌표 계산
        cam_x = (cx - self.intrinsics["ppx"]) * z_mm / self.intrinsics["fx"]
        cam_y = (cy - self.intrinsics["ppy"]) * z_mm / self.intrinsics["fy"]
        cam_coords = np.array([cam_x, cam_y, z_mm, 1.0])

        # 베이스 좌표 변환
        curr_pos = self.get_current_posx()[0]
        R = Rotation.from_euler("ZYZ", curr_pos[3:], degrees=True).as_matrix()
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=curr_pos[:3]
        target_pos = (T @ self.gripper2cam @ cam_coords)[:3]

        # 모션 시퀀스
        approach = self.posx([target_pos[0], target_pos[1], target_pos[2]+100, curr_pos[3], curr_pos[4], curr_pos[5]])
        pick = self.posx([target_pos[0], target_pos[1], target_pos[2]-40, curr_pos[3], curr_pos[4], curr_pos[5]])
        
        self.movel(approach, vel=60)
        self.movel(pick, vel=30)
        self.set_do(1, 1); self.set_do(2, 0); self.wait(1.0) # 그리퍼 닫기
        self.movel(approach, vel=60)

    # --- 매대별 내려놓기 동작 ---
    def stand1(self):
        self.movej(self.posj(-30.25, 38.45, 23.09, -11.04, 110.93, 53.81), ra=1)
        self.set_do(1, 0); self.set_do(2, 1); self.wait(1.0) # 그리퍼 열기
        self.movel(self.posx(0, 0, 100, 0, 0, 0), mod=1)

    def stand2(self):
        # stand2, 3, 4는 좌표에 맞춰 구현
        self.stand1()

    def stand3(self): self.stand1()
    def stand4(self): self.stand1()

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

# def main(args=None):
#     rclpy.init(args=args)
    
#     # 1. 노드 생성 (네임스페이스를 명시적으로 부여)
#     # 실제 환경에서 토픽들이 /dsr01/order_item 인지 /order_item 인지 확인이 필요합니다.
#     node = SmartStoreNode(None) # 임시 생성
    
#     # 2. DSR 초기화 (이전 해결 방식 유지)
#     DR_init.__dsr__node = node
#     DR_init.__dsr__id = "dsr01"
#     DR_init.__dsr__model = "m0609"

#     import DSR_ROBOT2 as dsr
#     from DR_common2 import posx, posj
    
#     dsr_funcs = {
#         'get_current_posx': dsr.get_current_posx,
#         'movej': dsr.movej, 'movel': dsr.movel, 'wait': dsr.wait,
#         'posx': posx, 'posj': posj, 'set_digital_output': dsr.set_digital_output
#     }

#     # 3. 실제 기능을 가진 노드로 다시 세팅
#     node.get_current_posx = dsr_funcs['get_current_posx']
#     node.movel = dsr_funcs['movel']
#     node.movej = dsr_funcs['movej']
#     node.wait = dsr_funcs['wait']
#     node.posx = dsr_funcs['posx']
#     node.posj = dsr_funcs['posj']
#     node.set_do = dsr_funcs['set_digital_output']

#     # 4. 실행기 설정 (하나의 노드만 돌려도 충분합니다)
#     executor = MultiThreadedExecutor()
#     executor.add_node(node)

#     # 로그를 통해 토픽 연결 확인
#     node.get_logger().info("--- 노드가 실행되었습니다. 토픽 대기 중 ---")

#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

if __name__ == '__main__':
    main()