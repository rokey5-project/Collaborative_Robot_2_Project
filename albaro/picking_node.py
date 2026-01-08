import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import threading
import numpy as np
import time
from ultralytics import YOLO
from scipy.spatial.transform import Rotation
from .onrobot import RG

# DSR 초기화용
import DR_init

class VisionPickingNode(Node):
    def __init__(self, dsr_functions):
        super().__init__('picking_node')
        self.bridge = CvBridge()

        # DSR 함수 바인딩
        self.get_current_posx = dsr_functions['get_current_posx']
        self.movel = dsr_functions['movel']
        self.movej = dsr_functions['movej']
        self.wait = dsr_functions['wait']
        self.posx = dsr_functions['posx']

        # YOLOv8 모델 로드
        self.model = YOLO('best.pt')

        # 구독 설정 (이미지 및 Depth)
        self.img_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.vision_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # 카메라 파라미터 및 캘리브레이션 데이터 (실제 값으로 확인 필요)
        self.intrinsics = {"fx": 606.33, "fy": 605.55, "ppx": 323.07, "ppy": 238.41}
        self.gripper2cam = np.load("T_gripper2camera.npy")

        # 그리퍼 초기화
        self.gripper = RG("rg2", "192.168.1.1", "502")

        # 상태 변수
        self.last_frame = None
        self.last_depth_frame = None
        self.is_robot_busy = False

    def depth_callback(self, msg):
        # Depth 이미지를 mm 단위 16비트 이미지로 변환
        self.last_depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    def vision_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model.predict(source=frame, conf=0.6, verbose=False)
        self.last_frame = results[0].plot()

        # [수정] 타겟 레이블 지정
        target_label = "pringles"  # "oreo", "pringles"

        if not self.is_robot_busy:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = results[0].names[cls_id]

                if label == target_label:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    self.get_logger().info(f"타겟 [{label}] 감지! 피킹 스레드 시작.")
                    # 피킹 스레드 실행 (label 인자 포함)
                    threading.Thread(target=self.execute_pick, args=(cx, cy, label)).start()
                    break

    def execute_pick(self, cx, cy, label):
        self.is_robot_busy = True

        # Depth 데이터가 들어올 때까지 최대 3초간 기다림 (처음 실행 시 필요)
        timeout = 3.0
        start_time = time.time()
        while self.last_depth_frame is None:
            if (time.time() - start_time) > timeout:
                self.get_logger().error("Depth 이미지가 수신되지 않아 피킹을 취소합니다.")
                self.is_robot_busy = False
                return
            time.sleep(0.1)

        self.get_logger().info(f"--- {label} Picking Process Started ---")

        try:
            if self.last_depth_frame is None:
                self.get_logger().error("Depth 이미지가 수신되지 않았습니다.")
                self.is_robot_busy = False
                return

            # 1. Depth 값 추출 및 검증
            z_mm = self.last_depth_frame[cy, cx]
            if z_mm <= 0:
                self.get_logger().warn("유효하지 않은 Depth 값입니다.")
                self.is_robot_busy = False
                return

            # 2. 픽셀 -> 카메라 좌표 변환
            cam_x = (cx - self.intrinsics["ppx"]) * z_mm / self.intrinsics["fx"]
            cam_y = (cy - self.intrinsics["ppy"]) * z_mm / self.intrinsics["fy"]
            cam_z = z_mm
            cam_coords = np.array([cam_x, cam_y, cam_z, 1.0])

            # 3. 카메라 -> 로봇 베이스 좌표 변환
            curr_pos = self.get_current_posx()[0] # [x, y, z, rx, ry, rz]
            R = Rotation.from_euler("ZYZ", curr_pos[3:], degrees=True).as_matrix()
            base2gripper = np.eye(4)
            base2gripper[:3, :3] = R
            base2gripper[:3, 3] = curr_pos[:3]

            base2cam = base2gripper @ self.gripper2cam
            target_pos = np.dot(base2cam, cam_coords)[:3]

            self.get_logger().info(f"Target Robot Pos: {target_pos}")

            # 4. 로봇 동작 시퀀스 (실제 movel 호출)
            # A. 접근 위치 (물체 100mm 위)
            approach = self.posx([target_pos[0], target_pos[1], target_pos[2] + 100,
                                 curr_pos[3], curr_pos[4], curr_pos[5]])
            self.movel(approach, vel=60, acc=60)

            # B. 피킹 위치로 하강
            pick = self.posx([target_pos[0], target_pos[1], target_pos[2]-40,
                             curr_pos[3], curr_pos[4], curr_pos[5]])
            self.movel(pick, vel=30, acc=30)

            # C. 그리퍼 작동
            self.get_logger().info("그리퍼 닫기")
            self.gripper.close_gripper()
            self.wait(1.0)

            # D. 들어올리기
            self.movel(approach, vel=60, acc=60)

            # E. 기본 위치로 복귀
            self.movej([0, 0, 90, 0, 90, 0], vel=40, acc=40)
            self.gripper.open_gripper()
            self.get_logger().info("피킹 완료 및 복귀")

        except Exception as e:
            self.get_logger().error(f"동작 중 오류 발생: {e}")

        finally:
            self.is_robot_busy = False

def main(args=None):
    rclpy.init(args=args)

    # 로봇 노드 생성
    robot_node = rclpy.create_node("dsr_vision_node", namespace="dsr01")

    # DSR 라이브러리 설정
    DR_init.__dsr__node = robot_node
    DR_init.__dsr__id = "dsr01"
    DR_init.__dsr__model = "m0609"

    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait
        from DR_common2 import posx, posj

        dsr_funcs = {
            'get_current_posx': get_current_posx,
            'movej': movej,
            'movel': movel,
            'wait': wait,
            'posx': posx
        }
    except ImportError as e:
        print(f"DSR Import Error: {e}")
        return

    # 피킹 노드 생성
    picking_node = VisionPickingNode(dsr_funcs)

    # 멀티스레드 실행기 (이미지 수신과 로봇 통신 분리)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(robot_node)
    executor.add_node(picking_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            if picking_node.last_frame is not None:
                cv2.imshow('YOLOv8 Auto Picking System', picking_node.last_frame)
                if cv2.waitKey(1) == 27: # ESC 종료
                    break
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()