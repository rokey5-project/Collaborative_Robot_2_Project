import cv2
import rclpy
from rclpy.node import Node
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG
from ultralytics import YOLO
import os
import time
import numpy as np
import DR_init

# 로봇 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class SmartStoreNode(Node):
    def __init__(self):
        super().__init__("smart_store_node")
        
        # 1. YOLO 모델 초기화
        project_folder = "oreo_pringles"
        model_filename = "best.pt"
        model_path = os.path.join(os.getcwd(), project_folder, model_filename)
        self.model = YOLO(model_path)
        
        # 2. 이미지 및 그리퍼 초기화
        self.img_node = ImgNode()
        self.gripper = RG("rg2", "192.168.1.1", "502")
        
        # 카메라 파라미터 및 변환 행렬 로드
        rclpy.spin_once(self.img_node)
        self.intrinsics = self.img_node.get_camera_intrinsic()
        self.gripper2cam = np.load("T_gripper2camera.npy")
        
        # 상태 변수
        self.is_robot_moving = False
        
        # 3. 타이머 설정 (메인 루프를 타이머가 대신함)
        # 0.1초마다 비전 인식 및 로직 수행
        self.timer = self.create_timer(0.1, self.main_loop)
        
        self.get_logger().info("시스템이 준비되었습니다.")

    def main_loop(self):
        # 이미지 노드 업데이트
        rclpy.spin_once(self.img_node, timeout_sec=0)
        frame = self.img_node.get_color_frame()
        depth_frame = self.img_node.get_depth_frame()
        
        if frame is None or depth_frame is None:
            return

        # YOLO 예측
        results = self.model.predict(source=frame, conf=0.5, verbose=False)
        result = results[0]
        boxes = result.boxes
        classes = result.names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = f"{classes[cls_id]}"

            # 물체 중심 계산
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 화면 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 로봇이 멈춰있고 특정 물체를 잡아야 할 때 (예: 클릭 대신 자동 감지)
            # 여기서는 예시로 'q'를 누르거나 특정 조건 시 picking 하도록 구성 가능
            # 일단 마우스 클릭 대신 '자동 인식' 모드로 가려면 아래 주석 해제
            # if not self.is_robot_moving:
            #     self.start_picking_process(cx, cy, depth_frame)

        cv2.imshow("YOLO + Robot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def start_picking_process(self, x, y, depth_frame):
        self.is_robot_moving = True
        
        # Depth 및 좌표 변환
        z = self.get_depth_value(x, y, depth_frame)
        if z is None or z <= 0:
            self.is_robot_moving = False
            return

        camera_center_pos = self.get_camera_pos(x, y, z, self.intrinsics)
        robot_coordinate = self.transform_to_base(camera_center_pos)
        
        self.get_logger().info(f"물체 감지! 로봇 좌표로 이동: {robot_coordinate}")
        self.pick_and_drop(*robot_coordinate)
        
        self.is_robot_moving = False

    # --- 기존 헬퍼 함수들 (get_camera_pos, transform_to_base 등은 그대로 유지) ---
    def get_depth_value(self, x, y, depth_frame):
        # 픽셀 노이즈 방지를 위해 주변 5x5 평균값 권장
        return depth_frame[y, x]

    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        camera_x = (center_x - intrinsics["ppx"]) * center_z / intrinsics["fx"]
        camera_y = (center_y - intrinsics["ppy"]) * center_z / intrinsics["fy"]
        return (camera_x, camera_y, center_z)

    def transform_to_base(self, camera_coords):
        coord = np.append(np.array(camera_coords), 1)
        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def pick_and_drop(self, x, y, z):
        # 실제 로봇 동작 (blocking 함수들이므로 주의)
        current_pos = get_current_posx()[0]
        # 높이 보정 (z-70은 그리퍼 길이에 맞춰 조정 필요)
        pick_pos = posx([x, y, z - 50, current_pos[3], current_pos[4], current_pos[5]])
        
        movel(pick_pos, vel=100, acc=80)
        self.gripper.close_gripper()
        wait(1)
        
        # 안전한 높이로 들어올리기
        movel(posx([x, y, z + 100, current_pos[3], current_pos[4], current_pos[5]]), vel=100, acc=80)
        
        # 시작 위치로 돌아가기
        movej([0, 0, 90, 0, 90, 0], vel=100, acc=80)
        self.gripper.open_gripper()

if __name__ == "__main__":
    rclpy.init()
    # DSR 노드 초기화
    node = rclpy.create_node("dsr_main_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait
        from DR_common2 import posx, posj
    except ImportError:
        print("DSR 라이브러리를 로드할 수 없습니다.")
        exit()

    smart_node = SmartStoreNode()
    
    try:
        rclpy.spin(smart_node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()