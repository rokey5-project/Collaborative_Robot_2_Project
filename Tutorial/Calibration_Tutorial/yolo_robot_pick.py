import cv2
import rclpy
from rclpy.node import Node
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG
from ultralytics import YOLO
import time
import numpy as np
import os

# 두산 로봇 라이브러리 임포트
import DR_init

# 로봇 및 그리퍼 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

class YoloAutoPickNode(Node):
    def __init__(self):
        super().__init__("yolo_auto_pick_node")
        self.get_logger().info("Node Initializing...")

        # 1. 이미지 노드 및 카메라 파라미터 초기화
        self.img_node = ImgNode()
        # 데이터가 들어올 때까지 잠시 대기
        for _ in range(5):
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
        
        self.intrinsics = self.img_node.get_camera_intrinsic()
        
        try:
            self.gripper2cam = np.load("T_gripper2camera.npy")
            self.get_logger().info("Calibration file loaded.")
        except FileNotFoundError:
            self.get_logger().error("T_gripper2camera.npy NOT FOUND!")
            self.gripper2cam = np.eye(4)
        
        # 2. YOLO 모델 로드 (경로 재확인)
        model_path = "oreo_pringles/best.pt" 
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found at {model_path}")
        self.model = YOLO(model_path)
        
        # 3. 그리퍼 초기화
        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
        except Exception as e:
            self.get_logger().warn(f"Gripper connection failed: {e}")
        
        self.target_label = "oreo" 
        self.is_picking = False

    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        camera_x = (center_x - intrinsics["ppx"]) * center_z / intrinsics["fx"]
        camera_y = (center_y - intrinsics["ppy"]) * center_z / intrinsics["fy"]
        camera_z = center_z
        return (camera_x, camera_y, camera_z)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords):
        coord = np.append(np.array(camera_coords), 1)
        from DSR_ROBOT2 import get_current_posx # 함수 내부 임포트
        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    def pick_and_drop(self, x, y, z):
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait
        from DR_common2 import posx
        
        self.is_picking = True
        self.get_logger().info(f"START PICKING: {x}, {y}, {z}")
        
        current_pos = get_current_posx()[0]
        # 접근 높이 (z-70은 예시이므로 환경에 맞춰 수정)
        pick_pos = posx([x, y, (z - 70), current_pos[3], current_pos[4], current_pos[5]])
        
        movel(pick_pos, vel=60, acc=60)
        self.gripper.close_gripper()
        wait(1.0)

        movej([0, 0, 90, 0, 90, 0], vel=60, acc=60)
        self.gripper.open_gripper()
        wait(1.0)
        self.is_picking = False
        self.get_logger().info("PICKING FINISHED")

    def run_detection_loop(self):
        # ROS 통신 업데이트 (매 루프마다 실행 필수)
        rclpy.spin_once(self.img_node, timeout_sec=0.01)
        
        frame = self.img_node.get_color_frame()
        depth_frame = self.img_node.get_depth_frame()

        if frame is None:
            # 프레임이 안 들어오면 로그 출력 후 리턴
            # self.get_logger().warn("Waiting for frame...")
            return None

        # YOLO 예측 (신뢰도 낮춰서 테스트)
        results = self.model.predict(source=frame, conf=0.25, verbose=False)
        result = results[0]
        
        target_found = False
        temp_robot_pos = None

        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 박스 그리기
            color = (0, 255, 0) if label == self.target_label else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == self.target_label:
                target_found = True
                z = depth_frame[cy, cx] if depth_frame is not None else 0
                if z > 0:
                    camera_pos = self.get_camera_pos(cx, cy, z, self.intrinsics)
                    temp_robot_pos = self.transform_to_base(camera_pos)

        cv2.imshow("Detection Debug", frame)
        key = cv2.waitKey(1)
        
        if target_found and key == ord('p') and not self.is_picking:
            if temp_robot_pos is not None:
                self.pick_and_drop(*temp_robot_pos)

        return key

if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_yolo_pick", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait
        from DR_common2 import posx, posj
    except ImportError:
        print("Robot library import failed.")
        exit()

    auto_node = YoloAutoPickNode()
    print("--- Loop Started (Press 'p' on window to pick, 'q' to quit) ---")

    try:
        while rclpy.ok():
            key = auto_node.run_detection_loop()
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    rclpy.shutdown()