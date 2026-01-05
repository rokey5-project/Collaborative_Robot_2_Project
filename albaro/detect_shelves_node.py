import cv2
import rclpy
from rclpy.node import Node
from ultralytics import YOLO
from std_msgs.msg import String
import json


class DetectShelvesNode(Node):
    def __init__(self):
        super().__init__('detect_shelves_node')
        

        self.model = YOLO('best.pt')
        self.detect_pub = self.create_publisher(String, '/detect_info', 10)
        self.detect_camera() # 카메라 실행

    def detect_camera(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("webcam is not working")

        while rclpy.ok():
            ret, frame = cap.read()

            if not ret:
                print('webcam error!!')
                break

            results = self.model.predict(source=frame, conf=0.7, verbose=False)

            result = results[0]
            boxes = result.boxes
            classes = result.names

            self.detect_info = {
                    'cass': 0,
                    'cocacola': 0,
                    'fanta': 0,
                    'sprite': 0
                }

            for box in boxes:
                cls_id = int(box.cls[0])              # 클래스 ID
                conf = float(box.conf[0]) * 100       # 신뢰도 (0~1 → 0  # Q 키 누르면 종료~100%)

                
                self.detect_info[classes[cls_id]] = self.detect_info[classes[cls_id]] + 1

                label = f"{classes[cls_id]} {conf:.1f}%"

                # Bounding box 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 정수 변환

                # 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 클래스 및 확률 표시
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            msg = String()
            msg.data = json.dumps(self.detect_info)
            self.detect_pub.publish(msg)

            cv2.imshow("YOLO Predict", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main(args=None):
    rclpy.init(args=args)
    node = DetectShelvesNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



