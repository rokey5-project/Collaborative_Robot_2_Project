import cv2
import os
from src.faceAge import predict

cap = cv2.VideoCapture(8)

if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다")

save_dir = "./captures"
os.makedirs(save_dir, exist_ok=True)

count = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    predicted = predict.main(image_path=frame, weights="./src/faceAge/weights/weights.pt")

    cv2.imshow("Webcam", predicted)

    key = cv2.waitKey(1)

    if key == ord('s'):  # s 키 → 저장
        filename = f"webcam_{count}.jpg"
        path = os.path.join(save_dir, filename)
        cv2.imwrite(path, frame)
        print(f"{filename} 저장 완료")
        count += 1       # ← 번호 증가

    elif key == 27:      # ESC → 종료
        break

cap.release()
cv2.destroyAllWindows()
