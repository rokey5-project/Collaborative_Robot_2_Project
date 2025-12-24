import os
import cv2
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch

import rclpy
from rclpy.node import Node
# from std_msgs.msg import Bool

class FaceAgeNode(Node):
  def __init__(self):
    super().__init__('face_age_node')

    model_name = "prithivMLmods/facial-age-detection"
    self.model = SiglipForImageClassification.from_pretrained(model_name)
    self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    self.id2label = {
        "0": "age 01-10",
        "1": "age 11-20",
        "2": "age 21-30",
        "3": "age 31-40",
        "4": "age 41-55",
        "5": "age 56-65",
        "6": "age 66-80",
        "7": "age 80 +"
    }

    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  def classify_image(self, image):
    image = Image.fromarray(image).convert("RGB")
    inputs = self.processor(images=image, return_tensors="pt")

    with torch.no_grad():
      outputs = self.model(**inputs)
      logits = outputs.logits
      probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {self.id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    predicted_age_group = max(prediction, key=prediction.get)
    return predicted_age_group

  def detect_camera(self):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
      raise RuntimeError("웹캠을 열 수 없습니다")

    while rclpy.ok():
      ret, frame = cap.read()

      if not ret:
        break

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))

      for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face = frame[y:y + h, x:x + w]

        predicted_age_group = self.classify_image(face)

        cv2.putText(frame, f"Predicted Age: {predicted_age_group}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

      cv2.imshow("Webcam", frame)

def main(args=None):
  rclpy.init(args=args)
  node = FaceAgeNode()

  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      pass
  finally:
      cv2.destroyAllWindows()
      node.destroy_node()
      rclpy.shutdown()


if __name__ == '__main__':
  main()