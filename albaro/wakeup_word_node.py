import time
import numpy as np
import sounddevice as sd
import tensorflow as tf

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class WakeupNode(Node):
    def __init__(self):
        super().__init__("wakeup_node")

        self.wakeup_pub = self.create_publisher(Bool, "/start_item_check", 10)
        # TFlite 모델 로드   
        self.interpreter = tf.lite.Interpreter(model_path="trained.tflite")
        # 텐서 메모리 할당
        self.interpreter.allocate_tensors()
        # 입력 텐서 정보
        self.input_details = self.interpreter.get_input_details()[0]
        # 출력 텐서 정보
        self.output_details = self.interpreter.get_output_details()[0]

        # 오디오 입력 버퍼
        self.buf = np.zeros(3960, dtype=np.float32)
        self.hit_count = 0
        self.last_wake_time = 0.0

        self.get_logger().info("Wakeup node started")

        '''
        마이크 오디오 캡쳐
        samplerate: 마이크 입력 샘플링 주파수
        channels: 채널 수
        dtype: 데이터 타입
        blocksize: 한 번에 콜백으로 전달되는 오디오 샘플 수
        '''
        self.stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocksize=1600,
            callback=self.audio_cb,
        )
        self.stream.start()

    def push_audio(self, audio):
        n = len(audio)
        self.buf[:-n] = self.buf[n:]
        self.buf[-n:] = audio

    def predict(self):
        x = self.buf.reshape(self.input_details["shape"]).astype(np.float32)
        self.interpreter.set_tensor(self.input_details["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details["index"])[0]

    def audio_cb(self, indata, frames, time_info, status):
        now = time.time()

        # 쿨다운 중이면 무시 (중복 감지 방지)
        if now - self.last_wake_time < 5.0:
            return

        # 오디오 추출
        audio = indata[:, 0].astype(np.float32)
        # 무음, 잡음 필터
        rms = np.sqrt(np.mean(audio ** 2))

        # 너무 작은 소리는 무시
        if rms < 0.02:
            self.hit_count = 0
            return

        self.push_audio(audio)
        probs = self.predict()
        conf = probs[1]

        # 연속적으로 높은 확률로 맞으면 wake로 판정
        if conf >= 0.95:
            self.hit_count += 1
        else:
            self.hit_count = 0

        if self.hit_count >= 3:
            msg = Bool()
            msg.data = True
            self.wakeup_pub.publish(msg)

            self.last_wake_time = now
            self.hit_count = 0

            self.get_logger().info("🔥 wakeup 발생!!")


def main():
    rclpy.init()
    node = WakeupNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
