import time
import numpy as np
import sounddevice as sd
import tensorflow as tf

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

MODEL_PATH = "trained.tflite"
SAMPLE_RATE = 16000
MODEL_SAMPLES = 3960
BLOCK_SAMPLES = 1600

ENERGY_THRESHOLD = 0.02
WAKE_THRESHOLD = 0.95
DETECT_COUNT_REQUIRED = 3
WAKE_INDEX = 1

WAKE_COOLDOWN_SEC = 5.0   # wake 후 재감지 방지


class WakeupNode(Node):
    def __init__(self):
        super().__init__("wakeup_node")

        # 🔽 토픽명 변경
        self.pub = self.create_publisher(Bool, "/start_item_check", 10)

        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.buf = np.zeros(MODEL_SAMPLES, dtype=np.float32)
        self.hit_count = 0
        self.last_wake_time = 0.0

        self.get_logger().info("Wakeup node started")

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SAMPLES,
            callback=self.audio_cb,
        )
        self.stream.start()

    def _push_audio(self, audio):
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

        # 쿨다운 중이면 무시
        if now - self.last_wake_time < WAKE_COOLDOWN_SEC:
            return

        audio = indata[:, 0].astype(np.float32)
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < ENERGY_THRESHOLD:
            self.hit_count = 0
            return

        self._push_audio(audio)
        probs = self.predict()
        conf = probs[WAKE_INDEX]

        if conf >= WAKE_THRESHOLD:
            self.hit_count += 1
        else:
            self.hit_count = 0

        if self.hit_count >= DETECT_COUNT_REQUIRED:
            msg = Bool()
            msg.data = True
            self.pub.publish(msg)

            self.last_wake_time = now
            self.hit_count = 0

            self.get_logger().info("🔥 Wake detected → start_item_check published")


def main():
    rclpy.init()
    node = WakeupNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
