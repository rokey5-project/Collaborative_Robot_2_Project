import time
import numpy as np
import sounddevice as sd
import tensorflow as tf

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from python_speech_features import mfcc


class WakeupNode(Node):
    def __init__(self):
        super().__init__("wakeup_node")

        # 🔥 StateManager로 intent만 전달
        self.intent_pub = self.create_publisher(
            String, "/wakeup_intent", 10
        )

        # -----------------------------
        # TFLite 모델 로드
        # -----------------------------
        self.interpreter = tf.lite.Interpreter(
            model_path="/home/rokey/albaro/albaro/trained.tflite"
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.expected_feat_len = int(self.input_details["shape"][1])

        # -----------------------------
        # 오디오 설정
        # -----------------------------
        self.SR = 16000
        self.BLOCK = 1600          # 100ms
        self.ONSET_SEC = 0.4      # 발화 초반만 캡처
        self.ONSET_SAMPLES = int(self.SR * self.ONSET_SEC)

        self.audio_buf = np.zeros(self.ONSET_SAMPLES, dtype=np.float32)
        self.write_idx = 0
        self.capturing = False

        # -----------------------------
        # 상태 / 파라미터
        # -----------------------------
        self.FRAME_RMS_TH = 0.020
        self.last_pub_time = 0.0
        self.PUBLISH_COOLDOWN = 3.0

        # 🔥 분기 기준
        
        self.CALC_STRONG_TH = 0.95
        self.CALC_WEAK_TH = 0.70

        self.get_logger().info(
            f"Wakeup node started | onset_capture={self.ONSET_SEC}s"
        )

        # -----------------------------
        # 마이크 스트림
        # -----------------------------
        self.stream = sd.InputStream(
            samplerate=self.SR,
            channels=1,
            dtype="float32",
            blocksize=self.BLOCK,
            callback=self.audio_cb,
        )
        self.stream.start()

    # -----------------------------
    # MFCC (Edge Impulse 호환)
    # -----------------------------
    def extract_mfcc(self, audio):
        feat = mfcc(
            signal=audio,
            samplerate=self.SR,
            winlen=0.025,
            winstep=0.01,
            numcep=13,
            nfilt=32,
            nfft=512,
            preemph=0.98,
            appendEnergy=False,
        ).flatten().astype(np.float32)

        if len(feat) < self.expected_feat_len:
            feat = np.pad(feat, (0, self.expected_feat_len - len(feat)))
        else:
            feat = feat[: self.expected_feat_len]

        return feat.reshape(1, -1)

    # -----------------------------
    # 추론
    # -----------------------------
    def predict(self, audio):
        x = self.extract_mfcc(audio)
        self.interpreter.set_tensor(self.input_details["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details["index"])[0]

    # -----------------------------
    # 판단 + intent publish
    # -----------------------------
    def decide_and_publish(self, probs):
        now = time.time()

        calc = float(probs[0])
        pick = float(probs[2])

        self.get_logger().info(
            f"[ONSET CONF] calc={calc:.2f}, pick={pick:.2f}"
        )

        if now - self.last_pub_time < self.PUBLISH_COOLDOWN:
            return

        msg = String()

        # 🧮 계산
        if calc >= self.CALC_STRONG_TH:
            msg.data = "calc"
            self.intent_pub.publish(msg)
            self.get_logger().info("→ INTENT: calc")
            self.last_pub_time = now
            return

        # 📦 정리
        if calc <= self.CALC_WEAK_TH:
            msg.data = "pick"
            self.intent_pub.publish(msg)
            self.get_logger().info("→ INTENT: pick")
            self.last_pub_time = now
            return

        self.get_logger().info("⚠️ ambiguous ignored")

    # -----------------------------
    # 오디오 콜백
    # -----------------------------
    def audio_cb(self, indata, frames, time_info, status):
        audio = indata[:, 0]
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # 발화 시작 감지
        if not self.capturing:
            if rms < self.FRAME_RMS_TH:
                return
            self.capturing = True
            self.write_idx = 0

        # 버퍼 채우기
        remain = self.ONSET_SAMPLES - self.write_idx
        n = min(len(audio), remain)
        self.audio_buf[self.write_idx:self.write_idx + n] = audio[:n]
        self.write_idx += n

        # 캡처 완료 → 추론
        if self.write_idx >= self.ONSET_SAMPLES:
            probs = self.predict(self.audio_buf.copy())
            self.decide_and_publish(probs)

            # 리셋
            self.capturing = False
            self.write_idx = 0


def main():
    rclpy.init()
    node = WakeupNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
