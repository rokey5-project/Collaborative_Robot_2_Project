import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import rclpy
import sys
from rclpy.node import Node
from std_msgs.msg import String, Bool
from python_speech_features import mfcc

class WakeupNode(Node):
    def __init__(self):
        super().__init__("wakeup_node")
        self.intent_pub = self.create_publisher(String, "/wakeup_intent", 10)

        self.kill_sub = self.create_subscription(
            Bool, "/kill_wakeup", self.kill_cb, 10
        )

        try:
            self.interpreter = tf.lite.Interpreter(model_path="/home/rokey/albaro/albaro/train_end.tflite")
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.input_shape = self.input_details["shape"]
            self.expected_feat_len = np.prod(self.input_shape)
        except Exception as e:
            self.get_logger().error(f"Model Load Error: {e}")
            return

        self.SR = 16000
        self.BLOCK = 1600
        self.audio_buf = np.zeros(self.SR, dtype=np.float32)
        self.LABELS = ["calc", "noise", "pick", "silence"]

        self.last_pub_time = 0.0
        self.NORM_WIN_SIZE = 101
        self.VOLUME_TH = 0.12
        self.PUB_COOLDOWN = 1.5
        self.calc_counter = 0
        self.CALC_REQUIRED = 2

        self.get_logger().info("--- Wakeup Node: Active (Kill Signal Mode) ---")

        # 마이크 스트림 시작
        self.start_mic_stream()

    def start_mic_stream(self):
        try:
            self.stream = sd.InputStream(
                samplerate=self.SR, channels=1, dtype="float32",
                blocksize=self.BLOCK, callback=self.audio_cb
            )
            self.stream.start()
            self.get_logger().info("🎤 마이크 감지 중...")
        except Exception as e:
            self.get_logger().error(f"마이크 시작 실패: {e}")

    def kill_cb(self, msg: Bool):
        """종료 신호를 받으면 자원을 해제하고 프로세스를 종료합니다."""
        if msg.data:
            self.get_logger().warn("🛑 [KILL] ORC 가동을 위해 노드를 종료하고 자원을 반환합니다.")
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()

            # ROS2 노드 및 Python 프로세스 종료
            rclpy.shutdown()
            sys.exit(0)

    def get_rms(self, audio):
        return np.sqrt(np.mean(np.square(audio)))

    def edge_impulse_normalize(self, feat):
        feat_normalized = np.empty_like(feat)
        actual_win = min(self.NORM_WIN_SIZE, feat.shape[0])
        for i in range(feat.shape[1]):
            column = feat[:, i]
            mean = np.convolve(column, np.ones(actual_win)/actual_win, mode='same')
            feat_normalized[:, i] = column - mean
        return feat_normalized

    def extract_mfcc(self, audio):
        feat = mfcc(signal=audio, samplerate=self.SR, winlen=0.025, winstep=0.01,
                    numcep=20, nfilt=40, nfft=512, preemph=0.97, appendEnergy=False)
        feat = self.edge_impulse_normalize(feat)
        feat = feat.astype(np.float32).flatten()
        if len(feat) < self.expected_feat_len:
            feat = np.pad(feat, (0, self.expected_feat_len - len(feat)))
        else:
            feat = feat[:self.expected_feat_len]
        return feat.reshape(self.input_shape)

    def audio_cb(self, indata, frames, time_info, status):
        now = time.time()
        if now - self.last_pub_time < self.PUB_COOLDOWN:
            self.audio_buf.fill(0)
            return

        self.audio_buf = np.roll(self.audio_buf, -len(indata))
        self.audio_buf[-len(indata):] = indata[:, 0]

        try:
            current_audio = self.audio_buf.copy()
            rms_val = self.get_rms(current_audio)
            if rms_val < self.VOLUME_TH:
                self.calc_counter = 0
                return

            x = self.extract_mfcc(current_audio)
            self.interpreter.set_tensor(self.input_details["index"], x)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]["index"])[0]

            p_idx = self.LABELS.index("pick")
            c_idx = self.LABELS.index("calc")

            pick_conf = probs[p_idx]
            calc_conf = probs[c_idx]

            if pick_conf >= 0.90:
                self.publish_intent("pick", pick_conf, rms_val)
                return

            if calc_conf >= 0.98:
                self.calc_counter += 1
                if self.calc_counter >= self.CALC_REQUIRED:
                    self.publish_intent("calc", calc_conf, rms_val)
            else:
                self.calc_counter = 0

        except Exception:
            pass

    def publish_intent(self, label, conf, rms):
        msg = String()
        msg.data = label
        self.intent_pub.publish(msg)
        self.audio_buf.fill(0)
        self.calc_counter = 0
        self.last_pub_time = time.time()
        self.get_logger().info(f"===> [PUBLISHED]: {label.upper()} (Conf: {conf:.2f}, RMS: {rms:.2f})")

def main():
    rclpy.init()
    node = WakeupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 정상 종료 시에도 자원 해제
        if hasattr(node, 'stream'):
            node.stream.stop()
            node.stream.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()