import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import subprocess
import os

# ===============================
# 설정값 (네 환경 기준)
# ===============================
MODEL_PATH = "trained.tflite"      # float32 wakeword 모델
SAMPLE_RATE = 16000

MODEL_SAMPLES = 3960               # 모델 입력 길이
BLOCK_SAMPLES = 1600               # 0.1초 단위

ENERGY_THRESHOLD = 0.02            # RMS 에너지 컷
WAKE_THRESHOLD = 0.95              # wake 확률
DETECT_COUNT_REQUIRED = 3          # 연속 n회

WAKE_INDEX = 1                     # wake 클래스 인덱스

# ===============================
# 실행할 코드 경로 (★ 반드시 실제 경로)
# ===============================
PYTHON_EXEC = "/bin/python3"
FACE_NODE_PATH = "/home/rokey/albaro/albaro/face_age_node.py"


class Wakeup:
    def __init__(self, model_path: str):
        # -------------------------------
        # TFLite 모델 로드
        # -------------------------------
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        print("[INFO] Model input :", self.input_details["shape"], self.input_details["dtype"])
        print("[INFO] Model output:", self.output_details["shape"])

        # -------------------------------
        # 오디오 링버퍼
        # -------------------------------
        self.buf = np.zeros(MODEL_SAMPLES, dtype=np.float32)
        self.hit_count = 0

        # -------------------------------
        # 실행 프로세스 핸들
        # -------------------------------
        self.face_proc = None

    def _push_audio(self, audio: np.ndarray):
        n = len(audio)
        if n >= MODEL_SAMPLES:
            self.buf[:] = audio[-MODEL_SAMPLES:]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = audio

    def predict_probs(self) -> np.ndarray:
        x = self.buf.reshape(self.input_details["shape"]).astype(np.float32)
        self.interpreter.set_tensor(self.input_details["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details["index"])[0]

    def _launch_face_node(self):
        if not os.path.exists(FACE_NODE_PATH):
            print("[ERROR] face_age_node.py 경로가 존재하지 않음")
            return

        if self.face_proc is None or self.face_proc.poll() is not None:
            print("[INFO] Launching face age node...")
            self.face_proc = subprocess.Popen([
                PYTHON_EXEC,
                FACE_NODE_PATH
            ])
        else:
            print("[INFO] Face node already running")

    def run(self):
        print("[INFO] Wake word listening started")

        def callback(indata, frames, time_info, status):
            if status:
                return

            audio = indata[:, 0].astype(np.float32)

            # 1️⃣ RMS 에너지 체크
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < ENERGY_THRESHOLD:
                self.hit_count = 0
                return

            # 2️⃣ 버퍼 업데이트
            self._push_audio(audio)

            # 3️⃣ 추론
            probs = self.predict_probs()
            conf = float(probs[WAKE_INDEX])

            print(f"rms={rms:.4f}, conf={conf:.3f}, probs={probs}")

            # 4️⃣ 연속 프레임 검증
            if conf >= WAKE_THRESHOLD:
                self.hit_count += 1
            else:
                self.hit_count = 0

            # 5️⃣ Wake 감지 → 실행
            if self.hit_count >= DETECT_COUNT_REQUIRED:
                print("🔥 Wake word detected!")
                self._launch_face_node()
                self.hit_count = 0

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SAMPLES,
            callback=callback,
        ):
            while True:
                time.sleep(0.1)


def main():
    wake = Wakeup(MODEL_PATH)
    wake.run()


if __name__ == "__main__":
    main()
