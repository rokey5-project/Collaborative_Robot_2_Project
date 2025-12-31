from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv


# =====================================================
# 1. .env 로딩
# =====================================================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")


# =====================================================
# 2. STT 클래스
# =====================================================
class STT:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.duration = 5
        self.samplerate = 16000

    def speech2text(self):
        print("음성 녹음을 시작합니다. (5초 동안 말해주세요)")

        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        print("녹음 완료. Whisper에 전송 중...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)

            with open(temp_wav.name, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )

            wav_path = temp_wav.name

        text = transcript.text.strip()
        if not text:
            print("STT 결과 없음")
            return None, wav_path

        print("STT 결과:", text)
        return text, wav_path


# =====================================================
# 3. 단독 실행 테스트
# =====================================================
if __name__ == "__main__":
    stt = STT(OPENAI_API_KEY)
    text, wav_path = stt.speech2text()
    print("최종 반환:", text)
    print("wav 경로:", wav_path)
