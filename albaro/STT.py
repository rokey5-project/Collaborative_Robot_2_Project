from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os
from pathlib import Path

from dotenv import load_dotenv


# =====================================================
# 1. .env 로딩 (파일 기준, 경로 문제 방지)
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[2]   # 프로젝트 루트
load_dotenv(BASE_DIR / ".env")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not loaded. Check .env path and content.")


# =====================================================
# 2. STT 클래스
# =====================================================
class STT:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.duration = 5       # seconds
        self.samplerate = 16000 # Whisper 권장 샘플레이트

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

            wav_path = temp_wav.name  # ✅ 추가

        text = transcript.text.strip()
        if not text:
            print("STT 결과 없음 (침묵 또는 인식 실패)")
            return None, wav_path     # ✅ 기존 return 확장

        print("STT 결과:", text)
        return text, wav_path         # ✅ 기존 return 확장


# =====================================================
# 3. 단독 실행 테스트
# =====================================================
if __name__ == "__main__":
    stt = STT(openai_api_key)
    text, wav_path = stt.speech2text()
    print("최종 반환:", text)
    print("wav 경로:", wav_path)
