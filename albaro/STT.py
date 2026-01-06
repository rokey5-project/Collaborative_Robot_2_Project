from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import os
from pathlib import Path

class STT:
    def __init__(self, openai_api_key=None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # [수정] 조건문 단순화: 키가 sk-로 시작하면 정상으로 판단
        if self.api_key and self.api_key.strip().startswith("sk-"):
            try:
                self.client = OpenAI(api_key=self.api_key)
                # 초기화 성공 로그 (길이만 살짝 노출)
                print(f"✅ [STT] OpenAI 클라이언트 로드 완료")
            except Exception as e:
                print(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
                self.client = None
        else:
            print("⚠️ [STT 경고] 유효한 OpenAI API 키가 없습니다.")
            self.client = None

        self.duration = 5       
        self.samplerate = 16000 

    def speech2text(self):
        if not self.client:
            print("❌ [STT] API 키가 없어 분석을 시작할 수 없습니다.", flush=True)
            return None, None

        print("🔴 [STT] 음성 녹음을 시작합니다. (5초 동안 말씀해주세요)", flush=True)
        try:
            # 1. 녹음 수행
            audio = sd.rec(
                int(self.duration * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            print("🟢 [STT] 녹음 완료. Whisper 분석 중...", flush=True)

            # 2. 파일 저장 (인코딩 에러 방지를 위해 단순 경로 사용)
            # tempfile 대신 사용자 홈 디렉토리나 현재 작업 디렉토리에 고정된 이름으로 저장
            wav_path = os.path.join(os.path.expanduser("~"), "temp_stt_audio.wav")
            wav.write(wav_path, self.samplerate, audio)

            # 3. Whisper API 호출
            with open(wav_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            
            text = transcript.text.strip()
            print(f"✅ [STT 결과]: {text}", flush=True)
            return text, wav_path

        except Exception as e:
            # [수정] ascii 에러 방지를 위해 에러 메시지를 str로 명시적 변환
            error_msg = str(e)
            print(f"❌ [STT 에러 발생]: {error_msg}", flush=True)
            return None, None

# ==========================================
# 단독 실행 테스트 (python3 STT.py)
# ==========================================
if __name__ == "__main__":
    stt = STT()
    text, path = stt.speech2text()
    
    if text:
        print(f"\n최종 인식 문자열: {text}")
    
    # 생성된 파일 삭제 (필요시)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except:
            pass