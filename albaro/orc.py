import os
import sqlite3
from pathlib import Path
from datetime import datetime

from STT import STT
from keyword_extraction import ExtractKeyword
from STT import openai_api_key


# =====================================================
# DB 설정
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "pipeline_logs.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stt_text TEXT,
            items TEXT,
            counts TEXT,
            poses TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(text, result):
    items, counts, poses = result

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (stt_text, items, counts, poses, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            text,
            str(items),
            str(counts),
            str(poses),
            datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()


# =====================================================
# Orchestrator
# =====================================================
class Orchestrator:
    def __init__(self, openai_api_key):
        self.stt = STT(openai_api_key)
        self.extractor = ExtractKeyword()

    def run_once(self):
        text, wav_path = self.stt.speech2text()

        try:
            if not text:
                print("[ORCH] STT 실패, 종료")
                return None

            result = self.extractor.extract_keyword(text)
            if not result:
                print("[ORCH] 키워드 추출 실패")
                return None

            save_to_db(text, result)
            print("[ORCH] DB 저장 완료:", result)
            return result

        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
                print("[ORCH] wav 삭제 완료:", wav_path)


# =====================================================
# 실행
# =====================================================
if __name__ == "__main__":
    init_db()
    orch = Orchestrator(openai_api_key)
    orch.run_once()
