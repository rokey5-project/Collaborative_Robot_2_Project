import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


class ExtractKeyword:
    def __init__(self):
        # ---------------------------
        # 1. .env 로딩 (파일 기준)
        # ---------------------------
        BASE_DIR = Path(__file__).resolve().parents[2]   # 프로젝트 루트
        env_path = BASE_DIR / ".env"
        load_dotenv(env_path)

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not loaded. Check .env path and content.")

        # ---------------------------
        # 2. LLM 초기화
        # ---------------------------
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            openai_api_key=openai_api_key
        )

        # ---------------------------
        # 3. Prompt 정의
        # ---------------------------
        prompt_content = """
당신은 사용자의 문장에서 사물(item), 개수(count), 목적지(position)를 추출하는 시스템입니다.

<중요 규칙>
- 사용자 입력에 사물(item)에 해당하는 명확한 단어가 하나도 없으면
  반드시 다음 형식으로만 출력하세요:
  / / 

<추출 대상>
- 사물(item): coke, cider, pringles, oreo
- 목적지(position): pos1, pos2, pos3, pos4

<개수 규칙>
- 숫자가 명시되면 해당 값을 사용
- 개수가 명시되지 않으면 출력 시 1을 사용

<목적지 변환 규칙>
- "1번", "트레이 1번", "첫 번째 트레이" → pos1
- "2번", "트레이 2번" → pos2
- "3번", "트레이 3번" → pos3
- "4번", "트레이 4번" → pos4

<기본 목적지 규칙>
- 목적지가 명시되지 않은 경우:
  - coke → pos1
  - cider → pos2
  - pringles → pos3
  - oreo → pos4

<출력 형식>
- 반드시 다음 형식만 출력하세요:
  item1 item2 ... / count1 count2 ... / pos1 pos2 ...
- 항목들은 공백으로 구분
- 설명이나 문장은 절대 출력하지 마세요
- 등장 순서를 반드시 유지하세요

<의미 추론 규칙>
- 명확한 사물명이 없더라도 의미가 명확하면 추론하세요
  (예: "빨간색 음료" → coke)

<사용자 입력>
"{user_input}"
"""
        self.prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=prompt_content
        )

        self.lang_chain = self.prompt_template | self.llm

    def extract_keyword(self, output_message: str):
        # ---------------------------
        # 4. 침묵 / 빈 입력 차단 (가장 중요)
        # ---------------------------
        if not output_message or not output_message.strip():
            print("No user input detected. Skip LLM.")
            return None

        # ---------------------------
        # 5. LLM 호출
        # ---------------------------
        response = self.lang_chain.invoke({"user_input": output_message})
        content = response.content.strip()

        parts = [p.strip() for p in content.split("/")]
        if len(parts) != 3:
            warnings.warn(f"Invalid LLM output format: {content}")
            return None

        items_raw, counts_raw, poses_raw = parts

        items = items_raw.split() if items_raw else []
        counts = counts_raw.split() if counts_raw else []
        poses = poses_raw.split() if poses_raw else []

        # ---------------------------
        # 6. 개수 기본값 보정
        # ---------------------------
        if len(counts) < len(items):
            counts += ["1"] * (len(items) - len(counts))

        # ---------------------------
        # 7. 길이 불일치 방어
        # ---------------------------
        if not (len(items) == len(counts) == len(poses)):
            warnings.warn(
                f"Length mismatch: items={items}, counts={counts}, poses={poses}"
            )
            return None

        counts = [int(c) for c in counts]

        print(f"llm's response(items): {items}")
        print(f"llm's response(counts): {counts}")
        print(f"llm's response(positions): {poses}")

        return items, counts, poses


# ---------------------------
# 8. 테스트용 main
# ---------------------------
if __name__ == "__main__":
    extractor = ExtractKeyword()

    # 테스트 1: 정상 입력
    text = "콜라 2개를 트레이 1번에 넣고 프링글스를 놔"
    result = extractor.extract_keyword(text)
    print("final result:", result)

    # 테스트 2: 침묵 입력
    text = ""
    result = extractor.extract_keyword(text)
    print("final result:", result)
