import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


class ExtractKeyword:
    def __init__(self):
        # ---------------------------
        # 1. .env 로딩
        # ---------------------------
        ENV_PATH = Path(__file__).resolve().parent / ".env"
        load_dotenv(ENV_PATH, override=True)

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not found")

        # ---------------------------
        # 2. LLM 초기화
        # ---------------------------
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            openai_api_key=OPENAI_API_KEY
        )

        # ---------------------------
        # 3. Prompt 정의
        # ---------------------------
        prompt_content = """
당신은 사용자의 문장에서 사물(item), 개수(count)를 추출하는 시스템입니다.

<중요 규칙>
- 사용자 입력에 사물(item)에 해당하는 명확한 단어가 하나도 없으면
  반드시 다음 형식으로만 출력하세요:
  / 

<출력 규칙>
- 반드시 단 한 줄만 출력
- 형식: item1 item2 ... / count1 count2 ...
- 다른 문장, 줄바꿈, 예시는 절대 출력하지 말 것

<추출 대상>
- 사물(item): cocacola, fanta, sprite, cass
- 개수(count): 한개, 두개, 세개

<개수 규칙>
- 한개 = 1, 두개 = 2, 세개 = 3
- 숫자가 명시되면 해당 값을 사용
- 개수가 없으면 1

<사용자 입력>
"{user_input}"


<출력 형식>
item1 item2 ... / count1 count2 ...
"""

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=prompt_content
        )

        self.lang_chain = self.prompt_template | self.llm

    def extract_keyword(self, output_message: str):
        # ---------------------------
        # 4. 빈 입력 차단
        # ---------------------------
        if not output_message or not output_message.strip():
            return None

        # ---------------------------
        # 5. LLM 호출
        # ---------------------------
        response = self.lang_chain.invoke({"user_input": output_message})
        content = response.content.strip()

        parts = [p.strip() for p in content.split("/")]
        if len(parts) != 2:
            warnings.warn(f"Invalid LLM output format: {content}")
            return None

        items_raw, counts_raw = parts
        items = items_raw.split() if items_raw else []
        counts = counts_raw.split() if counts_raw else []

        if len(counts) < len(items):
            counts += ["1"] * (len(items) - len(counts))

        if len(items) != len(counts):
            return None

        counts = [int(c) for c in counts]
        return items, counts


if __name__ == "__main__":
    extractor = ExtractKeyword()
    print(extractor.extract_keyword("콜라 2개랑 스프라이트"))