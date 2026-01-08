import os
import warnings
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class ExtractKeyword:
    def __init__(self, api_key=None):
        self.api_key = os.getenv("OPENAI_API_KEY")

        if self.api_key:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.5,
                    openai_api_key=self.api_key
                )
                print("✅ [ExtractKeyword] LLM 초기화 성공")
            except Exception as e:
                print(f"❌ [ExtractKeyword] LLM 초기화 에러: {e}")
                self.llm = None
        else:
            print("⚠️ [ExtractKeyword] 유효한 API 키가 없어 LLM을 초기화할 수 없습니다.")
            self.llm = None

        # ---------------------------
        # 2. Prompt 정의 (동일)
        # ---------------------------
        prompt_content = """
당신은 사용자의 문장에서 사물(item), 개수(count)를 추출하는 시스템입니다.

<중요 규칙>
- 사용자 입력에 사물(item)에 해당하는 명확한 단어가 하나도 없으면 반드시 다음 형식으로만 출력하세요: /

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

        if self.llm:
            self.lang_chain = self.prompt_template | self.llm
        else:
            self.lang_chain = None

    def extract_keyword(self, output_message: str):
        if not self.lang_chain:
            print("❌ [ExtractKeyword] LLM 체인이 구성되지 않았습니다.")
            return None

        if not output_message or not output_message.strip():
            return None

        try:
            response = self.lang_chain.invoke({"user_input": output_message})
            content = response.content.strip()

            parts = [p.strip() for p in content.split("/")]
            if len(parts) != 2:
                print(f"⚠️ [ExtractKeyword] 형식 오류: {content}")
                return None

            items_raw, counts_raw = parts
            items = items_raw.split() if items_raw else []
            counts = counts_raw.split() if counts_raw else []

            # 개수 보정
            if len(counts) < len(items):
                counts += ["1"] * (len(items) - len(counts))

            # 정수 변환 및 예외 처리
            final_counts = []
            for c in counts[:len(items)]:
                try:
                    # '한개' 같은 텍스트가 섞여올 경우를 대비해 숫자로 변환 시도
                    final_counts.append(int(c))
                except:
                    final_counts.append(1)

            return items, final_counts
        except Exception as e:
            print(f"❌ [ExtractKeyword] 호출 중 에러: {str(e)}")
            return None

if __name__ == "__main__":
    extractor = ExtractKeyword()
    if extractor.lang_chain:
        # 단독 테스트
        res = extractor.extract_keyword("콜라 2개랑 스프라이트 한개")
        print(f"테스트 결과: {res}")