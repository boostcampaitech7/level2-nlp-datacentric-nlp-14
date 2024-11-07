import pandas as pd
import torch
from tqdm import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration


def back_translate(data: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 소스 언어에서 타겟 언어로 번역
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)

    def back_translate(text, source_lang="ko_KR", target_lang="en_XX"):
        """
        주어진 텍스트를 역번역을 통해 데이터 증강하는 함수

        Args:
            text (str): 원본 텍스트(증강하려는 문장)
            source_lang (str): 소스 언어 코드 (기본값: 'ko_KR'). 예: 'ko_KR' (한국어), 'en_XX' (영어).
            target_lang (str): 타겟 언어 코드 (기본값: 'en_XX'). 예: 'ko_KR' (한국어), 'en_XX' (영어).

        Returns:
            str: 역번역 후의 텍스트. 만약 역번역이 적용되지 않으면 원본 텍스트를 그대로 반환.
        """
        try:
            # 소스 언어에서 타겟 언어로 번역
            tokenizer.src_lang = source_lang
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
            intermediate_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"소스 언어에서 타겟 언어로 번역 중 오류 발생: {e}")
            return text

        try:
            # 타겟 언어에서 소스 언어로 다시 번역
            tokenizer.src_lang = target_lang
            inputs = tokenizer(intermediate_text, return_tensors="pt", padding=True, truncation=True).to(device)
            back_translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[source_lang])
            final_text = tokenizer.batch_decode(back_translated, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"타겟 언어에서 소스 언어로 번역 중 오류 발생: {e}")
            return text

        return final_text

    back_translated_data = data.copy()

    for i, row in tqdm(data.iterrows(), total=len(data)):
        back_translated_data.loc[i, "text"] = back_translate(row["text"])

    return back_translated_data


if __name__ == "__main__":
    data = pd.DataFrame(
        {
            "text": [
                "갤노트8 주말 27만대 개통…시장은 불법 보조금 얼룩",
                "美성인 6명 중 1명꼴 배우자·연인 빚 떠안은 적 있다",
                "아가메즈 33득점 우리카드 KB손해보험 완파…3위 굳...",
                "朴대통령 얼마나 많이 놀라셨어요…경주 지진현장 방문종합",
                "듀얼심 아이폰 하반기 출시설 솔솔…알뜰폰 기대감",
                "NH투자 1월 옵션 만기일 매도 우세",
                "황총리 각 부처 비상대비태세 철저히 강구해야",
                "게시판 KISA 박민정 책임연구원 APTLD 이사 선출",
                "공사업체 협박에 분쟁해결 명목 돈 받은 언론인 집행유예",
                "월세 전환에 늘어나는 주거비 부담…작년 역대 최고치",
                "페이스북 인터넷 드론 아퀼라 실물 첫 시험비행 성공",
                "추신수 타율 0.265로 시즌 마감…최지만은 19홈런·6...",
            ],
        }
    )
    back_translated_data = back_translate(data)
    print("Back Translated Test")
    print("===================Back Translated Test===================")
    for i, row in back_translated_data.iterrows():
        print(f"Original: {data.loc[i, 'text']}")
        print(f"Back Translated: {row['text']}")
        print("*" * 50)
