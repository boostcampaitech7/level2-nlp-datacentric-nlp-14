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
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(script_dir, "..", "examples", "normal_text_example.csv"))

    back_translated_data = back_translate(data)
    print("Back Translated Test")
    print("===================Back Translated Test===================")
    for i, row in back_translated_data.iterrows():
        print(f"Original: {data.loc[i, 'text']}")
        print(f"Back Translated: {row['text']}")
        print("*" * 50)
