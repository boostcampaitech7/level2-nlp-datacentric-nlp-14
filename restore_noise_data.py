import os
import random
import re
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from configs import DATA_DIR, DEVICE
from utils import set_seed


def restore_noise(data: pd.DataFrame) -> pd.DataFrame:
    # 데이터 로드 및 필터링
    gold, black = load_and_filter_data(data)

    # 학습용 예시 생성
    examples = create_training_examples(gold, sample_size=50, mask_prob=0.5)

    # 모델과 토크나이저 설정
    model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)

    # 프롬프트 및 인스트럭션 설정
    prompt = (
        "당신은 제목 데이터 복원가 입니다. 주어진 예시를 통해 데이터를 복원시키는 방법을 학습합니다.\n"
        "You are a title data restorer. Learn how to restore data through given examples."
    )
    instruction = "주어진 real_problem을 네이버 기사 제목 형태로 복원하세요"

    # 복원된 문장 생성
    return generate_restored_sentences(data, examples, black, tokenizer, model, prompt, instruction)


def load_and_filter_data(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """데이터를 로드하고 필터링하여 gold와 black 리스트를 반환합니다."""
    sentences = list(data["text"])

    # gold: 특수 문자나 영어 문자가 없는 문장
    gold = [sentence for sentence in sentences if not re.search(r"[^\w\s…·]|[A-Za-z]", sentence)]
    # black: gold에 속하지 않는 문장
    black = [sentence for sentence in sentences if sentence not in gold]

    return gold, black


def mask_sentences(sentences: List[str], mask_prob: float = 0.1) -> List[str]:
    """
    주어진 문장들에서 mask_prob 확률로 문자를 랜덤한 문자로 대체하여 마스킹된 문장을 반환합니다.
    """
    # 마스킹에 사용할 문자 집합
    replacement_chars = (
        "abcdefghijklmnopqrstuvwxyz" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789" "!@#$%^&*()_+-=[]{}|;:'\",.<>/?`~"
    )

    masked_sentences = []
    for sentence in sentences:
        sentence_chars = list(sentence)
        for i in range(len(sentence_chars)):
            if random.random() < mask_prob:
                sentence_chars[i] = random.choice(replacement_chars)
        masked_sentence = "".join(sentence_chars)
        masked_sentences.append(masked_sentence)
    return masked_sentences


def create_training_examples(gold: List[str], sample_size: int = 50, mask_prob: float = 0.5) -> List[Dict[str, str]]:
    """
    gold 문장들 중에서 샘플을 선택하고 마스킹하여 학습용 예시를 생성합니다.
    """
    selected_sentences = random.sample(gold, sample_size)
    masked = mask_sentences(selected_sentences, mask_prob=mask_prob)

    examples = []
    for original, masked_sentence in zip(selected_sentences, masked):
        examples.append({"role": "user", "content": masked_sentence})
        examples.append({"role": "assistant", "content": original})

    return examples


def generate_restored_sentences(
    data: pd.DataFrame,
    examples: List[Dict[str, str]],
    black: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt: str,
    instruction: str,
) -> pd.DataFrame:
    """
    black 리스트의 각 문장에 대해 모델을 사용하여 복원된 문장을 생성하고 DataFrame에 추가합니다.
    """
    # 준비된 메시지
    messages = [{"role": "system", "content": prompt}, *examples, {"role": "user", "content": instruction}]

    original_sentences = []
    restored_sentences = []

    for sentence in tqdm(black, desc="Processing sentences"):
        # 현재 문장을 real_problem으로 설정
        real_problem = [{"role": "user", "content": sentence}]

        # 전체 메시지 구성
        input_messages = messages + real_problem

        # 토크나이즈
        inputs = tokenizer.apply_chat_template(
            input_messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        terminators = [
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        # 생성
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        # 디코드
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        # 결과 저장
        original_sentences.append(sentence)
        restored_sentences.append(generated_text.strip())

    # DataFrame에 복원된 문장 추가
    data["restored"] = data["text"].apply(
        lambda x: restored_sentences[original_sentences.index(x)] if x in original_sentences else x
    )

    return data


if __name__ == "__main__":
    set_seed()
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    restored_data = restore_noise(data)
    # 결과 저장
    restored_data.to_csv(os.path.join(DATA_DIR, "restored_train.csv"), index=False)

    print("Restored sentences have been saved to 'restored_train.csv'.")
