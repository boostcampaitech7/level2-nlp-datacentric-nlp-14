import os
from itertools import combinations

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from configs import DATA_DIR, DEVICE
from main import main
from noise_data_filter import noise_labeling
from utils import set_seed


def correct_label_errors(data: pd.DataFrame) -> pd.DataFrame:

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(model_name),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        max_new_tokens=128,
        device=DEVICE,
    )

    category_named_data, category_map = label_category(pipe, data)

    return category_named_data


def label_category(pipe: Pipeline, data: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    small_noised_correct_label_data = data[(data["noise_label"]) & (data["noise_ratio"] <= 0.35)]

    categories = []

    for i in range(7):
        selected_label_data = small_noised_correct_label_data[small_noised_correct_label_data["target"] == i]
        response = inference_category(pipe, selected_label_data["restored"].tolist(), categories)
        categories.append(response)
        print(f"{i}번째 카테고리: {response}")

    for i, j in combinations(range(7), 2):
        if categories[i] == categories[j]:
            print(f"{i}번째 카테고리와 {j}번째 카테고리가 같습니다.")
            s1 = small_noised_correct_label_data[small_noised_correct_label_data["target"] == i]["restored"].tolist()
            limited_s1 = s1[:10]
            s2 = small_noised_correct_label_data[small_noised_correct_label_data["target"] == j]["restored"].tolist()
            limited_s2 = s2[:10]
            if compare_categories(pipe, limited_s1, limited_s2, categories[i]):
                categories[j] = inference_new_category(pipe, s2, categories, categories[i])
                print(f"{j}번째 카테고리를 {categories[j]}로 변경합니다.")
            else:
                categories[i] = inference_new_category(pipe, s1, categories, categories[i])
                print(f"{i}번째 카테고리를 {categories[i]}로 변경합니다.")

    corrected_data = data.copy()
    corrected_data["category"] = corrected_data["target"].apply(lambda x: categories[x])

    return corrected_data, {i: category for i, category in enumerate(categories)}


def inference_category(pipe: Pipeline, sentences: list[str], selected_categories: list[str]) -> str:
    prompt = "\n".join([f"{i + 1}. {sentence}\n" for i, sentence in enumerate(sentences)])

    messages = [
        {
            "role": "system",
            "content": (
                "Please analyze the given list of text snippets and identify the most suitable topic.\n"
                "- It should be a wide range of topics that can be used as a major category of news.\n"
                f"- You must select except for this list that has already been selected. {selected_categories}\n"
                "- Answer in korean single word."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return pipe(text, return_full_text=False)[0]["generated_text"]


def compare_categories(pipe: Pipeline, sentences1: list[str], sentences2: list[str], category: str) -> bool:
    """
    sentence1이 sentence2 보다 더 category에 적합하다면 True, 아니면 False를 반환합니다.
    """
    s1_prompt = "\n".join([f"{sentence}\n" for i, sentence in enumerate(sentences1)])
    s2_prompt = "\n".join([f"{sentence}\n" for i, sentence in enumerate(sentences2)])

    messages = [
        {
            "role": "system",
            "content": (
                "Please analyze the following two lists of text snippets "
                f"and select the one that is more suitable for the category {category}.\n"
                "- Answer in single number."
            ),
        },
        {"role": "user", "content": f"Sentence 1:\n{s1_prompt}\n\nSentence 2:\n{s2_prompt}"},
    ]
    text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return pipe(text, return_full_text=False)[0]["generated_text"] == "1"


def inference_new_category(pipe: Pipeline, sentences: list[str], categories: list[str], old_category: str) -> str:
    prompt = "\n".join([f"{i + 1}. {sentence}\n" for i, sentence in enumerate(sentences)])

    messages = [
        {
            "role": "system",
            "content": (
                "Please analyze the given list of text snippets and identify the most suitable topic.\n"
                "- Select a plausible topic that has not been chosen yet.\n"
                f"- Avoid categories already selected: {set(categories)}.\n"
                f"- Your answer **must not be '{old_category}**'.\n"
                "- Answer in a single Korean word that summarizes the overall theme of the snippets."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return pipe(text, return_full_text=False)[0]["generated_text"]


if __name__ == "__main__":
    set_seed()
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    labeled_data = noise_labeling(data)

    # TODO: restored_sentence 만드는 함수로 교체 필요
    restored = pd.read_csv(os.path.join(DATA_DIR, "restored_sentences.csv"))
    restored_data = pd.merge(labeled_data, restored[["ID", "restored"]], on="ID")
    # TODO End

    corrected_data = correct_label_errors(restored_data)
    corrected_data.to_csv(os.path.join(DATA_DIR, "label_corrected_train.csv"), index=False)
    main(corrected_data, do_predict=False)
