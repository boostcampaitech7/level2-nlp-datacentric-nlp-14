import os

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from configs import DATA_DIR, DEVICE
from main import main
from noise_data_filter import noise_labeling
from utils import set_seed


def inference_category(pipe: Pipeline, sentences: list[str], selected_categories: list[str]) -> str:
    prompt = ""
    for i, sentence in enumerate(sentences):
        prompt += f"{i + 1}. {sentence}\n"

    messages = [
        {
            "role": "system",
            "content": "Please analyze the given list of text snippets and identify the most suitable topic.\n"
            "- It should be a wide range of topics that can be used as a major category of news\n"
            f"- You must select except for this list that has already been selected. {selected_categories}\n"
            "- Answer in korean single word.",
        },
        {"role": "user", "content": prompt},
    ]
    text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return pipe(text, return_full_text=False)[0]["generated_text"]


def label_category(data: pd.DataFrame) -> pd.DataFrame:
    small_noised_correct_label_data = data[(data["noise_label"]) & (data["noise_ratio"] <= 0.35)]

    categories = []

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(model_name),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        max_new_tokens=512,
        device=DEVICE,
    )

    for i in range(7):
        selected_label_data = small_noised_correct_label_data[small_noised_correct_label_data["target"] == i]
        response = inference_category(pipe, selected_label_data["restored"].tolist(), categories)
        categories.append(response)
        print(f"{i}번째 카테고리: {response}")

    corrected_data = data.copy()
    corrected_data["category"] = corrected_data["target"].apply(lambda x: categories[x])

    return corrected_data


def correct_label_errors(data: pd.DataFrame):
    # TODO: Do Something
    corrected_data = data.copy()
    return corrected_data


if __name__ == "__main__":
    set_seed()
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    labeled_data = noise_labeling(data)

    # TODO: restored_sentence 만드는 함수로 교체 필요
    restored = pd.read_csv(os.path.join(DATA_DIR, "restored_sentences.csv"))
    restored_data = pd.merge(labeled_data, restored[["ID", "restored"]], on="ID")
    # TODO End

    category_named_data = label_category(restored_data)

    corrected_data = correct_label_errors(category_named_data)
    corrected_data.to_csv(os.path.join(DATA_DIR, "label_corrected_train.csv"), index=False)
    main(corrected_data, do_predict=False)
