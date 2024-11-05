import os

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from configs import DATA_DIR, DEVICE
from main import main
from noise_data_filter import noise_labeling
from utils import set_seed


def label_category(data: pd.DataFrame) -> pd.DataFrame:

    small_noise = data[(data["noise_label"]) & (data["noise_ratio"] <= 0.35)]

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
        noises = small_noise[small_noise["target"] == i]
        prompt = ""
        for j, (_, noise) in enumerate(noises.iterrows()):
            prompt += f"{j + 1}. {noise['restored']}\n"

        messages = [
            {
                "role": "system",
                "content": "Please analyze the given list of text snippets and identify the most suitable topic.\n"
                "- It should be a wide range of topics that can be used as a major category of news\n"
                f"- You must select except for this list that has already been selected. {categories}\n"
                "- Answer in korean single word.",
            },
            {"role": "user", "content": prompt},
        ]
        text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = pipe(text, return_full_text=False)[0]["generated_text"]
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
