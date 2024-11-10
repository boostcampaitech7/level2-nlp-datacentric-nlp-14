import argparse
import random

import numpy as np
import pandas as pd
import torch
from transformers import PretrainedModelForSequenceClassification, PretrainedTokenizer, Trainer, TrainingArguments

from configs import DEVICE
from main import main
from utils import set_seed


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 텍스트를 토큰화하고, PyTorch 텐서 형식으로 반환
        inputs = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        # Trainer가 요구하는 형식으로 딕셔너리 생성
        item = {key: val.squeeze(0) for key, val in inputs.items()}  # input_ids와 attention_mask
        item["labels"] = torch.tensor(label, dtype=torch.long)  # 레이블 추가
        return item


def train_classifier(
    model,
    tokenizer,
    train_texts,
    train_labels,
    output_dir="./results",
    epochs=4,
    batch_size=8,
    learning_rate=2e-5,
):
    # 편의상 val_dataset은 그냥 train_dataset과 동일하게 세팅
    # 어차피 학습 후에 re-label 능력을 보는게 중요
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(train_texts, train_labels, tokenizer)

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 학습 시작
    trainer.train()

    # 학습이 완료된 모델 반환
    return model


def get_sentence_label(
    model: PretrainedModelForSequenceClassification, tokenizer: PretrainedTokenizer, sentences: list[str]
):
    # Tokenize sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

    # Classify sentence labels
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Predict label
    if len(sentences) == 1:
        return logits.argmax(dim=1).item()  # 한 문장일 경우 단일 값 반환
    return logits.argmax(dim=1).tolist()


def get_train_data(clean_data, origin):
    noise_data = origin[origin["noise_label"]]

    # 1. 노이즈 데이터의 text는 restored 데이터로 대체
    origin.loc[noise_data.index, "text"] = noise_data["restored"]

    # 2. 노이즈 없는 데이터의 target은 re-labeling한 값으로 대체
    clean_data_target = clean_data[["ID", "target"]]

    # origin과 clean_data_target을 ID를 기준으로 병합하여, origin에 target_new가 추가됨
    origin = origin.merge(clean_data_target, on="ID", how="left", suffixes=("", "_new"))

    # origin의 target 값을 새로 병합된 target_new 값으로 업데이트
    origin["target"] = origin["target_new"].combine_first(origin["target"]).astype(int)

    # 3. train data 구성 완료
    train_data = origin[["ID", "text", "target"]]

    return train_data


if __name__ == "__main__":

    my_seed = 42  # 원하는 seed 값 설정
    random.seed(my_seed)
    np.random.seed(my_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--do-predict", action="store_true")
    args = parser.parse_args()

    # 편의상 restored에 noise_ratio, noise_label이 추가된 csv를 바로 불러옵니다
    restored_with_filtered = pd.read_csv("data/restored_with_filtered.csv")

    noise_data = restored_with_filtered[
        (restored_with_filtered["noise_label"])
        & (0.3 <= restored_with_filtered["noise_ratio"])
        & (restored_with_filtered["noise_ratio"] <= 0.5)
    ]

    # 1084개
    train_texts = noise_data["restored"].tolist()
    train_labels = noise_data["target"].tolist()

    model_name = "jhgan/ko-sroberta-multitask"
    tokenizer = PretrainedTokenizer.from_pretrained(model_name)
    model = PretrainedModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    # 모델 학습
    model = train_classifier(model, tokenizer, train_texts, train_labels)

    not_noise = restored_with_filtered[~restored_with_filtered["noise_label"]]
    to_classify = not_noise["restored"].tolist()

    # re-labeling
    classified_labels = get_sentence_label(model, tokenizer, to_classify)
    not_noise["target"] = classified_labels

    # 최종 train 데이터셋 구성
    train_data = get_train_data(clean_data=not_noise, origin=restored_with_filtered)

    set_seed()
    main(train_data)
