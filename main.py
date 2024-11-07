import os

import evaluate
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from configs import DATA_DIR, DEVICE, OUTPUT_DIR, SEED
from utils import set_seed


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data["text"]
        targets = data["target"]
        self.inputs = []
        self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"].squeeze(0),
            "attention_mask": self.inputs[idx]["attention_mask"].squeeze(0),
            "labels": self.labels[idx].squeeze(0),
        }

    def __len__(self):
        return len(self.labels)


def train(data: pd.DataFrame, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
    dataset_train, dataset_valid = train_test_split(data, test_size=0.3, random_state=SEED)

    data_train = BERTDataset(dataset_train, tokenizer)
    data_valid = BERTDataset(dataset_valid, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1.compute(predictions=predictions, references=labels, average="macro")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy="no",
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        save_total_limit=2,
        learning_rate=2e-05,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    train_results = trainer.train()
    eval_results = trainer.evaluate()
    trainer.save_model(OUTPUT_DIR)

    print("Training Summary")
    print("-" * 30)
    print(f"{'Train Data Count':20} : {len(data_train)}")
    print(f"{'Eval Data Count':20} : {len(data_valid)}")
    print(f"{'Train Loss':20} : {train_results.training_loss:.4f}")
    print(f"{'Eval Loss':20} : {eval_results['eval_loss']:.4f}")
    print(f"{'Eval F1':20} : {eval_results['eval_f1']:.4f}")


def predict(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
    dataset_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    model.eval()
    preds = []

    for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Predicting"):
        inputs = tokenizer(sample["text"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)
    dataset_test["target"] = preds
    dataset_test.to_csv(os.path.join(DATA_DIR, "output.csv"), index=False)


def main(data: pd.DataFrame, do_predict: bool = True):
    """
    주어진 DataFrame `data`는 분류 작업에 필요한 `text`와 `target` 열을 포함해야 합니다.

    - `text`: 분류할 자연어 문자열을 담고 있습니다.
    - `target`: 0에서 6 사이의 정수로 이루어진 라벨입니다.

    `text`는 모델의 입력으로 사용되며, `target`은 해당 `text`의 분류 라벨로 사용됩니다.
    """
    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    train(data, model, tokenizer)
    if do_predict:
        predict(model, tokenizer)


if __name__ == "__main__":
    set_seed()
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    data = data[~data["text"].str.contains("\n")]

    main(data)
