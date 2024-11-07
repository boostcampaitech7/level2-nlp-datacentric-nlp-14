import os
import random

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Trainer, TrainingArguments

from configs import DATA_DIR, DEVICE, SEED
from noise_data_filter import noise_labeling
from relabel_with_embedding import get_sentence_embedding
from utils import set_seed


class TripletTrainer(Trainer):
    def compute_loss(self, model: PreTrainedModel, inputs: dict[str, dict[str, torch.Tensor]], **kwargs):
        # Get embeddings for anchor, positive, and negative
        anchor_emb = self._get_embeddings(model, inputs, "anchor")
        positive_emb = self._get_embeddings(model, inputs, "positive")
        negative_emb = self._get_embeddings(model, inputs, "negative")

        # Compute Triplet Loss
        margin = 1.0  # Margin for Triplet Loss
        pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
        neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
        loss = F.relu(pos_dist - neg_dist + margin).mean()

        return loss

    def _get_embeddings(
        self, model: PreTrainedModel, tokenized_input: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        input_ids = tokenized_input[f"{prefix}_input_ids"].to(self.model.device)
        attention_mask = tokenized_input[f"{prefix}_attention_mask"].to(self.model.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        token_embeddings = outputs[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_triplet_data(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data: pd.DataFrame,
    label_column: str,
    text_column: str,
) -> Dataset:
    all_texts = data[text_column].tolist()
    embeddings = get_sentence_embedding(model, tokenizer, all_texts)
    similarity_matrix = cosine_similarity(embeddings.cpu().numpy())

    triplet_data = {"anchor": [], "positive": [], "negative": []}

    for idx, row in data.iterrows():
        anchor_text: str = row[text_column]
        anchor_label: int = row[label_column]

        for i in range(7):
            if anchor_label == i:
                continue

            positive_pool = data[data[label_column] == anchor_label].index.tolist()
            if idx in positive_pool:
                positive_pool.remove(idx)
            if not positive_pool:
                continue

            negative_pool = data[data[label_column] == i].index.tolist()
            if not negative_pool:
                continue

            # Semi-Hard Positive: Select a moderately difficult Positive
            positive_sorted_indices = sorted(positive_pool, key=lambda x: similarity_matrix[idx, x])
            hard_positive_idx = random.choice(
                positive_sorted_indices[len(positive_pool) // 2 : len(positive_pool) // 3 * 2]
            )
            hard_positive_text = data.loc[hard_positive_idx, text_column]

            negative_sorted_indices = sorted(negative_pool, key=lambda x: -similarity_matrix[idx, x])
            hard_negative_idx = random.choice(
                negative_sorted_indices[len(negative_pool) // 2 : len(negative_pool) // 3 * 2]
            )
            hard_negative_text = data.loc[hard_negative_idx, text_column]

            # Create triplet and append to list
            triplet_data["anchor"].append(anchor_text)
            triplet_data["positive"].append(hard_positive_text)
            triplet_data["negative"].append(hard_negative_text)

    return Dataset.from_dict(triplet_data)


if __name__ == "__main__":
    set_seed()
    model_name = "jhgan/ko-sroberta-multitask"
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    labeled_data = noise_labeling(data)

    # TODO: restored_sentence 만드는 함수로 교체 필요
    restored = pd.read_csv(os.path.join(DATA_DIR, "restored_sentences.csv"))
    restored_data = pd.merge(labeled_data, restored[["ID", "restored"]], on="ID")
    # TODO End

    noise_data = restored_data[
        (restored_data["noise_label"]) & (0.3 <= restored_data["noise_ratio"]) & (restored_data["noise_ratio"] <= 0.5)
    ].reset_index(drop=True)

    dataset = create_triplet_data(model, tokenizer, noise_data, "target", "restored")

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, dict[str, list[int]]]:
        anchor = tokenizer(examples["anchor"], padding="max_length", truncation=True, return_tensors="pt")
        positive = tokenizer(examples["positive"], padding="max_length", truncation=True, return_tensors="pt")
        negative = tokenizer(examples["negative"], padding="max_length", truncation=True, return_tensors="pt")

        return {
            "anchor_input_ids": anchor["input_ids"],
            "anchor_attention_mask": anchor["attention_mask"],
            "positive_input_ids": positive["input_ids"],
            "positive_attention_mask": positive["attention_mask"],
            "negative_input_ids": negative["input_ids"],
            "negative_attention_mask": negative["attention_mask"],
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    output_dir = f"./results/{model_name}"
    training_args: TrainingArguments = TrainingArguments(
        output_dir=output_dir,
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = TripletTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
