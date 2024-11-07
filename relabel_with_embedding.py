import argparse
import os
from collections import defaultdict

import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from configs import DATA_DIR, DEVICE
from main import main
from noise_data_filter import noise_labeling
from utils import set_seed


def relabel_data(data: pd.DataFrame) -> pd.DataFrame:
    noise_data = data[(data["noise_label"]) & (0.3 <= data["noise_ratio"]) & (data["noise_ratio"] <= 0.5)]
    base_sentences = noise_data["restored"].tolist()

    unnoised_data = data[~data["noise_label"]]
    sentences = unnoised_data["text"].tolist()

    top_similar_indices, similarity_scores = get_similar_indices_with_scores(base_sentences, sentences)

    relabeled_data = data.copy()
    relabeled_data["new_target"] = data["target"]

    for i, (indices, scores) in enumerate(zip(top_similar_indices, similarity_scores)):
        labels = noise_data.iloc[indices]["target"].tolist()
        new_target = ensemble_target_label_with_similarity(labels, scores)
        relabeled_data.loc[unnoised_data.index[i], "new_target"] = new_target

    return relabeled_data


def get_similar_indices_with_scores(
    base_sentences: list[str], sentences: list[str], topk: int = 5
) -> tuple[list[int], list[float]]:
    model_name = "jhgan/ko-sroberta-multitask"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)

    base_embeddings = get_sentence_embedding(model, tokenizer, base_sentences)
    embeddings = get_sentence_embedding(model, tokenizer, sentences)

    similarity_matrix = cosine_similarity(embeddings.cpu().numpy(), base_embeddings.cpu().numpy())

    topk_indices = similarity_matrix.argsort(axis=1)[:, -topk:][:, ::-1]
    topk_scores = [similarity_matrix[i, indices] for i, indices in enumerate(topk_indices)]
    return topk_indices, topk_scores


def get_sentence_embedding(model: AutoModel, tokenizer: AutoTokenizer, sentences: list[str]) -> torch.Tensor:
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        model_output = model(**encoded_input)

    return mean_pooling(model_output, encoded_input["attention_mask"])


def mean_pooling(model_output: tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def ensemble_target_label_with_similarity(labels: list[int], similarities: list[float], decay: float = 1) -> int:

    weighted_count = defaultdict(float)
    normalized_similarities = [sim / sum(similarities) for sim in similarities]

    for rank, (label, similarity) in enumerate(zip(labels, normalized_similarities)):
        weight = (decay**rank) * similarity
        weighted_count[label] += weight

    return max(weighted_count, key=weighted_count.get)


def ensemble_target_label(labels: list[int], decay: float = 0.7) -> int:

    weighted_count = defaultdict(float)

    for rank, label in enumerate(labels):
        weight = decay**rank
        weighted_count[label] += weight

    return max(weighted_count, key=weighted_count.get)


if __name__ == "__main__":
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--do-predict", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    labeled_data = noise_labeling(data)

    # TODO: restored_sentence 만드는 함수로 교체 필요
    restored = pd.read_csv(os.path.join(DATA_DIR, "restored_sentences.csv"))
    restored_data = pd.merge(labeled_data, restored[["ID", "restored"]], on="ID")
    # TODO End

    corrected_data = relabel_data(restored_data)
    corrected_data.to_csv(os.path.join(DATA_DIR, "train_relabel_with_embedding.csv"), index=False)

    noise_df = corrected_data[corrected_data["noise_label"]]

    final_data = corrected_data[["ID", "text", "target"]]

    final_data.loc[noise_df.index, "text"] = noise_df["restored"]
    final_data.loc[:, "target"] = corrected_data["new_target"]

    main(final_data, do_predict=args.do_predict)
