from collections import defaultdict

import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from configs import DEVICE

from .train_contrastive_embedding import train_contrastive


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
        new_target = ensemble_target_label(labels)
        relabeled_data.loc[unnoised_data.index[i], "new_target"] = new_target

    return relabeled_data


def get_similar_indices_with_scores(
    base_sentences: list[str], sentences: list[str], topk: int = 5
) -> tuple[list[int], list[float]]:
    model_name = "./results/jhgan/ko-sroberta-multitask"
    try:
        model = AutoModel.from_pretrained(model_name).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        model, tokenizer = train_contrastive()

    base_embeddings = get_sentence_embedding(model, tokenizer, base_sentences)
    embeddings = get_sentence_embedding(model, tokenizer, sentences)

    similarity_matrix = cosine_similarity(embeddings.cpu().numpy(), base_embeddings.cpu().numpy())

    topk_indices = similarity_matrix.argsort(axis=1)[:, -topk:][:, ::-1]
    topk_scores = [similarity_matrix[i, indices] for i, indices in enumerate(topk_indices)]
    return topk_indices, topk_scores


def get_sentence_embedding(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, sentences: list[str]
) -> torch.Tensor:
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)

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
