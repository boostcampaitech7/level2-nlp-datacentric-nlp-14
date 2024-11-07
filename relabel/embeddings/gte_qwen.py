import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

from configs import DEVICE


def get_similar_indices_qwen(base_sentences: list[str], sentences: list[str], topk: int = 5) -> list[int]:
    """
    Alibaba-NLP/gte-Qwen2-7B-instruct 모델(LLM base enmbedding model)을 사용하여 base_sentences와 sentences 사이의 유사도를 계산합니다.

    Eval F1 0.6549
    """
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
    model = AutoModel.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True).to(DEVICE)

    base_embeddings = get_sentence_embedding(model, tokenizer, base_sentences)
    embeddings = get_sentence_embedding(model, tokenizer, sentences, prompt_name="query")

    similarity_matrix = cosine_similarity(embeddings.cpu().numpy(), base_embeddings.cpu().numpy())
    return similarity_matrix.argsort(axis=1)[:, -topk:][:, ::-1]


def get_sentence_embedding(model: AutoModel, tokenizer: AutoTokenizer, sentences: list[str], prompt_name="query"):
    # Each query must come with a one-sentence instruction that describes the task
    task = "Given a web search query, retrieve relevant passages that answer the query"
    if prompt_name == "query":
        input_texts = [get_detailed_instruct(task, query) for query in sentences]
    else:
        input_texts = sentences

    embeddings_list = []
    batch_size = 2
    for i in trange(0, len(input_texts), batch_size):
        batch_texts = input_texts[i : i + batch_size]  # Slice batch
        batch_dict = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

        # Move tensors to the same device as the model
        batch_dict = {key: value.to(model.device) for key, value in batch_dict.items()}

        with torch.no_grad():  # Ensure no gradients are stored
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embeddings_list.append(F.normalize(embeddings, p=2, dim=1))

    # Concatenate all batch embeddings
    return torch.cat(embeddings_list, dim=0)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"
