import os

import pandas as pd
from kiwipiepy import Kiwi

from configs import DATA_DIR


# 노이즈 여부를 판별하는 함수 정의
def noise_check(text):

    tokens = kiwi.tokenize(text)

    # 한자(SH)는 제외하고, 부호, 외국어, 특수 문자의 태그 목록
    noise_tags = {"SF", "SP", "SS", "SSO", "SSC", "SO", "SW", "SL"}

    # 노이즈 태그가 포함된 토큰 수 카운트
    noise_token_count = 0
    for token in tokens:
        # ‘·’ 기호가 포함된 토큰과 영문자가 2개 이상인 토큰은 노이즈 카운트에서 제외
        if "·" in token.form or (token.tag == "SL" and token.len >= 4) or (token.form in ["↑", "↓", "→"]):
            continue
        # 노이즈 태그에 해당하는 토큰이면 카운트 증가
        if token.tag in noise_tags:
            noise_token_count += 1

    # 전체 토큰 수를 기준으로 노이즈 비율 계산
    total_tokens = len(tokens)
    if total_tokens == 0:
        return False  # 토큰이 없는 경우 비어있는 데이터로 간주하여 "정상"으로 처리

    # 노이즈 태그가 전체 토큰의 20% 이상을 차지하는 경우에만 노이즈로 간주
    noise_ratio = noise_token_count / total_tokens
    return noise_ratio


if __name__ == "__main__":

    # Kiwi 형태소 분석기 초기화
    kiwi = Kiwi()

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    train_df["noise_ratio"] = train_df["text"].apply(noise_check)

    noisy_data = train_df[train_df["noise_ratio"] >= 0.2]
    normal_data = train_df[train_df["noise_ratio"] < 0.2]

    # 결과 출력
    print("Noisy Data:")
    print(noisy_data.info())
    print("\nNormal Data:")
    print(normal_data.info())

    train_df.to_csv(os.path.join(DATA_DIR, "train_with_noise_ratio.csv"), index=False)
