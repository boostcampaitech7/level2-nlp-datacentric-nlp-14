# Data Centric NLP project

## Project Overview

### 개요

Data-Centric NLP Project는 한국어 뉴스 헤드라인을 7가지 주제로 분류하는 작업에서, **모델 코드 수정 없이 데이터만을 개선하여** 성능을 향상시키는 것을 목표로 한다. 노이즈와 라벨 오류가 포함된 데이터를 정제하고 증강하여 분류 정확도를 높인다.

### 데이터셋

| 데이터 종류      | 개수    | 설명                      |
| ---------------- | ------- | ------------------------- |
| 라벨 에러 데이터 | 1,000개 | 라벨이 잘못 지정된 데이터 |
| 노이즈 데이터    | 1,600개 | 노이즈가 추가된 데이터    |
| 정상 데이터      | 200개   | 정상 데이터               |

- 노이즈 데이터는 라벨이 정확하지만, 문장의 20~80%가 무작위 ASCII 코드로 대체됨
- 라벨 에러 데이터는 노이즈는 없지만, 잘못된 라벨이 지정됨

### 주요 접근법

1. **노이즈 탐지 및 복구**: 데이터에 포함된 노이즈를 식별하고 이를 개선
2. **라벨 에러 탐지 및 re-labeling**: 잘못된 라벨을 찾아내어 정확한 라벨로 수정
3. **데이터 증강**: 데이터의 다양성을 높여 모델 학습에 도움을 줌

## Getting Started

**Requirement**: Python 3.10

### 1. Clone the repository

```bash
$ git clone git@github.com:boostcampaitech7/level2-nlp-datacentric-nlp-14.git
$ cd level2-nlp-datacentric-nlp-14
```

### 2. Create Virtual Environment with Pipenv

`pipenv install` 명령어를 통해 필요한 패키지들을 받습니다.

```shell
$ pip install pipenv
$ pipenv install
```

`pipenv` 가상환경에 진입합니다.

```shell
$ pipenv shell
(level2-nlp-datacentric-nlp-14)$
```

### 3. Set Up Data

`data/` 폴더 내에 데이터들을 위치시킵니다.

- `sample_submission.csv`
- `test.csv`
- `train.csv`

`test.csv`에 대한 추론은 `data/` 내에 `output.csv`로 저장됩니다.

### 4. Run the Project

다음 명령어를 통해 프로젝트를 실행합니다.

```bash
$ python main.py
```

## Project Structure

```plaintext
level2-nlp-datacentric-nlp-14
├── augment
│   └── back_translate.py         # 역번역을 통한 데이터 증강
├── configs
│   └── config.py                 # 프로젝트 설정 파일
├── denoise
│   ├── noise_data_filter.py      # 형태소 분석기를 이용한 노이즈 필터
│   └── restore_noise_data.py     # LLM을 활용한 노이즈 문장 복원
├── relabel
│   ├── relabel_with_embedding.py       # 임베딩을 사용한 re-labeling
│   ├── relabel_with_llm.py             # LLM을 이용한 re-labeling
│   └── train_contrastive_embedding.py  # 대조 학습을 통한 임베딩 학습
├── utils
│   └── util.py
└── main.py                       # 메인 실행 파일
```

## Collaborators

<h3 align="center">NLP-14조 Word Maestro(s)</h3>

<div align="center">

|          [김현서](https://github.com/kimhyeonseo0830)          |          [단이열](https://github.com/eyeol)          |          [안혜준](https://github.com/jagaldol)          |          [이재룡](https://github.com/So1pi)          |          [장요한](https://github.com/DDUKDAE)          |
| :------------------------------------------------------------: | :--------------------------------------------------: | :-----------------------------------------------------: | :--------------------------------------------------: | :----------------------------------------------------: |
| <img src="https://github.com/kimhyeonseo0830.png" width="100"> | <img src="https://github.com/eyeol.png" width="100"> | <img src="https://github.com/jagaldol.png" width="100"> | <img src="https://github.com/So1pi.png" width="100"> | <img src="https://github.com/DDUKDAE.png" width="100"> |

</div>
