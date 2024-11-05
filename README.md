# Data Centric NLP project

데이터의 오류(`noise`, `miss-label` 등)을 검사하고, 수정하여 모델의 성능을 높입니다.

## Getting Started

## Requirement

- Python: 3.10

## Create Virtual Environment with Pipenv

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

## Data

`data/` 폴더 내에 데이터들을 위치시킵니다.

- `sample_submission.csv`
- `test.csv`
- `train.csv`

`test.csv`에 대한 추론은 `data/` 내에 `output.csv`로 저장됩니다.
