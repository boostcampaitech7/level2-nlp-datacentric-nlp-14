# Data Centric NLP project

## Getting Started

`pipenv sync` 명령어를 통해 Pipfile.lock의 환경을 동일하게 구축합니다.

```shell
$ pip install pipenv
$ pipenv sync
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
