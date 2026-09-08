# DR-Classification

비대칭 환자에서 양안 융합이 어느 등급 경계를 해치는지 분석하고, 해당 경계만 단안 예측으로 되돌리는 연구 코드입니다.
현재 버전은 **PHASE 1: EyePACS / ConvNeXt-S / seed 42 / patient-level 5-fold** 기준입니다.

**PHASE 1 실행용 저장소입니다. 완료된 새 본실험 결과는 아직 없습니다.**
이전 실행의 체크포인트는 head·전처리·학습 방식이 달라 이번 프로토콜에서 재사용하지 않습니다.

## 코드 위치

| 작업 | 실제 코드 |
| --- | --- |
| 단안·양안 ConvNeXt-S | [models/experts.py](src/drpair/models/experts.py) |
| Ordered-CORAL head | [models/ordinal.py](src/drpair/models/ordinal.py) |
| 이미지·환자 분할·5-fold | [data/](src/drpair/data/) |
| 학습·EMA·예측 저장 | [training/](src/drpair/training/) |
| Teacher A / **one-boundary Teacher B** | [teachers/targets.py](src/drpair/teachers/targets.py) |
| Gate 입력·네트워크·학습 | [gates/](src/drpair/gates/) |
| 지표·환자 bootstrap·local/joint 효과 | [evaluation/](src/drpair/evaluation/) |
| 실행 단계·비교군 정의 | [experiments/](src/drpair/experiments/) |
| 검증 테스트 | [tests/](tests/) |
| 아이디어 원문 PDF | [docs/papers/research/](docs/papers/research/) |
| 확정한 방법과 PDF 해석 | [docs/research_protocol.md](docs/research_protocol.md) |

## 설치와 데이터

Python 3.12, NVIDIA CUDA 12.8용 PyTorch 환경입니다. 의존성은 `pyproject.toml`과 **`uv.lock`**으로 관리합니다.
Conda 환경 YAML은 사용하지 않습니다. uv 자체가 없다면 [공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)를 따릅니다.

```sh
uv sync --locked
uv run --locked drpair --help
uv run --locked dvc remote add --local --default local /path/to/dvc-store
uv run --locked dvc pull data/eyepacs.dvc
uv run --locked drpair prepare
uv run --locked pytest -q
```

이 PC에서는 `pair/dvc-store`가 로컬 DVC 저장소로 설정되어 있습니다.
원본 이미지 88,702장은 Git에 들어가지 않고 `data/eyepacs.dvc`가 내용을 추적합니다.
Git에는 상대 경로·환자 ID·라벨·fold가 담긴 [분할 CSV](data/splits/eyepacs.csv)도 포함됩니다.

다른 PC가 `dvc pull`하려면 이 로컬 저장소에 접근할 수 있어야 합니다.
공유 NAS/S3/SSH 저장소가 생기면 그 주소를 설정하고 데이터를 `dvc push`합니다.
현재 로컬 하드링크는 디스크 공간을 아끼는 연결이며 별도 물리 백업은 아닙니다.
데이터 등록·복원 방법은 [data/README.md](data/README.md)에 있습니다.

## 실행

아래 명령은 **사용자가 실행할 때만** 학습을 시작합니다. 설치·prepare·status는 학습하지 않습니다.

```sh
uv run --locked drpair status
uv run --locked python scripts/run_phase1.py
uv run --locked drpair experts --job final_single
uv run --locked drpair experts --all
uv run --locked drpair gates
uv run --locked drpair report
```

`--all`은 final 2개와 OOF 10개를 처리합니다. 각 expert는 자신의 training 환자 안에서
epoch 선택용 pilot을 학습한 뒤, 선택된 epoch 수로 해당 training 환자 전체를 새로 학습합니다.
따라서 **최종 expert 12개 + 선택용 pilot 12개**이며, 가벼운 한 번의 실행이 아닙니다.
공식 val/test나 OOF holdout 라벨로 expert의 epoch를 고르지 않습니다.
완료한 job은 같은 코드·config·manifest인 경우 체크포인트를 재사용합니다.
중단된 epoch의 optimizer 자동 재개는 구현하지 않았으며, 미완료 job은 처음부터 다시 학습합니다.

`scripts/run_phase1.py`는 experts -> Teacher A/B 및 gates -> report를 한 번 실행한 뒤 종료합니다.
동일 run의 중복 실행은 lock으로 막고 오류가 나면 중단하여 `execution_status.json`에 기록합니다.
Windows에서는 실행 중 시스템 자동 절전만 막으며 화면 켜짐은 강제하지 않습니다. 예약 작업은 아닙니다.

설정은 [configs/phase1.toml](configs/phase1.toml)에 있습니다.
RTX 5070 Ti 사전 측정에 따라 단안 B16 x 2 / 양안 B8 x 4 / effective batch 32 /
512px RGB / BF16 / channels-last / workers 6을 사용합니다. PyTorch allocator 상한은 13 GiB입니다.
측정 원본은 `experimental_results/gpu_preflight/`에 있으며 성능 비교 결과가 아닙니다.
학습 진행률과 VRAM peak는 각 job의 `logs/*.progress.json` 및 표준 출력에 기록합니다.

## 결과

실제 실행 결과는 `experimental_results/<run_name>/` 한 곳에 모입니다.
`checkpoints`, `predictions`, `features`, `teachers`, `gates`, `logs`, `metrics`, `figures`, `REPORT.md`는 실행 시 생성합니다.
작은 CSV·JSON·리포트는 Git에, 큰 NPZ·체크포인트는 DVC에 기록합니다.
현재 `experimental_results/repository_validation/`은 설치·코드·데이터 검증 기록이며 성능 결과가 아닙니다.

Teacher B supervision은 각 경계의 `U_b`입니다. 실제 joint mixing 효과는 별도로 보고하며
16-mask oracle은 분석 전용입니다. PHASE 1 보고서 뒤에는 자동으로 다음 실험이 시작되지 않습니다.
다중 seed·외부 데이터·추가 backbone 순서는 [PLAN.md](PLAN.md)에 있습니다.
