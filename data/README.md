# EyePACS 데이터

모델이 읽는 위치는 `data/eyepacs/{train,val,test}/{0,1,2,3,4}/*.jpeg`입니다.
`DRPAIR_ROOT` 환경변수로 이미지 경로를 바꿀 수 있습니다.
`splits/eyepacs.csv`는 상대 경로와 환자·좌우·정답을, `splits/folds_seed42.csv`는 train/validation fold를 기록합니다.
공식 분할은 train 35,126 / val 10,906 / test 42,670장입니다. val/test는 각각 Kaggle Public/Private 역할입니다.

현재 PC: 기존 `pair/eyepacs_official_split` 이미지와 `pair/dvc-store`를 하드링크로 연결했습니다.
같은 실제 파일을 여러 경로로 읽는 것이므로 이 경로의 원본 이미지는 수정하지 않습니다.
파일 내용을 변경하려면 DVC 작업 사본을 먼저 `dvc unprotect`하여 연결을 분리합니다.

## 받기

```sh
uv sync --locked
uv run --locked dvc remote add --local --default local /accessible/path/to/dvc-store
uv run --locked dvc pull data/eyepacs.dvc
uv run --locked drpair prepare
```

이 PC의 remote/cache 설정은 `.dvc/config.local`에 있고 Git에 올리지 않습니다.
동료에게는 코드 외에 접근 가능한 store 주소가 필요합니다. 이 PC의 폴더만 설정된 상태는 인터넷 공유가 아닙니다.
데이터 접근 권한은 원 배포처의 조건을 따릅니다. 인증키는 Git에 넣지 않습니다.

## 기존 다운로드 데이터 등록

```sh
uv run --locked python scripts/register_local_data.py --source /path/to/eyepacs_official_split --store /same-volume/dvc-store
uv run --locked python scripts/verify_dvc_restore.py --store /same-volume/dvc-store
```

등록은 source 이미지를 제거하지 않습니다. 다른 볼륨에 하드링크할 수 없으면 중단합니다.
`verify_dvc_restore.py`는 별도의 빈 cache에서 세 split의 양쪽 눈 6장을 `dvc pull`로 복원하고 SHA-256을 대조합니다.
전체 이미지 내용의 hash는 DVC가 추적합니다. 중복 이미지 및 라벨 모순 감사는 별도로 기록합니다.

## 결과 보존

각 run의 큰 산출물은 완료 후 아래처럼 DVC에 추가할 수 있습니다.

```sh
uv run --locked dvc add experimental_results/RUN/checkpoints experimental_results/RUN/predictions experimental_results/RUN/features experimental_results/RUN/teachers experimental_results/RUN/gates
uv run --locked dvc push
```

작은 metrics CSV/JSON, logs, REPORT.md는 Git으로 기록합니다. 테스트용 출력과 연구 결과를 혼합하지 않습니다.
