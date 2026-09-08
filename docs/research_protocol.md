# PDF 기반 확정 프로토콜

원문 네 편은 `papers/research/`에 보존합니다. v3를 기본으로 삼고, 가장 최근 사용자 설명인
**Teacher B = one-boundary U_b; joint effect = 별도 평가**를 우선합니다.
PDF의 예시 수치와 참고 저장소의 결과는 우리 실험 결과로 사용하지 않습니다.

## 제안 방법

각 target eye에서 `q_s,b = P_s(Y>b)`, `q_f,b = P_f(Y>b)`를 구합니다 (`b=0,1,2,3`).
Single expert는 target만 보고, paired expert는 target/fellow를 순서 있는 feature concatenation으로 융합합니다.
두 expert 사이에는 파라미터 공유가 없습니다. Paired 내부의 양쪽 분기는 같은 encoder를 공유하며 모두 학습됩니다.
동일 ConvNeXt-S 사전학습 모델, head, 입력, 증강, optimizer, effective batch를 적용하고 파라미터 수 차이는 공개합니다.

Head는 CORAL의 shared-score 구조에 증가하는 cutpoint를 명시적으로 넣은 **ordered-CORAL 변형**입니다.
공식 CORAL 구현을 그대로 재현했다고 표기하지 않습니다.
[CORAL 원 논문](https://arxiv.org/abs/1901.07884)의 구조를 참고했습니다.
단안과 양안 모두 `q_0 >= q_1 >= q_2 >= q_3`를 만족합니다.
서로 다른 alpha로 섞은 뒤에는 이 성질이 자동 보장되지 않으므로 PAVA를 기본 적용합니다.
누적 minimum은 낮추는 방향으로 치우칠 수 있어 Euclidean isotonic projection을 사용합니다.

### Teacher A

`U_A,b = BCE(q_f,b,t_b) - BCE(q_s,b,t_b)`, `t_b=1[y>b]`.

### Teacher B (사용자 확정)

`q_cf^(b)`는 다른 좌표를 fusion 그대로 두고, b번째만 single 값으로 바꾼 벡터입니다.
`U_B,b = (decode(PAVA(q_f))-y)^2 - (decode(PAVA(q_cf^(b)))-y)^2`.
한 경계의 교체 후 projection이 다른 좌표도 바꿀 수 있습니다. U_b는 그 전체 결과를 측정합니다.

U_b 자체를 파일에 보존하고, gate 회귀 타깃은 사전 고정한
`target_alpha_b = 1-exp(-max(U_b,0)/T)`로 변환합니다 (`T=1`).
이는 utility를 alpha로 옮기는 설계 선택이며, closed-form 최적 alpha라는 주장을 하지 않습니다.

### Gate와 최종 평가

Gate는 q_s, q_f, fellow single 확률, signed/absolute asymmetry, shift, margin, entropy,
prediction conflict만 봅니다. true grade/gap/patient ID는 입력 feature에 넣지 않습니다.
`q_final,b = (1-alpha_b) q_f,b + alpha_b q_s,b`를 네 경계에 동시에 적용한 뒤 PAVA와 decode를 합니다.
`alpha=0`은 fusion, `alpha=1`은 single 복귀입니다. 희귀 boundary는 training support 기반으로 보수화합니다.

`U_joint`는 **실제로 gate가 낸 alpha**를 함께 적용한 후의 squared-grade-error 개선량입니다.
U_b의 합과 같다고 가정하지 않습니다. `local_vs_joint`에 개별 이득/실제 공동 이득의 불일치를 기록합니다.
16개 binary mask를 조사한 oracle은 별도 diagnostic입니다. Teacher B를 대체하지 않습니다.
soft alpha 전체에 대한 수학적 상한도 아닙니다.

## 누수 없는 두 종류의 교차검증

- Expert OOF: 공식 train을 환자 5-fold로 나눕니다. 각 outer training 안에서만 pilot epoch를 고른 뒤
  outer training 전체에 refit하고 heldout 환자를 예측합니다. OOF teacher 분석과 추후 연구에 보존합니다.
- Gate cross-fitting: v3의 공식 validation 5-fold 설계를 사용합니다. final expert는 공식 train 안에서만
  epoch를 결정하고 train 전체에 refit했습니다. 따라서 gate용 validation 라벨을 expert가 보지 않았습니다.
  각 gate는 validation의 4개 fold로 fit하고 나머지 환자를 예측합니다.
- 최종 test gate는 동일 고정 설정으로 validation 전체에 fit합니다. test는 마지막 평가에만 사용합니다.

이 구성은 기존 코드의 "train OOF로 gate fit"과 다릅니다. PDF v3의 validation cross-fitting을 적용한 결정입니다.
환자 OOF와 다중 training seed는 다른 불확실성을 측정합니다.

## 비교군과 평가

유일한 arm 정의는 `src/drpair/experiments/arms.py`입니다.
A single / B paired / C fixed50 / D global learned / E boundary task-loss /
F Loss-Guided / **G one-boundary Counterfactual**.
비교용 global asymmetry와 entropy reliability는 내부 baseline이며 특정 논문 공식 재현이 아닙니다.
E는 저장된 expert 확률 위에서 gate를 task loss로 학습하며 encoder까지 함께 학습하는 모델은 아닙니다.

Overall 및 gap별 QWK, Macro-F1, Grade-3 F1, MAE, Accuracy, under/over-grading을 기록합니다.
Severe under-grading 비율의 분모는 실제 grade>=3인 눈입니다.
Boundary AUC/Brier와 harmful/beneficial count, alpha 평균·분산·분위수·붕괴 지표를 보고합니다.
기본 threshold는 모든 arm에 동일하게 0.5이며, weighted BCE는 사용하지 않습니다.
Calibration/threshold 민감도 분석은 PHASE 2에서 사전에 정의할 추가 비교입니다.

환자 단위 paired bootstrap은 같은 환자의 두 눈을 함께 재표집합니다.
BH/Holm은 boundary slope 4개와 B0 대비 interaction 3개를 각각 별도 family로 보정합니다.
이는 descriptive regression이고 인과성·novelty를 자동 입증하지 않습니다. 조정 전 p와 조정값을 함께 공개합니다.

## 구현 수정 사항

EMA로 평가한 가중치를 복원하기 전에 복사해 저장합니다. 마지막 미완료 accumulation window도 실제 샘플 수로
정규화해 update합니다. Single/paired augmentation은 RGB이며 pair에 같은 변환 파라미터를 적용합니다.
DataLoader worker seed를 분리합니다. 경로는 상대 경로/환경변수이고 의존성은 uv.lock으로 고정합니다.
코드·config·manifest가 달라지면 이전 run의 checkpoint를 자동 혼용하지 않습니다.

## 보류된 주장

Localized harm, clinical benefit, overall 비열등성, 새로운 현상이라는 주장은 결과로 검증해야 합니다.
ODIR per-eye severity label과 외부 boundary power는 아직 확인하지 않았습니다.
Fuzzy, 외부 논문 공식 구현, 다중 seed, 추가 backbone, Grad-CAM은 이번 저장소 정리에서 실행한 실험이 아닙니다.
