## 진행 기록 - 2025-12-07

### 문서 작업
- `기획서_1203_윤현주.pdf`를 분석해 프로젝트 목표, MDP 정의, 알고리즘 전략 등을 `docs/development_plan.md`에 종합 정리.
- 립정 중이던 개발 일정 섹션은 현 단계에서 불필요하다고 판단해 삭제하고, 리스크·산출물·운영 항목 번호를 재정렬.

### 환경 셋업
- 시스템에 설치된 Python 3.14.1 경로(`C:\Users\ariny\AppData\Local\Python\pythoncore-3.14-64\python.exe`)를 확인해 워크스페이스 기본 인터프리터로 전환.
- `requirements.txt`, `pyproject.toml`, `README.md`를 작성해 의존성과 설치 절차를 문서화. `stable-baselines3` 요구사항에 맞춰 `torch` 포함.
- `pip install -e .` 과정에서 `numpy` 휠 빌드 오류가 발생해 `--no-deps` 옵션을 적용, 사전 의존성 설치 후 editable 설치 성공.

### 코드 베이스 초기화
- `src/hed_rl` 패키지 스켈레톤 구축 (`__init__.py`, `envs/` 디렉터리) 및 `pyproject.toml` 기반 패키징 구조 정립.
- `HEDPostProcessEnv`를 구현해 파라메터 스펙 정의, 상태 인코딩(이미지 통계 + 파라메터), 보상 계산(타겟 파라메터와의 정규화 거리) 로직 구성.
- 랜덤 액션 샘플러(`sample_action`) 추가로 gym이 없어도 테스트 가능하도록 조치, 관측 벡터 차원(3 + 2N) 검증 및 수정.

### 검증 및 유틸리티
- `scripts/simulate_env.py`로 랜덤 시뮬레이션 스크립트 작성, 초기 관측·보상 출력 확인.
- `mcp_pylance` 실행 스니펫으로 환경을 직접 reset/step 하며 관측 차원과 보상 흐름 검증.

## 다음 단계 제안
1. HED 후처리 실제 파이프라인(OpenCV + HED 모델) 연동 및 GT 기반 정밀 보상 계산 구현.
2. Stable-Baselines3 기반 DQN·PPO 학습 스크립트 및 실험 로깅/시각화 워크플로우 구축.

## 진행 기록 - 2025-12-08

### 파이프라인 고도화
- `hed_rl.pipeline` 모듈을 신설하고 `HEDModel`, `HedConfig`, `infer_hed_edges` 유틸리티를 구현해 Caffe 기반 HED 추론 지원.
- `scripts/run_hed_edges.py` 작성으로 테스트 이미지(예: `imgs/test`)에 대한 HED 에지맵을 일괄 생성 가능하도록 구성.
- `simulate_env.py`를 업데이트해 HED 프로토텍스트/모델 경로를 옵션으로 받아 실제 에지맵 기반 리워드 평가를 수행할 수 있게 함.

### 환경 보상 체계 개선
- `HEDPostProcessEnv`에 HED 추론 연동, 후처리 모듈(블러, 임계값, 형태학 연산) 구현, 타겟 파라메터로 생성한 pseudo GT 대비 F1-score 보상 계산 로직 추가.
- 관측 벡터에 에지 밀도 특성을 포함하고, 에피소드 정보에 리워드 지표(F1) 표기를 추가.

### 학습 스크립트 작성
- `scripts/train_dqn.py`로 Stable-Baselines3 DQN 학습 파이프라인을 구성, TensorBoard 로깅 및 모델 세이브 경로 옵션화.
- README에 HED 가중치 다운로드 경로와 추론/학습 절차를 상세 서술.

### 학습 실행 및 시각화
- 가상환경 의존성(NumPy 1.26, OpenCV 4.8.1, TensorBoard 2.20.0) 정합성 점검 후 `scripts/train_dqn.py`를 HED 설정으로 2,000 스텝 실행하여 `artifacts/dqn_test.zip` 모델과 `runs/dqn_test/DQN_1` 로그 확보.
- TensorBoard를 `tensorboard --logdir runs/dqn_test`로 구동해 실시간 지표 확인, `tensorboard.backend.event_processing`으로 이벤트를 파싱해 지표 요약값 확인.
- 학습 지표를 Matplotlib으로 가시화해 `artifacts/plots/dqn_training_metrics.png`에 저장, 보상·에피소드 길이·손실 등 핵심 곡선 공유.

### 다음 단계 제안 (업데이트)
1. HED 후처리 파이프라인의 파라메터-연산 매핑 정교화 및 추가 지표(IoU, BPR) 계산 모듈 구현.
2. 실제 GT 에지 라벨이 포함된 데이터셋을 도입하여 pseudo GT 대비 평가의 신뢰성을 향상시키고, DQN 학습 실험을 반복/로그 정리.

## 진행 기록 - 2025-12-11

### 학습 자동화
- `requirements.txt`, `pyproject.toml`에 `ray[tune]`을 추가해 분산 하이퍼파라미터 탐색을 지원.
- `scripts/tune_hyperparams.py`를 작성해 DQN/PPO용 Ray Tune 파이프라인을 구축, 학습 환경 생성 로직을 통합하고 `mean_reward` 기준으로 최적 config를 자동 추출하도록 구성.
- README에 Ray Tune 실행 예시와 결과 디렉터리(`runs/tune/hed_rl_tune`) 확인 방법을 문서화.

### 다음 단계 제안
1. `tune_hyperparams.py` 결과를 활용해 선정된 하이퍼파라미터로 장기 학습을 재실행하고 TensorBoard 로그와 이미지 롤아웃을 비교 분석.
2. Optuna/Ray Tune 결과를 `docs/development_plan.md`에 실험 로그 표로 정리해 향후 실험 재현성을 높일 것.

## 진행 기록 - 2025-12-12

### PPO/DQN 최적화 실행
- Ray Tune으로 PPO와 DQN 각각 200k 스텝 기준 4-trial 탐색을 수행, DQN 최고 trial의 `mean_reward≈21.05` 구성 확인.
- Tune 결과를 기반으로 `scripts/train_ppo.py`, `scripts/train_dqn.py`를 200k timesteps로 재학습해 `artifacts/ppo_train.zip`, `artifacts/dqn_hed.zip`과 `runs/ppo/`, `runs/dqn/` TensorBoard 로그를 확보.
- 학습 중 RolloutImageCallback을 통해 모델별 비교 이미지를 TensorBoard에 기록, `image-log-count` 파라미터로 샘플 수를 5장으로 확장.

### 평가 및 결과 아카이빙
- `scripts/evaluate_agents.py`를 새로 작성해 PPO/DQN을 동일 환경에서 20 에피소드 평가하고 SummaryWriter로 `runs/eval`에 지표·이미지를 기록.
- 평가 요약(`artifacts/eval_results.json`)에 모델별 mean/std reward와 에피소드 길이를 저장, 콘솔 로그로도 확인 가능하게 구성.
- TensorBoard 이벤트 파일에서 롤아웃 이미지를 추출해 `artifacts/eval_images/ppo_rollout_xx.png`, `artifacts/eval_images/dqn_rollout_xx.png` 형태로 저장, 시각 비교 자료로 활용.

### 다음 단계 제안
1. 평가 스크립트에 PSNR, IoU 등 추가 지표를 연동해 모델 성능을 다각도로 분석.
2. 학습·평가 로그를 기반으로 문서화(README, development_plan) 및 테스트셋 분리 실험을 준비.
