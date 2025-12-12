# HED Post-Processing RL Prototype

이 레포지토리는 HED 후처리 파라메터 최적화를 위한 강화학습 실험을 준비하기 위한 프로토타입 환경을 제공합니다. 주요 구성 요소는 다음과 같습니다.

- `hed_rl` 패키지: OpenAI Gym 스타일의 `HEDPostProcessEnv` 환경, 파라메터 스펙, HED 추론 유틸리티.
- `scripts/simulate_env.py`: 환경을 빠르게 실행해보는 데모 스크립트.
- `scripts/run_hed_edges.py`: HED Caffe 모델로 테스트 이미지의 에지 맵을 생성하는 유틸리티.
- `scripts/evaluate_agents.py`: 학습된 에이전트를 일괄 평가하고 TensorBoard/PNG로 비교 자료를 남기는 스크립트.
- `docs/development_plan.md`: 전체 개발 방향과 리서치 포커스 정리 문서.

## 설치

```powershell
# 가상환경 활성화 후 (필수 패키지 선 설치)
"C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" -m pip install -r requirements.txt
"C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" -m pip install -e . --no-deps
```

필수 의존성은 `requirements.txt`에 정리되어 있습니다. `stable-baselines3` 사용을 위해 PyTorch가 필요하며, CUDA 가속을 활용하려면 GPU 환경에 맞는 PyTorch 빌드를 별도로 설치해 주세요.

## 사용법

### HED 추론 (선택적 준비)

OpenCV DNN으로 HED를 사용하려면 BSDS에서 학습된 Caffe 가중치가 필요합니다.

1. [deploy.prototxt](https://github.com/s9xie/hed/blob/master/examples/hed/deploy.prototxt)와
	[hed_pretrained_bsds.caffemodel](http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel)를 다운로드합니다.
2. 레포지토리 내 `models/hed/` 폴더에 파일을 저장합니다.
3. 다음 명령으로 테스트 이미지의 HED 에지 맵을 생성해 확인할 수 있습니다.

```powershell
"C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" scripts/run_hed_edges.py --prototxt models/hed/deploy.prototxt --caffemodel models/hed/hed_pretrained_bsds.caffemodel --image-dir imgs/test --output-dir outputs/hed
```

### 데이터셋 준비

- 학습 입력 이미지는 `inputs/train/`에, 대응하는 GT(edge) 맵은 `inputs/GT/`에 동일한 파일명으로 저장합니다.
- 예: `inputs/train/12003.jpg` ↔ `inputs/GT/12003.png`.
- 리포지토리에는 경로만 추적되고 실제 이미지는 포함되지 않으니, 직접 다운로드하거나 변환해 배치해야 합니다.

### 학습용 HED 에지맵 생성 (사전 계산)

```powershell
"C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" scripts/run_hed_edges.py ^
	--prototxt models/hed/deploy.prototxt ^
	--caffemodel models/hed/hed_pretrained_bsds.caffemodel ^
	--image-dir inputs/train ^
	--output-dir outputs/hed/train_baseline ^
	--width 500 ^
	--height 500
```

- `--width`/`--height`는 OpenCV DNN 안정성을 위한 리사이즈 옵션이며 필요에 따라 조정할 수 있습니다.
- 이후 RL 학습 단계에서 `--edge-dir outputs/hed/train_baseline`를 넘기면 반복적으로 HED를 다시 계산하지 않아도 됩니다.

### 강화학습 환경 스모크 테스트

```powershell
"C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" scripts/simulate_env.py --steps 15
```

출력으로 강화학습 환경의 랜덤 탐색 보상과 종료 여부를 확인할 수 있습니다. 추후 DQN, PPO 등의 에이전트를 학습시키기 위한 베이스라인으로 사용할 수 있습니다. HED 모델 경로를 `HEDPostProcessConfig`에 전달하면 실제 에지 맵 기반 보상(F1-score)이 계산됩니다.

### DQN 학습 실행

```powershell
"C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" scripts/train_dqn.py ^
	--timesteps 200000 ^
	--image-dir inputs/train ^
	--ground-truth-dir inputs/GT ^
	--edge-dir outputs/hed/train_baseline ^
	--prototxt models/hed/deploy.prototxt ^
	--caffemodel models/hed/hed_pretrained_bsds.caffemodel ^
	--hed-width 500 ^
	--hed-height 500 ^
	--image-log-frequency 5000 ^
	--image-log-count 5 ^
	--tensorboard-log runs/dqn ^
	--output artifacts/dqn_hed
```

- 실행 전 `numpy==1.26.4`, `opencv-python==4.8.1.78`, `tensorboard==2.20.0` 버전 호환성을 확인하세요.
- `--prototxt/--caffemodel`을 생략하면 사전 계산된 에지(`--edge-dir`)만으로도 학습이 가능하지만, 옵션을 남겨두면 캐시 미스 시 자동으로 HED를 재생성합니다.
- 학습이 끝나면 모델은 `artifacts/dqn_hed.zip`(기본 확장자 `.zip`)로 저장되고, TensorBoard 로그는 `runs/dqn/` 하위에 생성됩니다.
- 메모리를 절약하려면 `--no-cache-edges`를 지정해 에지 캐시를 비활성화할 수 있습니다.
- `rollout_images/` 서브 디렉터리에는 TensorBoard 이미지 로그(초기 HED · 예측 · GT를 가로로 붙인 비교)가 저장되며, `--image-log-frequency`, `--image-log-count`로 간격과 샘플 수를 조정할 수 있습니다.

### PPO 학습 실행

```powershell
"C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" scripts/train_ppo.py ^
	--timesteps 200000 ^
	--image-dir inputs/train ^
	--ground-truth-dir inputs/GT ^
	--edge-dir outputs/hed/train_baseline ^
	--prototxt models/hed/deploy.prototxt ^
	--caffemodel models/hed/hed_pretrained_bsds.caffemodel ^
	--hed-width 500 ^
	--hed-height 500 ^
	--image-log-frequency 5000 ^
	--image-log-count 5 ^
	--tensorboard-log runs/ppo ^
	--output artifacts/ppo_train
```

- Ray Tune로 얻은 최적 하이퍼파라미터를 `--learning-rate`, `--gamma`, `--batch-size` 등 CLI 옵션으로 즉시 덮어쓸 수 있습니다.
- PPO 역시 TensorBoard 이미지 로그를 동일 형식으로 저장하며, `runs/ppo/`에서 학습 곡선과 롤아웃을 확인할 수 있습니다.

### Ray Tune 하이퍼파라미터 탐색

```powershell
"C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" scripts/tune_hyperparams.py ^
	--algo ppo ^
	--timesteps 100000 ^
	--num-samples 16 ^
	--image-dir inputs/train ^
	--ground-truth-dir inputs/GT ^
	--edge-dir outputs/hed/train_baseline ^
	--prototxt models/hed/deploy.prototxt ^
	--caffemodel models/hed/hed_pretrained_bsds.caffemodel
```

- `--algo`로 `dqn`/`ppo` 중 선택하고, `--num-samples`로 탐색 trial 개수를 조정하세요.
- 결과는 기본적으로 `runs/tune/hed_rl_tune` 하위에 저장되며, `result.json`으로 각 trial의 `mean_reward`를 확인할 수 있습니다.
- Ray Tune은 추가 CPU를 활용할 수 있으므로, 병렬화를 위해 `--cpus-per-trial`과 `RAY_NUM_CPUS` 환경변수를 상황에 맞게 설정하세요.

### 에이전트 평가 및 이미지 추출

```powershell
"C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" scripts/evaluate_agents.py ^
	--dqn-model artifacts/dqn_hed.zip ^
	--ppo-model artifacts/ppo_train.zip ^
	--prototxt models/hed/deploy.prototxt ^
	--caffemodel models/hed/hed_pretrained_bsds.caffemodel ^
	--image-dir inputs/train ^
	--ground-truth-dir inputs/GT ^
	--edge-dir outputs/hed/train_baseline ^
	--episodes 20 ^
	--image-log-count 5 ^
	--tensorboard-log runs/eval ^
	--output-json artifacts/eval_results.json
```

- 스크립트는 모델별 평균/표준편차 보상과 에피소드 길이를 계산해 콘솔·JSON·TensorBoard에 동시 기록합니다.
- 평가 중 로깅된 롤아웃 비교 이미지는 TensorBoard 이미지 탭과 `artifacts/eval_images/`(이벤트 파일에서 추출한 PNG)에서 시각화할 수 있습니다.
- `--image-dir`로 전달한 데이터가 평가 대상이므로, 테스트셋 구분이 필요하면 해당 경로를 별도로 지정하세요.

### 학습 로그 시각화

```powershell
& "C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" -m tensorboard.main --logdir runs
```

- 브라우저에서 `http://localhost:6006/`을 열면 학습(`runs/dqn`, `runs/ppo`), 튠(`runs/tune`), 평가(`runs/eval`) 로그를 하나의 대시보드에서 확인할 수 있습니다.
- 이벤트 파일은 하위 폴더별(`runs/dqn/DQN_1/` 등)로 저장되며, Python에서 `tensorboard.backend.event_processing.EventAccumulator`로 파싱해 요약 통계를 계산할 수 있습니다.

### 산출물 정리

- `artifacts/dqn_hed.zip`: Ray Tune 추천 하이퍼파라미터로 200k timesteps 학습한 DQN 정책.
- `artifacts/ppo_train.zip`: 동일 조건에서 학습한 PPO 정책.
- `artifacts/eval_results.json`: PPO/DQN 20-에피소드 평가 통계(JSON).
- `artifacts/eval_images/`: TensorBoard 이벤트에서 추출한 롤아웃 비교 PNG 묶음.
- `runs/`: 학습(`runs/dqn`, `runs/ppo`), 하이퍼파라미터 탐색(`runs/tune`), 평가(`runs/eval`)용 TensorBoard 로그.
