# HED Post-Processing RL Prototype

이 레포지토리는 HED 후처리 파라메터 최적화를 위한 강화학습 실험을 준비하기 위한 프로토타입 환경을 제공합니다. 주요 구성 요소는 다음과 같습니다.

- `hed_rl` 패키지: OpenAI Gym 스타일의 `HEDPostProcessEnv` 환경, 파라메터 스펙, HED 추론 유틸리티.
- `scripts/simulate_env.py`: 환경을 빠르게 실행해보는 데모 스크립트.
- `scripts/run_hed_edges.py`: HED Caffe 모델로 테스트 이미지의 에지 맵을 생성하는 유틸리티.
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
	--timesteps 20000 ^
	--image-dir inputs/train ^
	--ground-truth-dir inputs/GT ^
	--edge-dir outputs/hed/train_baseline ^
	--prototxt models/hed/deploy.prototxt ^
	--caffemodel models/hed/hed_pretrained_bsds.caffemodel ^
	--hed-width 500 ^
	--hed-height 500 ^
	--tensorboard-log runs/dqn_train ^
	--output artifacts/dqn_train
```

- 실행 전 `numpy==1.26.4`, `opencv-python==4.8.1.78`, `tensorboard==2.20.0` 버전 호환성을 확인하세요.
- `--prototxt/--caffemodel`을 생략하면 사전 계산된 에지(`--edge-dir`)만으로도 학습이 가능하지만, 옵션을 남겨두면 캐시 미스 시 자동으로 HED를 재생성합니다.
- 학습이 끝나면 모델은 `artifacts/dqn_train.zip`(기본 확장자 `.zip`)로 저장되고, TensorBoard 로그는 `runs/dqn_train/` 하위에 생성됩니다.
- 메모리를 절약하려면 `--no-cache-edges`를 지정해 에지 캐시를 비활성화할 수 있습니다.

### 학습 로그 시각화

```powershell
& "C:/02_Sogang/25_02 ReinforceLearning/EdgeDetection/.venv/Scripts/python.exe" -m tensorboard.main --logdir runs/dqn_test
```

- 브라우저에서 `http://localhost:6006/`을 열면 보상, 손실 등 핵심 지표를 실시간으로 확인할 수 있습니다.
- 이벤트 파일은 `runs/dqn_test/DQN_1/`에 저장되며, Python에서 `tensorboard.backend.event_processing.EventAccumulator`로 파싱해 요약 통계를 계산할 수 있습니다.

### 산출물 정리

- `artifacts/dqn_test.zip`: Stable-Baselines3 `DQN.load`로 재사용 가능한 최신 정책 가중치.
- `artifacts/plots/dqn_training_metrics.png`: TensorBoard 로그 기반 학습 지표(보상, 에피소드 길이, 손실 등) 시각화 이미지.
