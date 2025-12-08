# HED 후처리 강화학습 기반 파라메터 최적화 개발 기획

## 1. 프로젝트 개요
- **배경**: Holistically-Nested Edge Detection(HED)은 멀티 스케일 특징 추출을 통해 고품질 에지 맵을 생성하지만, 후처리 파라메터에 따라 노이즈, 약한 경계, 과다 검출이 빈번하게 발생한다.
- **문제 인식**: Binary Threshold, Gaussian Blur, Hysteresis Threshold 등 다수의 파라메터를 이미지별로 수동 튜닝하기 어렵고, 도메인별 최적 조합이 상이하다.
- **목표**: 강화학습을 활용해 HED 후처리 파이프라인의 최적 파라메터를 자동 탐색하고, 값 기반 DQN과 정책 기반 PPO의 성능을 비교 분석한다.
- **기대 효과**: 수동 튜닝 대비 에지 품질 향상, 도메인별 추천 파라메터 제공, RL 알고리즘 선택 기준 마련.

## 2. 강화학습 문제 정의 (MDP)
- **환경 구성**: OpenAI Gym 스타일의 환경으로 구현하여 안정적인 학습 루프 확보.
- **상태(State)**:
  - 입력 이미지 및 현재 파라메터 세팅(Threshold, Blur σ, NMS 강도, Morphology Kernel 등).
  - 에지 밀도(edge density), 연속성(edge continuity), 반복 횟수 등 중간 통계 피쳐 포함.
- **행동(Action)**:
  - 파라메터 증감(예: Threshold ±1, Gaussian Blur σ ±0.2, NMS 강도 ±0.1).
  - 연속 파라메터는 범위를 이산화하거나 PPO에서 직접 연속 제어.
- **보상(Reward)**:
  - Ground Truth 에지 맵과 비교한 정량 지표(F1-score, IoU, Pixel Accuracy, Boundary Precision Recall 등)를 결합.
  - Reward shaping을 통해 탐색 초기에 극단값으로 수렴하는 현상 방지.

## 3. 시스템 아키텍처
1. 입력 이미지 로딩
2. HED 모델 추론으로 초기 에지 맵 생성
3. 후처리 파라메터 적용 모듈 실행
4. RL 에이전트(DQN 또는 PPO)가 다음 행동 결정
5. 결과 에지 맵과 GT를 비교해 보상 계산
6. 에이전트 파라메터 업데이트
7. 수렴 시 최적 파라메터 조합 및 성능 리포트 생성

## 4. 알고리즘 전략
- **DQN(Deep Q-Network)**:
  - 이산 행동 공간에서 강점, discrete 파라메터 제어에 적합.
  - 안정성 향상을 위해 Double DQN, Prioritized Replay, Target Network 업데이트 주기 조정 검토.
- **PPO(Proximal Policy Optimization)**:
  - 연속 행동 공간 지원, 빠른 수렴 및 안정성 확보.
  - Advantage normalization, clipping 파라메터 조정, entropy bonus로 탐색 유도.
- **공통 전략**:
  - 동일한 상태 표현과 데이터셋을 사용해 공정 비교.
  - 학습 로그, 행동 분포, 보상 곡선을 동일 포맷으로 수집.

## 5. 데이터셋 및 전처리 계획
- **주요 데이터셋**: BSDS500, NYU Depth Dataset, 브라우니 기본 이미지셋 등 다양한 도메인 이미지.
- **전처리**:
  - 이미지 해상도 정규화, 채널 정규화.
  - GT 에지 맵 기준 통일.
  - 훈련·검증·테스트 세트 분할 시 도메인별 균형 유지.

## 6. 평가 및 분석 계획
- **정량 지표**: 최종 Reward, F1-score, IoU, BPR, Pixel Accuracy.
- **비교 항목**: 수렴 속도, 안정성(Variance), 최적 파라메터 품질, 이미지 유형별 성능.
- **시각화**: 학습 곡선, Reward 상승 그래프, 파라메터 영향도 히트맵, 파라메터 sensitivity 분석.

## 7. 리스크 및 대응
- **상태/행동 공간 복잡성**: 파라메터 범위가 과도하게 넓을 경우 학습 지연 → 영역별 스케일링, 파라메터 그룹화.
- **보상 설계 실패**: 단일 지표 편향 위험 → 복합 지표 가중합 및 Curriculum 도입.
- **연산 자원 한계**: 대량 이미지 학습 시 GPU 메모리 부족 → 배치 크기 조정, Mixed Precision 고려.

## 8. 산출물 (Deliverables)
- DQN/PPO 기반 후처리 파라메터 최적화 코드 및 학습 스크립트.
- 실험 로그, 모델 체크포인트, 파라메터 추천 리스트.
- 비교 분석 리포트 및 시각화 자료(학습 곡선, 파라메터 영향도 등).

## 9. 운영 및 배포 고려사항
- **개발 환경**: Windows + NVIDIA RTX 4070 Ti Super, Python 3.14.x, PyTorch, Stable-Baselines3, OpenCV.
- **버전 관리**: GitHub 레포지토리로 코드와 실험 로그 관리.
- **재현성 확보**: Seed 고정, Docker 혹은 Conda 환경 YAML 제공.
