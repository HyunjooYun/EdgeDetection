# 프로젝트 보고서

## I. 서론

### 1. 연구배경
에지 검출은 시각 정보 이해의 핵심 단계이며, Holistically-Nested Edge Detection(HED)은 학습 기반 접근법으로 높은 성능을 제공한다. 그러나 HED가 생성한 에지 맵은 후처리 설정에 따라 품질 편차가 크다. 본 프로젝트는 강화학습(RL)을 활용해 HED 후처리 파이프라인을 자동으로 최적화함으로써, 다양한 이미지에서 일관된 에지 품질을 확보하는 것을 목표로 한다. Stable-Baselines3(SB3)의 DQN, PPO 알고리즘을 채택했고, BSDS500 기반 데이터셋을 사용하였다.

### 2. 목적
- HED 후처리 환경을 RL이 다룰 수 있도록 정식화하고 SB3 기반 학습 파이프라인을 구축한다.
- 학습 중 이미지가 중복 추출되는 문제를 해결하기 위해 이미지 순환(cycle) 옵션을 도입하고, 전체 데이터셋을 순차적으로 활용한다.
- 200,000 타임스텝 학습을 진행한 DQN/PPO 모델에 대해 train/val/test 전 split을 대상으로 정량적·정성적 평가를 수행한다.

## II. 프로젝트 설계 및 진행

### 1. 데이터셋 및 전처리
- **데이터 구성**: BSDS500 이미지와 대응하는 GT(edge)를 inputs/train, inputs/val, inputs/test로 재구성하였다. 모든 split에서 200장 이미지를 사용 가능한 형태로 정리했다.
- **Baseline 생성**: 기존 HED Caffe 모델(hed_pretrained_bsds.caffemodel)을 이용해 outputs/hed/*_baseline 디렉터리에 기본 에지 맵을 생성했다. 이 맵이 RL 에이전트 후처리의 입력으로 사용된다.

### 2. 환경 설계
- **환경 구조**: src/hed_rl/envs/hed_postprocess_env.py 내 HEDPostProcessEnv는 상태로 HED 출력과 후처리 파라미터를, 행동으로 후처리 연산 조합을 사용한다. 보상은 HED 후처리 결과와 GT 간 F1-score 기반 측정값이다.
- **cycle_images 옵션**: 같은 이미지가 반복 선택되던 문제를 해결하기 위해 cycle_images 플래그를 추가했다. 옵션 활성화 시 이미지 목록을 순차적으로 순회하고 reset 단계에서 다음 이미지를 할당한다.

### 3. 학습 방법
- **DQN 설정**: scripts/train_dqn.py 실행 시 주요 하이퍼파라미터는 learning_rate 4.3e-4, gamma 0.9311, batch_size 128, buffer_size 100,000, train_freq 8, target_update_interval 2,000, gradient_steps 2, exploration_fraction 0.1642, exploration_final_eps 0.0124, seed 537700이다. 전체 타임스텝은 200,000으로 설정했고 cycle-images 옵션을 활성화해 train split 전체를 순환했다. TensorBoard 로그는 runs/dqn_cycle_200k에 기록했다.
- **PPO 설정**: scripts/train_ppo.py는 SB3 기본 하이퍼파라미터(learning_rate 3e-4 등)를 사용하고, 동일하게 cycle-images 옵션과 200,000 타임스텝을 적용했다. 로그 경로는 runs/ppo_cycle_200k이다.
- **로깅**: RolloutImageCallback이 학습 중 추출한 이미지 샘플을 TensorBoard(runs/*/rollout_images)에 기록한다.

### 4. 실험 프로토콜
- **평가 구성**: scripts/evaluate_agents.py에 --save-rollouts-dir 옵션을 추가하여 평가시 base/pred/GT 비교 이미지를 PNG로 저장하게 했다.
- **평가 파라미터**: train/val/test 각각 200 에피소드(=이미지) 평가, cycle-images 옵션으로 중복 없이 순회, image-log-count 200으로 설정하여 모든 이미지에 대한 rollouts를 저장했다. 결과 JSON은 artifacts/eval_*_cycle_200k.json, PNG는 artifacts/eval_images_cycle_200k/{split}/{model}/에 기록했다.

## III. 결과 분석

### 1. 학습 결과
- DQN 학습 로그(runs/dqn_cycle_200k)에서 평균 에피소드 보상이 타임스텝 증가에 따라 완만히 상승하여 15 전후 수준에서 수렴했다.
- PPO 학습(runs/ppo_cycle_200k)은 안정적으로 15~16 보상대를 유지했고, 엔트로피 감소와 KL 값 기록으로 정책이 점차 결정론적 행동으로 수렴함을 확인했다.

### 2. 테스트 결과 (정량)
- **Train split (200 episodes)**
  - DQN: mean_reward 12.29, std 8.87, mean_episode_length 27.40±7.30.
  - PPO: mean_reward 15.55, std 10.09, mean_episode_length 29.56±3.08.
- **Val split (200 episodes)**
  - DQN: mean_reward 13.12, std 9.61, mean_episode_length 27.59±7.69.
  - PPO: mean_reward 14.91, std 9.79, mean_episode_length 29.71±2.89.
- **Test split (200 episodes)**
  - DQN: mean_reward 12.90, std 9.18, mean_episode_length 29.11±4.53.
  - PPO: mean_reward 13.42, std 10.23, mean_episode_length 29.48±3.70.

PPO가 모든 split에서 평균 보상이 DQN보다 높았으며, episode length가 30에 근접해 더 많은 후처리 단계를 활용한다는 점이 드러난다. 다만 표준편차가 9~10으로 상당히 높아 이미지 간 편차가 존재한다.

### 3. 평가 결과 (정성)
- PNG 비교: artifacts/eval_images_cycle_200k/{split}/{model}에서 base/pred/GT가 가로로 배치된 결과를 확인했다. PPO는 저대비 이미지에서 더 안정적으로 얇은 에지를 유지하는 경향이 있었고, DQN은 일부 이미지에서 잡음이 더 많았다.
- TensorBoard 이미지 탭에서도 학습 진행 단계별 후처리 품질 개선을 확인할 수 있다.

## IV. 결론

### 1. 결론
cycle 이미지 순회와 200K 타임스텝 학습으로 DQN/PPO 모두 데이터 전체를 효율적으로 사용했고, PPO가 전 split에서 일관되게 더 높은 보상을 달성했다. rollouts PNG를 아카이브함으로써 정량/정성 분석 기반을 마련했다.

### 2. 한계와 개선방향
- 보상이 F1-score 기반 단일 지표에 의존해 결과 표준편차가 높다. 다른 메트릭(정밀도, 재현율) 조합이나 다중 보상 설계가 필요하다.
- PPO의 보상 분산이 크므로 advantage normalization 튜닝이나 학습률 스케줄링 등 안정화 기법을 시험해볼 수 있다.
- 200K 타임스텝 이후 추가 학습(예: 400K)이나 다른 RL 알고리즘(SAC 등) 비교가 필요하다.

### 3. 향후 연구
- 보상 구조 개선과 멀티타스크 학습 확장.
- 추가 데이터셋(예: 도시 풍경, 의료 영상) 적용 및 도메인 적응 실험.
- RL 결과를 전통적 최적화 기법과 비교하는 ablation study 수행.
