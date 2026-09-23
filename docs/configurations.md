# 학습 설정 비교

이 문서는 configs 디렉터리의 직접 실행 가능한 YAML을 비교한다. 모든 설정은 scripts/train.py의 --config 인자로 사용할 수 있으며, 명시하지 않은 값은 bari2d/utils/config.py의 기본값을 상속한다.

## 공통 조건

아래 설정은 모두 seed 7, 1단계 시작, 최대 400 step, 1,000 update, rollout 400, learning rate 0.0003, GRU hidden 64를 사용한다. 보상, action space, 기본 39차원 observation, PPO 계수, 최대 layer 10도 같다.

따라서 각 비교에서 달라지는 것은 대체로 actor 정보 처리 방식, training-only critic, 보조 손실, 그리고 개체 latent다. 학습 성능의 차이는 확률적이므로 한 seed 결과만으로 우열을 결론 내리면 안 된다.

## 설정 행렬

| 파일 | Actor | FiLM | 로봇 latent | Critic | 보조 표적 | 주된 질문 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline.yaml | gru | 아니오 | 아니오 | MLP | 없음 | 단순 순환 정책으로 가능한가 |
| mlp.yaml | mlp | 아니오 | 아니오 | MLP | 없음 | 순환 기억이 실제로 필요한가 |
| gru_traffic.yaml | gru_traffic | 아니오 | 아니오 | MLP | traffic | IR·행동 이력 가지가 유용한가 |
| gru_traffic_connectivity.yaml | gru_traffic_connectivity | 아니오 | 아니오 | MLP | traffic, connectivity, persistence | 연결성 추론 가지가 유용한가 |
| bio.yaml | bio | 아니오 | 아니오 | graph | traffic, connectivity, persistence, force trend | 모든 감각 가지의 효과는 무엇인가 |
| bio_film.yaml | bio_film | 예 | 아니오 | graph | 위와 같음 | 목표 하중 조건화가 필요한가 |
| bio_heterogeneity.yaml | bio_heterogeneity | 예 | 예 | graph | 위와 같음 | 영속적 개체 차이가 협업에 도움이 되는가 |
| curriculum.yaml | gru | 아니오 | 아니오 | MLP | 없음 | 더 넓은 무작위화·잡음에서 견디는가 |

## 각 설정이 학습에 만드는 차이

### baseline과 mlp

baseline의 GRU는 과거 IR, strain, action history에서 시간 문맥을 축적한다. mlp는 같은 순간 observation만 사용하므로, 이 둘의 차이는 순환 기억의 효과를 보여 주는 가장 작은 절제다. 두 설정은 모두 MLP critic과 보조 손실 0을 쓰므로 actor memory 이외의 학습 신호는 거의 같다.

### traffic과 connectivity

gru_traffic은 IR와 과거 action을 시간 합성곱으로 인코딩하고, 인접 접촉 로봇의 직전 이동 수를 예측하는 보조 손실을 받는다. gru_traffic_connectivity는 여기에 IR·strain·속도 이력 기반 연결성 GRU를 더해 접촉 이웃 수와 접촉 지속성도 예측한다.

이 두 설정과 baseline은 MLP critic을 공유한다. 따라서 이 세 설정은 critic을 고정한 채 actor 감각 가지를 단계적으로 비교하기에 적합하다.

### bio와 bio_film

bio는 traffic, connectivity, mechanical branch를 모두 사용한다. mechanical branch는 strain과 action 이력에서 force trend를 예측한다. bio_film은 같은 branch와 graph critic을 유지하면서 공통 목표 하중 표현으로 traffic과 mechanical branch를 FiLM 조건화한다.

그러므로 bio 대 bio_film은 목표 하중 조건화의 효과를 비교하는 직접적인 쌍이다. 두 설정은 branch_hidden 48, graph critic, 보조 손실 계수 0.1을 동일하게 사용한다.

### bio_film과 bio_heterogeneity

bio_heterogeneity는 bio_film 구조에 policy-visible 로봇별 latent를 더한다. latent_sigma 0.05로 에피소드 시작 때 로봇마다 4차원 latent를 표본화하고, 해당 값은 에피소드 동안 고정된다.

bio_film은 latent_sigma 0이므로 동일형 정책이다. 이 비교는 역할 분화가 개인 latent에서 창발하는지 확인한다. latent를 쓰는 정책은 latent의 의미가 사전에 정해져 있지 않으며, 성능 향상도 보장되지 않는다.

### curriculum

모든 설정은 기본 TrainingConfig에서 curriculum_enabled가 true이므로 성공률 기준으로 1단계에서 4단계까지 진행할 수 있다. 모든 설정의 기본 지형은 단계에 따라 폭·방향·경계 불규칙성의 표본 범위를 25%, 50%, 75%, 100%로 넓히고, 로봇 초기 위치와 방향은 매 episode 무작위화한다. curriculum.yaml은 그 위에 다음 환경 변화를 명시한다.

- 기본값과 같은 최대 방향 jitter 18°, 경계 irregularity 0.35, 간극 폭 [3, 5]를 명시
- 모든 단계에서 sensor noise와 actuator noise 0.015
- 별도 output_dir인 runs/curriculum

따라서 curriculum.yaml은 커리큘럼 자체의 on/off 비교라기보다, 모든 단계에서 더 강한 관측·구동기 잡음을 쓰는 GRU 기준선이다.

## Critic 비교의 주의점

baseline 계열은 MLP critic을, bio 계열은 graph critic을 기본으로 사용한다. critic은 실행 시 actor 입력이 아니지만, advantage 품질을 바꾸므로 학습 결과에는 영향을 준다.

따라서 baseline 대 bio의 차이를 감각 branch만의 효과라고 해석할 수 없다. actor 구조만 분리해 비교하려면 한 설정을 복사해 training.critic만 mlp 또는 graph로 맞춰야 한다. 반대로 critic만 비교하려면 actor, branch_hidden, FiLM, latent, reward, seed를 모두 같게 유지해야 한다.

## 권장 실험 순서

1. 기억 효과: mlp 대 baseline
2. 교통 감각: baseline 대 gru_traffic
3. 연결성 감각: gru_traffic 대 gru_traffic_connectivity
4. 목표 조건화: bio 대 bio_film
5. 개체 이질성: bio_film 대 bio_heterogeneity
6. 강건성: baseline 대 curriculum, 또는 각 정책을 stage 4 고정 평가

각 쌍에 대해 적어도 여러 seed를 사용하고 success rate, accurate capacity, construction time, used robot count, anchored robot count, fallen count, energy, action distribution을 함께 비교한다. 이 지표는 episodes.jsonl에 기록된다.

## 실행 예

bio_film 실험:

    conda run -n bari2d python scripts/train.py \
      --config configs/bio_film.yaml --updates 1000

개체 이질성 실험:

    conda run -n bari2d python scripts/train.py \
      --config configs/bio_heterogeneity.yaml --updates 1000

모든 파일의 간단한 목록은 configs/README.md, 보상과 MAPPO의 수식은 training.md를 참고한다.
