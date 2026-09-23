# 정책, 보상과 학습

이 문서는 현재 BARI2D 구현의 학습 목표, 팀 보상, MAPPO 업데이트, 커리큘럼, 설정값과 실험 산출물을 정리한다. 모든 수치는 별도 YAML 재정의가 없을 때의 기본값이다.

## 1. 학습 문제

한 에피소드에는 기본 20대의 로봇이 있다. 모든 로봇은 같은 actor 가중치를 공유하지만 각자의 관측 이력과 GRU hidden을 유지한다. actor는 로봇별 local observation과 유효 action mask만 받으며, 중앙 critic만 전역 상태를 학습 중에 사용할 수 있다.

| 구분 | actor | critic |
| --- | --- | --- |
| 실행 위치 | 각 로봇 | 학습 시에만 |
| 입력 | 로봇별 국소 observation 39차원, action mask, hidden | 전체 robot state 187차원 또는 접촉 그래프 |
| 전역 위치·접촉 그래프·현재 구조 용량 | 입력하지 않음 | 입력 가능 |
| 출력 | 10개 action의 분포 | 팀 반환의 스칼라 가치 |

새 기본 observation은 4개 시점의 전방·후방·좌·우·하향 IR, strain, action 이력, 내부 상태, 목표 하중, latent를 포함해 39차원이다. IR 구성을 바꾸거나 action space를 바꾸면 actor 입력 또는 출력 차원이 달라지므로 해당 정책은 새로 학습해야 한다.

각 환경 step에서 모든 로봇의 action을 동시에 적용하고 하나의 팀 보상 \(r_t\)을 계산한다. 이 팀 보상은 모든 로봇의 PPO advantage에 공유된다. 따라서 개별 로봇 action 로그 확률은 따로 저장하지만, credit signal은 협력적이고 동일하다.

## 2. 성공, 종료와 하중 시험

건설 단계에서 접촉·앵커 그래프의 좌우 제방 연결 여부와 빠른 구조 용량을 매 step 평가한다.

1. 그래프가 양쪽 제방을 연결하고 빠른 용량이 목표 하중 이상이면 종단 정밀 하중 시험을 수행한다.
2. 최대 step 400에 도달해도 정밀 하중 시험을 한 번 수행한다.
3. 정밀 용량이 목표 하중 이상이고 그래프가 연결되어 있으면 success이며 terminated가 된다.
4. 시간 제한만 도달한 실패는 truncated가 된다.

기본 1단계 목표 하중은 3.0이다. 2단계 이상에서는 [3.0, 10.0]에서 표본화하며, actor에는 max_target 15.0으로 나눈 목표값만 준다.

## 3. 팀 보상 체계

보상은 다음 항의 합이다.

\[
r_t =
r_{\mathrm{span}}+
r_{\mathrm{mechanical}}+
r_{\mathrm{gap\ progress}}+
r_{\mathrm{time}}+
r_{\mathrm{idle}}+
r_{\mathrm{energy}}+
r_{\mathrm{anchor}}+
r_{\mathrm{robot\_use}}+
r_{\mathrm{collapse}}+
r_{\mathrm{success}}.
\]

구조 품질을

\[
q_t=\min\left(\frac{\mathrm{capacity}_t}{\mathrm{target\_load}},1\right)
\]

로 두면 각 항의 기본값은 아래와 같다.

| 항 | 계산 | 기본 계수 | 의도 |
| --- | --- | ---: | --- |
| span | \(2.0(p_t-p_{t-1})\) | 2.0 | 좌측 제방 연결 component의 전진 진행도 증가 |
| mechanical | \(2.0(q_t-q_{t-1})\) | 2.0 | 목표 하중을 지지할 수 있는 구조 용량 증가 |
| gap_progress | \(0.5\sum_i \Delta p^{\max}_{t,i}\) | 0.5 | 간극 안에서 각 로봇이 오른쪽 제방 방향으로 갱신한 최고 진행도 |
| time | \(-0.002\) | 0.002 | 불필요하게 긴 건설 억제 |
| idle | \(-0.001 n_{\mathrm{idle}}\) | 0.001 | 낙하하지 않은 로봇의 장시간 정지 억제 |
| energy | \(-0.01\Delta E_t\) | 0.01 | 이동·조향·등반 에너지 절약 |
| anchor | \(-0.015 n_{\mathrm{new\ anchor}}\) | 0.015 | 과도한 앵커 사용 억제 |
| robot_use | \(0.0\) | 0.0 | 초기 탐색에서 로봇 사용 자체를 벌하지 않음 |
| collapse | \(-1.0 n_{\mathrm{new\ fall}}\) | 1.0 | 새 낙하에 강한 벌점 |
| success | \(+10.0\) | 10.0 | 정밀 시험을 통과한 제방 간 구조 완성 |

여기서 진행도 \(p_t\)는 좌측 제방에 연결된 로봇 중 간극을 가장 멀리 건넌 위치의 정규화 값이다. 개별 gap progress는 로봇별 최고값을 갱신할 때만 지급하므로 좌우 왕복으로 반복 수집할 수 없다. anchor 벌점은 이번 step에 실제로 새로 연결된 앵커만 센다.

보상에 앵커 파손 자체의 별도 항은 없다. 파손은 구조 용량 저하, 진행도 손실, 낙하와 최종 성공 실패를 통해 간접적으로 불리해진다. 또한 현재 local strain은 접촉·앵커 변형의 근사 센서이며, 종단 하중 시험의 응력이 보상 입력으로 직접 되돌아오지는 않는다.

## 4. Actor 구조와 action 선택

기본 action space는 전진, 후진, 네 방향의 조향 이동, CLIMB, ANCHOR, RELEASE, IDLE의 10개 이산 action이다. 정책 logits에서 물리적으로 불가능한 action은 \(-10^9\)으로 마스킹한다. 환경도 방어적으로 invalid action을 IDLE로 바꾼다.

| architecture | 처리 방식 | recurrent |
| --- | --- | --- |
| mlp | 39차원 observation을 직접 MLP에 입력 | 아니오 |
| gru | observation을 결합 MLP 뒤 GRUCell에 입력 | 예 |
| gru_traffic | IR·action 이력의 교통 가지와 strain 이력 결합 | 예 |
| gru_traffic_connectivity | 교통 가지와 연결성 가지 결합 | 예 |
| bio | 교통·연결성·기계 가지를 모두 사용 | 예 |
| bio_film | bio 가지에 목표 하중 FiLM 조건화 추가 | 예 |
| bio_heterogeneity | bio 가지와 에피소드별 latent를 결합 | 예 |

기본 GRU hidden 크기는 64이며, hidden은 로봇마다 분리된다. 에피소드가 끝나면 hidden을 0으로 초기화한다. actor의 기본 branch hidden은 32, fused hidden은 64이다. 기준 policy는 gru이며, 생물 영감형 기준 policy는 bio_film이다.

CLIMB는 같은 layer의 앞쪽 로봇을 밟아 한 층 올라간다. 높은 layer 로봇은 바로 아래 layer 로봇과 본체 면적이 겹치지 않으면 겹치는 가장 높은 하위 로봇의 한 층 위로 낙하하고, 겹치는 로봇이 없으면 layer 0까지 내려간다. 이 동작은 앵커 여부와 무관하고 별도 action이나 모멘트 계산을 사용하지 않으며, actor는 같은 겹침 규칙을 쓰는 하향 IR로 아래 표면 또는 절벽 신호를 관측한다.

## 5. 보조 표적

보조 손실은 actor가 보지 못하는 전역 정답을 학습 단계에서만 이용해 감각 가지가 유용한 잠재 표현을 만들도록 돕는다. 다음 표적은 환경이 매 step 계산한다.

| 표적 | 값 | 예측 가지 |
| --- | --- | --- |
| traffic | 접촉 이웃 중 직전 step에 이동한 로봇 수 | traffic |
| connectivity | 로봇 접촉 이웃 수 | connectivity |
| contact_persistence | 직전 step부터 유지된 로봇 간 접촉 수 | connectivity |
| force_trend | 최신 normalized strain - 직전 normalized strain | mechanical |

각 사용 가능한 보조 head에는 평균제곱오차를 적용한다. gru와 mlp에는 해당 head가 없으므로 보조 손실이 0이며, architecture에 따라 필요한 head만 합산한다.

## 6. 중앙 critic

기본 MLP critic은 187차원 global state를 두 개의 128-unit Tanh 은닉층으로 처리해 팀 가치 \(V_\phi(s_t)\)를 출력한다. global state는 로봇별 위치, 방향, strain, anchored, 속도, layer, fallen 상태와 간극·목표·진행도·용량·span 요약을 포함한다.

graph critic은 학습 전용 접촉 그래프를 사용한다.

- 노드 feature 13개: 로봇 위치·방향·strain·상태, 제방 접촉, 목표 하중, 간극 폭
- edge feature 6개: edge kind, force/capacity, capacity, 상대 pose
- 64-unit message passing layer 2개와 평균 pooling

실행 중 공유 actor에는 어느 critic 입력도 전달하지 않는다. baseline.yaml은 MLP critic, bio_film.yaml은 graph critic을 사용한다.

## 7. Rollout과 GAE

Trainer는 단일 환경에서 rollout_steps 400개의 팀 step을 수집한다. 각 step에 observation, global state, graph state 선택 사항, action, action mask, 과거 log probability, reward, value, done, reset mask와 보조 표적을 버퍼에 저장한다.

GAE는 다음 식으로 계산한다.

\[
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
\]

\[
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1},
\qquad
R_t=A_t+V(s_t).
\]

기본값은 \(\gamma=0.99\), \(\lambda=0.95\)다. rollout 마지막이 종료가 아니면 critic으로 bootstrap하고, 종료면 마지막 가치를 0으로 둔다. 이후 advantage는 rollout 시간축 전체에서 평균 0, 표준편차 1이 되도록 정규화한다.

GRU 학습을 위해 rollout의 시간 순서를 유지한다. update는 로봇 축만 섞어 기본 5대씩 minibatch를 만들고, 각 minibatch에는 400 step 전체 시퀀스와 rollout 시작 hidden을 전달한다. reset mask는 에피소드 경계에서 hidden을 0으로 만든다.

## 8. MAPPO 목적함수와 최적화

로봇 \(i\)의 현재와 수집 당시 정책 확률비를

\[
\rho_{t,i}=
\exp\left(
\log\pi_\theta(a_{t,i}|o_{t,i})
-\log\pi_{\theta_{\mathrm{old}}}(a_{t,i}|o_{t,i})
\right)
\]

로 둔다. PPO 정책 손실은

\[
L_{\pi}=
-\mathbb{E}_{t,i}
\left[
\min\left(
\rho_{t,i}A_t,
\operatorname{clip}(\rho_{t,i},1-\epsilon,1+\epsilon)A_t
\right)
\right]
\]

이며 \(\epsilon=0.2\)다. actor 전체 손실은

\[
L_{\mathrm{actor}}=
L_{\pi}
-c_H\mathbb{E}[\mathcal{H}(\pi_\theta)]
+c_{\mathrm{aux}}L_{\mathrm{aux}}.
\]

기본 \(c_H=0.03\), \(c_{\mathrm{aux}}=0.1\)이다. 팀 GAE advantage에는 gap progress, energy, idle, anchor, 사용과 낙하를 로봇별로 할인 누적·정규화한 credit advantage를 기본 계수 0.5로 더한다. critic은

\[
L_V=\operatorname{MSE}(V_\phi(s_t),R_t)
\]

를 사용하며, 역전파 시 value_coef 0.5를 곱한다.

기본 설정은 update당 PPO epoch 4회, actor와 critic 모두 Adam learning rate \(3\times10^{-4}\), gradient norm clip 0.5다. critic은 각 PPO epoch에서 전체 rollout을 한 번 사용하고, actor는 그 epoch의 모든 agent minibatch를 순회한다.

## 9. 커리큘럼과 무작위화

커리큘럼은 최근 25개 완료 에피소드의 success 평균이 0.70 이상이면 다음 단계로 올린다. 단계가 오르면 성공 기록 창을 비우며, 최대 단계는 4다.

| 단계 | 변화 |
| --- | --- |
| 1 | 전체 폭·방향·불규칙성 범위의 25% 안에서 무작위 지형, 고정 목표 하중 3.0 |
| 2 | 지형 무작위화 범위 50%, 목표 하중 [3, 10] |
| 3 | 지형 무작위화 범위 75% |
| 4 | 마찰, 질량, 앵커 강도, 센서·구동기 잡음 무작위화 |

모든 단계와 실행 경로에서 로봇은 왼쪽 제방의 겹치지 않는 무작위 위치와 전 방향 무작위 heading으로 시작한다. 같은 seed는 같은 배치를 재현한다. 4단계에서는 지형 범위 100%와 함께 마찰과 앵커 한계를 [0.8, 1.2] 배율로, 질량은 [0.85, 1.15] 배율로 바꾼다. 센서와 actuator noise는 설정값보다 작지 않게 최소 0.01로 둔다.

## 10. 설정과 실행

| 항목 | 기본값 | 비고 |
| --- | ---: | --- |
| total_updates | 1000 | CLI의 --updates로 임시 재정의 가능 |
| rollout_steps | 400 | 팀 환경 step 수 |
| ppo_epochs | 4 | update당 epoch 수 |
| sequence_minibatch_agents | 5 | 시간 전체를 유지하는 agent 묶음 |
| learning_rate | 0.0003 | actor·critic 공통 |
| checkpoint_interval | 50 | update 기준 |
| max_steps | 1000 | 에피소드 시간 제한 |
| robot.count | 20 | 공유 actor를 쓰는 로봇 수 |
| robot.max_layer | 10 | 이산 높이 최대값 |

기준 GRU policy:

    conda run -n bari2d python scripts/train.py \
      --config configs/baseline.yaml --updates 1000

FiLM과 graph critic을 쓰는 policy:

    conda run -n bari2d python scripts/train.py \
      --config configs/bio_film.yaml --updates 1000

비교용 YAML 전체 목록과 설계 의도는 configs/README.md, 실험군 간 학습 차이는 configurations.md를 참고한다.

Mac에서 MPS를 사용할 수 있으면 train.py의 기본 device가 mps이고, 그렇지 않으면 cpu다. 다른 device는 --device로 지정한다. training config의 output_dir에 checkpoint와 episode log가 저장된다.

## 11. 산출물, 모니터링과 재현

각 checkpoint에는 actor, critic, 두 optimizer 상태, 전체 experiment config, update 번호가 포함된다. 체크포인트의 config를 통해 inference 도구가 policy에 맞는 observation layout을 복원한다. 하향 IR 이전에 학습한 legacy checkpoint는 저장된 기존 sensor layout을 유지해 inference 시 입력 차원을 보존한다.

output_dir의 episodes.jsonl에는 각 완료 에피소드의 다음 정보가 기록된다.

- success, curriculum stage, 간극 파라미터와 목표·측정 용량
- capacity ratio, 건설 시간, 진행도, 낙하·앵커 실패 수
- 사용·앵커 로봇 수와 에너지 proxy
- 마지막 접촉 그래프와 morphology
- action 분포와 사용 가능한 branch latent 통계
- 누적 episode_reward와 항목별 reward_component_totals

업데이트별 반환값에는 actor_loss, critic_loss, entropy, auxiliary_loss, curriculum_stage가 있다. 각 step의 info에는 팀 reward_components와 로봇별 agent_rewards가 들어가므로, 새로운 보상 항을 조정할 때는 누적 항목과 success·용량·낙하 지표를 함께 확인한다.

재현 실험에서는 config 파일, seed, update 수, device, checkpoint 경로를 함께 기록한다. 현재 Trainer는 단일 환경 rollout을 사용하며, CLI에 학습 재개 옵션은 없다. 장기 학습 재개가 필요하면 checkpoint의 optimizer 상태와 experiment config를 명시적으로 복원하는 실행 경로를 추가해야 한다.

## 12. 해석 시 주의점

- actor가 전역 graph와 구조 용량을 보지 못한다는 경계를 유지해야 분산 실행 결과를 해석할 수 있다.
- local strain은 완전한 유한요소 응력이 아니라 접촉·앵커 힘에서 얻는 근사 신호다.
- 하향 IR의 최대값은 2.5D 지지면 부재를 나타내며, 실제 깊이 카메라의 3D 거리 측정은 아니다.
- 팀 보상은 간결한 구조를 유도하지만 특정 트러스 topology나 앵커 위치를 직접 지정하지 않는다.
- baseline과 bio policy, critic 종류, 센서 차원, 커리큘럼 유무를 바꾸는 실험은 같은 seed 집합과 평가 protocol로 비교해야 한다.
