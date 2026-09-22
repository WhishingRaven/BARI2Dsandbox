# 로봇 인터페이스 명세

이 문서는 BARI2D에서 한 대의 로봇이 무엇으로 구성되고, 정책이 받는 observation과 선택하는 action, 그리고 시뮬레이터가 유지하는 state가 어떻게 구분되는지 설명한다. 수치는 별도 YAML 설정이 없을 때의 기준값이다. 길이, 질량, 힘은 시뮬레이터 내부 단위이며 SI 단위로 보정된 물리 모델은 아니다.

## 세 종류의 정보

이 프로젝트에서 state라는 말은 용도에 따라 세 가지를 뜻한다.

| 이름 | 누가 사용하나 | 범위 | 기본 크기 |
| --- | --- | --- | ---: |
| RobotState | 시뮬레이터 | 한 로봇의 완전한 내부 상태 | 구조체 |
| local observation | 공유 actor 정책 | 한 로봇이 행동을 고르는 데 쓸 수 있는 국소 정보 | 39 |
| global state | 중앙 critic과 분석 코드 | 모든 로봇과 교량의 전역 요약 | 187 |

분산 실행 원칙상 actor에는 local observation과 자신의 순환 은닉 상태만 주어진다. 전역 위치, 모든 접촉 연결, 다른 로봇의 상태와 현재 구조 용량은 actor 입력에 포함되지 않는다.

## 기준 로봇 사양

| 항목 | 설정 키 | 기본값 | 의미 |
| --- | --- | ---: | --- |
| 로봇 수 | robot.count | 20 | 한 에피소드의 동일형 로봇 수 |
| 길이 | robot.length | 0.90 | 방향 축을 따른 회전 사각형 본체 길이 |
| 폭 | robot.width | 0.42 | 방향 축에 수직인 본체 폭 |
| 질량 | robot.mass | 1.00 | 등반 에너지와 무작위화에 쓰이는 상대 질량 |
| 최대 선속도 | robot.max_speed | 0.60 | 정방향 또는 후진 운동의 속도 크기 |
| 최대 조향각 | robot.max_steering_deg | 20° | left/right 명령의 조향 상태 절댓값 |
| 회전율 | robot.turn_rate | 1.20 rad/시간 단위 | 정규화 조향 명령이 만드는 각속도 계수 |
| 제어 간격 | environment.time_step | 0.10 | 한 환경 step의 시간 간격 |
| 앵커 반경 | robot.anchor_range | 1.05 | 다른 로봇 중심까지 앵커를 만들 수 있는 최대 거리 |
| 등반 높이 | robot.climb_height | 0.35 | CLIMB 에너지에 쓰이는 층간 높이 |
| 최대 층 | robot.max_layer | 10 | 바닥 0층 위에 허용되는 최고 층 번호 |

로봇 본체는 중심 위치, 방위 \(\theta\), 길이와 폭으로 정해지는 oriented bounding box다. 접촉은 이 상자의 중첩으로 판정한다. 환경은 2차원 위치를 적분하되 layer라는 이산 높이 상태를 함께 둔 2.5D 모델이다.

운동 action에서, 잡음이 반영된 정규화 명령 \(u_v,u_s \in [-1,1]\)에 대해 다음 순서로 상태를 갱신한다.

\[
v_{t+1}=0.60u_v,\qquad
\delta_{t+1}=\operatorname{rad}(20^\circ)u_s,
\]

\[
\theta_{t+1}=\operatorname{wrap}\left(\theta_t+1.20u_s\cdot0.10\right),\qquad
p_{t+1}=p_t+v_{t+1}\begin{bmatrix}\cos\theta_{t+1}\\\sin\theta_{t+1}\end{bmatrix}0.10.
\]

따라서 잡음이 없을 때 한 step의 직진 이동량은 0.06이다. 운동 에너지는 \(\lvert v\rvert\Delta t+0.1\lvert\delta\rvert\Delta t\)만큼 누적된다. 기준 설정에서는 센서와 actuator 잡음이 모두 0이며, 커리큘럼 4단계에서는 최소 0.01의 잡음과 마찰·앵커 강도 무작위화가 적용된다.

## RobotState: 시뮬레이터의 완전 상태

RobotState는 bari2d/env/robot.py에 정의된다. 아래 항목은 정책 관측 여부와 무관하게 환경이 유지하는 값이다.

| 필드 | 형식·범위 | 갱신 방식 | actor가 직접 보나 |
| --- | --- | --- | --- |
| robot_id | 0부터 N-1 | 생성 시 고정 | 아니오 |
| position | \([x,y]\) | 운동, 등반, 접촉 해소 후 필드 경계로 clamp | 아니오 |
| theta | \([-\pi,\pi)\) | left/right 운동 때 갱신 | 아니오 |
| velocity | \([-0.60,0.60]\) | 운동 action이 설정, 낙하 때 0 | 정규화 값만 |
| steering | \([-\operatorname{rad}(20°),\operatorname{rad}(20°)]\) | 운동 action이 설정 | 정규화 값만 |
| head_lifted | bool | 이번 action이 CLIMB이면 true | 예 |
| anchored | bool | ANCHOR와 RELEASE, 앵커 파손으로 갱신 | 예 |
| strain | 0 이상 | 접촉 및 앵커 edge의 힘을 합산해 매 step 재계산 | 이력의 정규화 값만 |
| previous_action | 0–9 | 유효 action 또는 IDLE | 정규화 이력만 |
| layer | 0–10 | 성공한 등반 때 지지 로봇 층 + 1, 지지 범위를 벗어나면 자동 하강 | 정규화 값만 |
| fallen | bool | 지지 없이 간극에 있으면 true | 예 |
| energy | 0 이상 | 이동·등반 비용을 누적 | 아니오 |
| anchor_partner | 로봇 ID, LEFT_BANK, RIGHT_BANK 또는 null | 앵커 연결/해제/파손 | 아니오 |
| recurrent_hidden | 벡터 또는 null | 구현상 actor rollout이 별도로 보관 | actor의 내부 상태 |
| latent | \(\mathbb{R}^{Z}\) | reset 때 로봇마다 표본화, 에피소드 동안 고정 | 이질성 actor에서만 |

ANCHOR, RELEASE, CLIMB, IDLE은 운동학 적분을 호출하지 않는다. 따라서 velocity와 steering은 별도의 정지 명령으로 자동 초기화되지 않고 가장 최근 운동 action의 상태를 유지한다. fallen 로봇은 velocity가 0으로 설정되고 앵커가 해제된다.

로봇은 간극 위에 있으면서 어느 제방에도 연결되지 않은 접촉 그래프 component에 속하면 낙하한다. 낙하 뒤에는 IDLE만 허용된다.

## Local observation: actor가 받는 벡터

reset과 step은 float32 배열 \(O\in\mathbb{R}^{N\times D}\)를 반환하며, i번째 행이 i번 로봇의 관측이다. 차원은 다음과 같다.

\[
D=H R+H+H+6+1+Z=H(R+2)+7+Z.
\]

기준 설정에서는 history \(H=4\), IR ray 수 \(R=5\), latent 차원 \(Z=4\)이므로 \(D=39\)이다. 이력은 오래된 값부터 최신 값 순서이고, 각 step 후 새 측정값이 마지막 칸에 들어간다.

| Python slice | 기본 인덱스 | 길이 | 값과 정규화 |
| --- | --- | ---: | --- |
| ir | [0:20) | 20 | 최근 4개 관측 × 전방·후방·좌·우·하향 IR 5개. 각 거리를 ir_range 3.0으로 나누어 [0, 1] |
| strain | [20:24) | 4 | 최근 4개 변형률. 아래 strain scale로 나누고 [0, 2] clip |
| action_history | [24:28) | 4 | 최근 action ID를 9로 나눈 값, 즉 [0, 1] |
| internal | [28:34) | 6 | 앵커·등반·속도·조향·층·낙하 상태 |
| goal | [34:35) | 1 | target_load / load.max_target |
| latent | [35:39) | 4 | 에피소드 내 고정 개인 latent |

여기서 [a:b)는 b를 포함하지 않는 Python slice 표기다. 설정에서 ray 수, history 또는 latent_dim을 바꾸면 모든 slice와 observation 크기가 함께 바뀐다.

### IR 관측

수평 ray는 로봇 방위에 ir_angles_deg를 더한 방향으로 0.025 간격으로 ray march한다. 다음 중 먼저 만나는 대상까지의 거리를 3.0으로 나눈다.

- 필드 밖 경계
- 현재 제방에서 간극 또는 반대 제방으로 넘어가는 경계
- 원형 장애물
- layer 차이가 1 이하인, 낙하하지 않은 다른 로봇의 본체

기준 수평 ray의 순서는 전방 [0°], 후방 [180°], 좌측 [90°], 우측 [-90°]다. 다섯 번째 하향 IR은 로봇 중심 아래의 제방 또는 반경 1.125 안에 있는 더 낮은 layer 로봇까지의 수직 거리를 반환한다. 아래에 표면이 없으면 최대 거리 1.0을 반환하며, 이는 절벽 또는 지지 상실 신호다. reset 직후 이력은 1.0으로 채운 후 최신 slot에 실제 첫 측정값을 기록한다.

### Strain과 행동 이력

strain은 contact edge와 anchor edge가 로봇에 가하는 힘의 합이다. 환경은 매 step 그래프를 다시 만들기 전에 strain을 0으로 초기화하고, 다시 계산한 값을 history에 넣는다. 기준 앵커 scale은

\[
\sqrt{\min(12,12)^2+8^2}=\sqrt{208}\approx14.42
\]

이고, 실제 정규화에는 이 값, contact_capacity 5, 1 중 최댓값을 쓴다. action_history에서 0은 FORWARD, 9/9=1은 IDLE이다. reset 직후 action history는 모두 IDLE이다.

### Internal 6차원

internal slice의 순서는 고정되어 있다.

| 상대 인덱스 | 값 | 범위 |
| ---: | --- | --- |
| 0 | anchored | 0 또는 1 |
| 1 | head_lifted | 0 또는 1 |
| 2 | velocity / max_speed | [-1, 1] |
| 3 | steering / rad(max_steering_deg) | [-1, 1] |
| 4 | layer / max_layer | [0, 1] |
| 5 | fallen | 0 또는 1 |

goal은 기준 1단계의 target_load 3.0에서 0.2다. 2단계 이상에서는 target_load가 [3, 10]에서 표본화되며 goal은 [0.2, 약 0.667]이다. latent는 기본 latent_sigma가 0이므로 네 값 모두 0이다. use_heterogeneity가 true인 actor만 이 latent를 정책 입력에 결합한다.

## Action space와 유효성 마스크

action은 로봇당 하나의 정수이며 shape은 (N,)이다. action_count는 10이다. 정책은 mask가 false인 action의 logits를 제외한 뒤 분포에서 표본화해야 한다. 환경도 방어적으로 mask가 false인 입력을 IDLE(9)로 바꾼다.

| ID | 이름 | 명령 | 효과 |
| ---: | --- | --- | --- |
| 0 | FORWARD | \(u_v=+1,u_s=0\) | 전진 |
| 1 | BACKWARD | \(u_v=-1,u_s=0\) | 후진 |
| 2 | FORWARD_LEFT | \(u_v=+1,u_s=+1\) | 전진하며 좌회전 |
| 3 | FORWARD_RIGHT | \(u_v=+1,u_s=-1\) | 전진하며 우회전 |
| 4 | BACKWARD_LEFT | \(u_v=-1,u_s=+1\) | 후진하며 좌회전 |
| 5 | BACKWARD_RIGHT | \(u_v=-1,u_s=-1\) | 후진하며 우회전 |
| 6 | CLIMB | 구조 action | 전방 지지 로봇 위로 한 층 등반 |
| 7 | ANCHOR | 구조 action | 제방 또는 인접 로봇과 앵커 연결 |
| 8 | RELEASE | 구조 action | 자신의 앵커 해제 |
| 9 | IDLE | \(u_v=0,u_s=0\) | 별도 운동학 갱신 없음 |

CLIMB는 앞쪽에 있고, 거리가 \(0.9\times1.25=1.125\) 이하이며, 아직 최고 층이 아닌 로봇이 있을 때만 유효하다. 성공하면 지지 로봇의 layer+1로 올라가며 현재 방위로 0.27만큼 이동한다. 에너지는 무작위화된 질량과 climb_height의 곱만큼 증가한다. 앵커되지 않은 로봇은 바로 아래 layer의 로봇이 반경 1.125 안에 없으면 action 뒤 자동으로 한 layer씩 내려온다. 이 하강은 별도 action이 아니다.

ANCHOR는 이미 anchored가 아니고 후보가 있을 때만 유효하다. 후보 선택은 제방 본체 접촉을 우선하고, 없으면 중심 거리가 1.05 이하이며 layer 차이가 1 이하인 낙하하지 않은 로봇 중 가장 가까운 것을 고른다. 한 로봇은 하나의 소유 앵커만 가질 수 있다.

마스크 규칙을 요약하면 다음과 같다.

| 로봇 조건 | 허용 action |
| --- | --- |
| fallen | IDLE만 |
| anchored | RELEASE와 IDLE만 |
| 일반 상태 | 여섯 운동 action과 IDLE, 조건을 만족할 때 CLIMB·ANCHOR |
| anchored가 아닌 일반 상태 | RELEASE는 금지 |

접촉 해소와 앵커 파손 판정은 action 적용 뒤에 발생할 수 있다. 따라서 actor는 관측만으로 실제 접촉 그래프와 미래 앵커 파손을 완벽히 알 수 없으며, 이것이 부분 관측 제어 문제의 일부다.

## Global state: 중앙 critic과 분석용 정보

global_state()는 기본적으로 187차원이다.

\[
20\times9+7=187.
\]

각 로봇에 대해 아래 9개를 순서대로 넣는다.

1. \(x/\text{field_length}\)
2. \(y/\text{field_width}\)
3. \(\sin\theta\)
4. \(\cos\theta\)
5. strain / anchor_capacity
6. anchored
7. velocity / max_speed
8. layer / max_layer
9. fallen

마지막 7개 전역 값은 gap_width / field_length, gap orientation의 sin과 cos, target_load / max_target, 현재 spanning progress, 현재 capacity / max_target, 그리고 좌우 제방이 연결되었는지를 나타내는 graph.spans다. 이 벡터는 중앙 집중 critic이나 실험 분석에 쓸 수 있지만 공유 actor에는 전달하지 않는다.

접촉 그래프 기반 critic을 위한 graph_state()는 이와 별개로 노드 feature 13개와 edge feature 6개를 제공한다. 이 역시 실행 시 정책 관측이 아니다.

## 정책 호출 순서

한 환경 step에서의 데이터 흐름은 아래와 같다.

1. actor는 현재 local observation, action mask, 자신의 GRU hidden을 받아 10개 action의 확률을 만든다.
2. 선택한 action을 BridgeEnv.step에 N개 묶어 전달한다.
3. 환경은 마스크를 다시 검사하고, 구조 action 또는 운동학을 적용한다.
4. 접촉을 해소하고, 접촉·앵커 그래프와 strain을 다시 계산하며, 지지 없는 로봇을 낙하시킨다.
5. 하중 용량과 진행도를 평가하고, IR·strain·action history를 갱신하여 다음 local observation을 반환한다.

공유 actor의 기본 GRU hidden 차원은 64이며, 로봇마다 별도로 이어진다. 이 hidden은 물리적 RobotState나 local observation 39차원에 포함되지 않고 rollout 또는 inference 세션이 보관한다. 에피소드 reset 시 hidden도 0으로 초기화해야 한다.

## 설정 및 확인

기준값은 configs/baseline.yaml과 bari2d/utils/config.py의 RobotConfig, SensorConfig, ContactConfig, EnvironmentConfig에 정의되어 있다. 실행 중 실제 인터페이스 크기는 다음처럼 확인할 수 있다.

    conda run -n bari2d python -c "
    from bari2d.env.bridge_env import BridgeEnv
    env = BridgeEnv()
    observation, _ = env.reset()
    print(observation.shape)       # (20, 39)
    print(env.action_masks().shape)  # (20, 10)
    print(env.global_state().shape)  # (187,)
    "

구현의 정본은 bari2d/env/robot.py, bari2d/env/sensors.py, bari2d/env/contact_model.py, bari2d/env/bridge_env.py다. 새 policy 또는 센서를 추가할 때는 observation layout, actor의 입력 분해, checkpoint와 문서를 함께 변경해야 한다.
