# 가속 학습과 시연 기반 학습 (v5)

`configs/beacon_structured_mappo_static_v5.yaml`은 v4와 같은 과제(7단계 시작 상태 커리큘럼, 구조 cue, 보상, 정적 물리)를 쓰면서 학습 방식만 바꾼다. v4는 1500 업데이트 동안 2단계까지만 갔다. 원인은 세 가지로 봤다. 초반에 IDLE로 굳었고, 5대가 팀 어드밴티지 하나를 나눠 받았고, 업데이트마다 400스텝 에피소드 하나만 봤다.

## 실행 순서

    conda activate bari_26
    python scripts/pretrain_bc.py --config configs/beacon_structured_mappo_static_v5.yaml
    python scripts/train.py --config configs/beacon_structured_mappo_static_v5.yaml \
      --pretrained runs/beacon_structured_mappo_static_v5/bc_pretrain.pt

- `pretrain_bc.py`는 시연을 만들어 `training.demo_path`에 저장하고, 행동 복제(BC)로 학습한 actor를 `bc_pretrain.pt`로 저장한다.
- 시연을 다시 만들지 않으려면 `--reuse-demos`를 붙인다. 워커 수는 `--workers`, 에피소드 수는 `--episodes`(기본 240)로 바꾼다.
- `--eval-episodes 5`를 붙이면 단계별로 BC 정책의 성공률을 출력한다.
- 이어서 학습할 때는 `--resume runs/.../checkpoint_XXXXXX.pt --updates 목표`를 쓴다. 이제 체크포인트에 커리큘럼 단계가 저장되어 그 단계부터 이어진다.

## 1. 시연 기반 사전학습 (reverse curriculum + behaviour cloning)

- **교사** (`bari2d/rl/teacher.py`): 시뮬레이터 전체 상태를 보는 스크립트 건설자다. 한 번에 한 대가 작업하고, 최대 두 대(`helpers`)가 짓고 있는 줄의 뿌리 뒤에 줄을 서서 기다린다(7절).
  1. 둑을 잡은 로봇이 없으면 가장자리에 가장 가까운 로봇이 가장자리로 가서 ANCHOR 한다.
  2. 구조물 위에 선 로봇이 있으면 앞끝까지 가서 grip-and-extend 한다.
  3. 없으면 가장 가까운 로봇이 뿌리 로봇 뒤로 가서 CLIMB 한다.
  - 보상에는 쓰이지 않는다. 오직 행동 레이블을 만드는 데만 쓴다.
- **시연 생성** (`bari2d/rl/demonstrations.py`): 각 단계의 시작 상태에서 교사를 돌린다.
  - 움직이는 로봇은 확률 ε(기본 0.1), 기다리는 로봇은 ε/4로 실제 실행 행동을 무작위 유효 행동으로 바꾼다. 이때 레이블은 교사 행동 그대로 둔다. 그래서 교사 경로 주변(tube)과 벗어났을 때의 복귀를 함께 보여 준다.
  - RELEASE는 주입하지 않는다.
  - 관측은 항상 시뮬레이터가 만든다. 센서 이력과 GRU 상태가 물리적으로 일관되게 유지된다.
- **손실**: `L_BC = −Σ w·log π(a_teacher | o, h) / Σ w`.
  - 교사가 움직이는 로봇의 가중치는 1이다.
  - 기다리는 로봇의 IDLE 레이블은 `bc_idle_weight`(기본 0.2)로 낮춘다. IDLE로 굳는 것을 막기 위해서다.
  - 사전학습에서는 보조 cue 손실도 함께 학습한다.
- **PPO 미세조정**: `L = L_PPO + bc_coef·L_BC + auxiliary_coef·L_aux`.
  - 처음 `critic_warmup_updates`(v5: 10) 업데이트 동안은 critic만 학습한다. 새 critic은 성공 보상(+30) 때문에 처음 손실이 크고, 이 상태로 사전학습된 정책을 흔들지 않게 하려는 것이다.
  - `bc_coef`는 `bc_decay_updates` 동안 `bc_coef_final`까지 선형으로 줄어든다. v5는 0.5에서 300 업데이트 동안 0으로 간다.
- 교사 성공률 (5대, ε=0 기준): 1–4단계는 각각 4/4, 4/4, 5/5, 5/5(약 90–150스텝), 5단계는 3/3이다. 6단계(틈 2.3)는 400스텝이 빠듯해서 1/3이다.

## 2. 에이전트 조건부 중앙 가치 (agent-conditioned centralized value)

- `training.critic_agent_conditioned: true`이면 그래프 critic이 로봇마다 `V_i = f(h_i, mean_j h_j)`를 낸다. 메시지 전달까지는 기존과 같고, 마지막 value head만 `[h_i, h_pooled]`를 입력으로 받는다.
- GAE는 로봇별로 계산한다. 팀 보상은 같아도 가치가 다르므로 로봇마다 어드밴티지가 달라진다.
- 시간 제한 bootstrap도 로봇별로 `γ·V_i(s_T)`를 더한다.
- actor는 여전히 국소 관측만 쓴다. 분산 실행은 그대로다.

## 3. connectivity branch 입력

- `model.connectivity_inputs: ir_strain_action_cue`이면 structured_mappo의 connectivity branch가 이력 한 스텝마다 다음을 읽는다.
  - IR 5방향
  - 자기 strain
  - 직전 행동(one-hot 10)
  - 국소 접촉 cue 2개
- 합계 18채널이다. 보조 타깃(접촉 수, 지속 접촉 수)을 관측으로 추정할 수 있게 하려는 것이다.
- 기본값 `ir`은 예전 branch 그대로라서 v3/v4 체크포인트가 그대로 로드된다.

## 4. 시퀀스 단위 인코딩과 nn.GRU

- 학습 시 `(T, N, D)` 관측을 `(T·N, D)`로 펴서 모든 branch를 한 번에 인코딩한다. 그다음 `(T, N, H)`를 `nn.GRU` 한 번에 넣는다.
- 에피소드가 시퀀스 중간에 끝난 열은 그 스텝부터 0 hidden으로 다시 돌려 이어 붙인다. 스텝별 계산과 수치가 같은지 테스트로 확인했다.
- `training.recurrent_chunk`(v5: 32)로 truncated BPTT를 한다. 각 조각은 rollout 때 저장한 hidden에서 시작한다.
- `training.num_minibatches`(v5: 4)는 (조각, 환경, 로봇) 시퀀스를 섞어 나눈다.
- `nn.GRUCell` 시절 체크포인트는 가중치 이름을 자동으로 바꿔 로드한다(`weight_ih` → `weight_ih_l0` 등).

## 5. 병렬 rollout

- `training.num_envs`개의 환경을 함께 돌린다(v5: 8개 × 128스텝 = 업데이트당 1024 팀 전이).
- `training.vector_env`
  - `sync`: 한 프로세스에서 차례로 돌린다.
  - `subprocess`: 환경마다 워커 프로세스를 띄운다(spawn 방식이라 Windows에서도 동작한다).
- 두 방식은 같은 seed에서 같은 결과를 낸다.
- 환경 i의 seed는 `seed + 1000·i`이고, 환경 0은 기존 seed를 쓴다.
- 커리큘럼은 모든 환경의 에피소드를 합쳐 판단한다. 단계가 오르면 모든 워커에 알리고, 다음 리셋부터 새 단계로 시작한다. 이전 단계에서 진행 중이던 에피소드는 판단에서 제외한다.
- `train.py --num-envs N`으로 설정을 덮어쓸 수 있다. CPU 코어 수에 맞추면 된다.

## 새로 기록되는 값

- `updates.jsonl`: `policy_loss`, `approx_kl`, `clip_fraction`, `explained_variance`, `bc_loss`, `bc_coef`, `episodes`.
- `episodes.jsonl`: 에피소드를 돌린 환경 번호 `env_index`와, 그 에피소드가 실제로 진행된 `curriculum_stage`.

## 6. v6: 설계와 같은 둑 잡기 규칙과 BC 하한

`configs/beacon_structured_mappo_static_v6.yaml`

v5는 update 376에 4단계(빈 상태에서 짓기)에 올라갔지만, update 792까지 4단계 성공률이 5% 안팎에 머물렀다. 4단계 에피소드의 62%에서 로봇이 가장자리보다 뒤에서 둑을 잡았다. 이런 grip이 있는 에피소드의 성공률은 2.7%, 없는 에피소드는 7.9%였다. 원인은 시뮬레이터가 설계와 달랐기 때문이다.

- 아래 IR이 로봇 **중심**에 있었다. 설계는 15 cm 몸체의 앞에서 4 cm다.
- 둑 ANCHOR는 수평 IR 네 방향 중 **아무 방향**에서나 중심에서 1.2 안에 절벽이 보이면 허용됐다. 그래서 앞끝이 가장자리보다 0.75 뒤에 있거나 옆·뒤를 보고 있어도 잡을 수 있었다.

v6에서 바꾼 것:

- `sensor.downward_ir_offset: 0.21`: 배쪽 IR을 중심보다 0.21 앞(= 0.45 − 0.9·4/15)에 둔다. 그 점 바로 아래의 제방이나 하위 로봇만 표면으로 본다.
- `contact.anchor_rule: front_downward_ir`: layer 0 로봇은 앞쪽 배 IR이 바닥을 보지 못하고 중심은 아직 둑 위에 있을 때만 둑을 잡는다. 앞발이 턱을 넘은 자세이며, 중심이 가장자리 0~0.21 뒤에서 앞을 향한 경우다. 노트의 "아래에 아무것도 없으면 멈춰서 가시로 결합" 규칙과 같다. 로봇끼리의 grip 조건은 그대로다.
- `bc_coef_final: 0.05`: BC 계수가 300 업데이트 동안 0.5에서 0이 아니라 0.05로 줄고, 이후 그 값을 유지한다. 보상만으로는 드물게 나오는 교사 행동(쓸모없는 grip 풀기, 뿌리 뒤로 가서 CLIMB)을 계속 약하게 당겨 준다. PPO 어드밴티지가 강하게 반대하면 PPO가 이긴다.
- `demo_episodes: 300`, `demo_stages: [1, 2, 3, 3, 4, 4, 4, 5, 5, 6]`: 아래 IR 값의 의미가 바뀌었으므로 시연을 새로 만든다. 4단계 90개, 3·5단계 각 60개, 1·2·6단계 각 30개다. 교사가 풀지 못하는 7단계는 뺐다. `pretrain_bc.py --stages`로 덮어쓸 수 있고, 같은 단계를 여러 번 쓰면 그만큼 비중이 커진다.
- `entropy_coef`와 나머지 설정은 v5와 같다. 설계 규칙과 BC 하한의 효과만 보기 위해서다.

확인한 것:

- 교사(ε=0)는 새 규칙에서 1~4단계 12/12 성공, 5단계 2/3이다. 같은 seed에서 예전 규칙과 결과가 같다. ε=0.1이면 1~4단계 12/12, 5단계 1/3이다.
- 둑 ANCHOR 마스크가 더 이상 수평 ray 탐색을 하지 않아 환경 스텝이 약 1.7배 빨라졌다(4단계 무작위 행동 기준 55 → 31 ms/step).
- `tests/test_front_anchor.py`: 가장자리 거리·방향별 마스크, 기본값(`edge_band`)에서 예전 동작 유지, 센서 위치와 값, 위층 로봇 제외, 잘못된 설정 거부, 교사 4단계 성공, 단계 가중 시연 생성, BC 하한.

실행 순서 (시연 생성과 BC부터 다시):

    conda activate bari_26
    python scripts/pretrain_bc.py --config configs/beacon_structured_mappo_static_v6.yaml --eval-episodes 5
    python scripts/train.py --config configs/beacon_structured_mappo_static_v6.yaml --pretrained runs/beacon_structured_mappo_static_v6/bc_pretrain.pt

- 시연 생성 중에는 10%마다 `k/300 teacher episodes done`이 출력된다.
- v5 결과(`runs/beacon_structured_mappo_static_v5`)는 그대로 남는다. v6는 `runs/beacon_structured_mappo_static_v6`에 쓴다.
- v5 체크포인트는 기본값이 예전 규칙이므로 그대로 로드되고 재생된다. 다만 v6 환경에서는 아래 IR 값의 의미가 다르므로 v6는 BC부터 새로 학습한다.

## 7. v7: 로봇 10대와 보강이 필요한 과제

`configs/beacon_structured_mappo_static_v7.yaml`

v6는 update 660에서 4단계 성공률이 약 25%였다(v5는 약 5%). 남은 실패는 대부분 아무도 뿌리에 올라타지 않는 경우(56%)였다. 가장자리에 여러 대가 나란히 붙는 현상도 있었다. 나란히 붙은 뿌리는 각자 자기 줄을 이어야 보강이 된다. 그런데 v6 목표 하중(1~3)은 1줄 용량(14~15)보다 훨씬 작아서 보강할 이유가 없었다.

바꾼 것:

- **로봇 10대** (`robot.count: 10`).
- **IR·절벽 센서 벡터화:** ray-march를 모든 스텝에 대해 한 번에 계산한다. 결과는 예전 스칼라 방식과 비트 단위까지 같다(`tests/test_reinforcement_task.py`에서 확인). 한 스텝 시간은 5대 31 → 9 ms, 10대 120 → 25 ms로 줄었다.
- **보강이 필요한 5~7단계:**
  - `load.max_target: 40`: 용량 상한이자 목표 정규화 값이다(예전 15).
  - `curriculum_targets: [1, 1, 1, 1, 20, 20, 20]`.
  - 틈 1.7·2.3·3.0에서 1줄 용량은 14~15, 2줄은 28~30이므로 20을 넘으려면 보강이 필요하다. 7단계는 앵커 강도가 ±20% 무작위화되는데, 이때도 1줄 ≤ 17 < 20 ≤ 22.8이다.
  - 앵커 한계는 5 그대로 둔다. 한계를 3 이하로 낮추면 2.3·3.0 틈에서 1줄을 짓는 도중에 캔틸레버 앵커가 끊어진다(교사로 확인).
  - 정적 모델에서는 뿌리와 다음 로봇 이음부 위에 올라선 로봇도 용량을 키운다(2.3·3.0 틈에서 약 +50%). 그래서 1줄 + 위층 로봇으로 20에 닿는 경우도 있다. 이것도 보강으로 본다. 다만 실제 물리에서도 그런지는 따로 검증이 필요하다.
- **`max_steps: 1200`:** 교사는 2줄을 1.7 틈에서 약 350~600스텝, 2.3 틈에서 약 500스텝, 3.0 틈에서 약 500~600스텝에 짓는다. `demo_epsilon` 0.05 교란이 있으면 370~990스텝이 걸리고, 6~7단계 일부 시연은 시간 안에 끝나지 않는다(실패 시연도 레이블은 유효하다).
- **`reward.climb_counts_extended: true`, `climb_cap: 8`:** 올라탔다가 끝에서 끝과 끝을 맞대 연장한 로봇도 climb potential에 센다. 그래서 연장할 때 climb 보상(0.8)이 되돌려지지 않는다. 떨어지거나 놓으면 되돌려진다. 두 줄의 모든 연결이 한 번씩 0.8을 받는다.
- **`entropy_coef: 0.03`:** v6에서 entropy가 약 0.1에 머물렀기 때문이다.
- **교사:**
  - 모든 줄이 반대편에 닿았는데 용량이 목표보다 작으면, 가장 바깥 뿌리 옆(`row_spacing` = 폭 × 1.6)에 새 뿌리를 잡고 같은 방식으로 두 번째 줄을 짓는다.
  - 로봇이 올라타거나 걸어 나가는 동안 다음 로봇 두 대가 뿌리 뒤 대기 지점으로 미리 간다. 이 덕분에 5대 설정(v6)에서도 6단계까지 교사가 성공한다.
- **시연:** 300개를 단계 `[1, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7]`에 나눠 만든다. `demo_epsilon: 0.05`는 움직이는 로봇의 교란 확률이고, 기다리는 로봇은 그 1/4이다.
- **PPO 중 BC 항:** 시연 8개에서 256스텝 구간을 무작위로 잘라 쓴다(`bc_window_steps`). 중간에서 시작한 구간은 GRU가 0 상태에서 출발하므로, 앞부분(최대 32스텝)은 손실 없이 GRU 예열에만 쓴다.

실행:

    python scripts/pretrain_bc.py --config configs/beacon_structured_mappo_static_v7.yaml --eval-episodes 5
    python scripts/train.py --config configs/beacon_structured_mappo_static_v7.yaml --pretrained runs/beacon_structured_mappo_static_v7/bc_pretrain.pt

- 시연 파일은 수백 MB가 된다. 에피소드가 길고 로봇이 10대이기 때문이다.
- BC 30 epoch는 v6보다 몇 배 오래 걸린다. 급하면 `--epochs 20`으로 줄인다.
- 로그의 `structure.edge_grips`가 2 이상이고 성공했다면 두 줄 또는 보강 구조를 지은 것이다.
