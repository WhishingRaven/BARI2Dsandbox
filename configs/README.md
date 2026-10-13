# 실험 설정

모든 YAML은 단독으로 scripts/train.py의 --config 인자로 사용할 수 있다. 공통 환경, 보상, MAPPO 기본값은 bari2d/utils/config.py에서 상속하며, 각 파일은 비교에 필요한 차이만 명시한다.

| 파일 | actor | critic | 비교 목적 |
| --- | --- | --- | --- |
| baseline.yaml | gru | mlp | 순환 정책 기준선 |
| mlp.yaml | mlp | mlp | 기억 없는 하한선 |
| gru_traffic.yaml | gru_traffic | mlp | IR·행동 이력 교통 가지 |
| gru_traffic_connectivity.yaml | gru_traffic_connectivity | mlp | 교통·연결성 actor 가지 |
| bio.yaml | bio | graph | 모든 감각 가지, FiLM 없음 |
| bio_film.yaml | bio_film | graph | bio와 목표 하중 FiLM의 차이 |
| structured_mappo.yaml | structured_mappo | graph | BARI_26 기본 네 분기 Structured MAPPO |
| bio_heterogeneity.yaml | bio_heterogeneity | graph | bio_film과 개체별 latent의 차이 |
| curriculum.yaml | gru | mlp | 난이도 무작위화가 켜진 기준선 |
| beacon_structured_mappo_v2.yaml | structured_mappo | graph | 비콘 통신과 커리큘럼 (legacy 물리) |
| beacon_structured_mappo_static.yaml | structured_mappo | graph | v2와 같은 설정에 중력·정적 평형 물리 |
| beacon_structured_mappo_static_v2.yaml | structured_mappo | graph | static + 로봇 10대, 400스텝, frontier 진행도, 낮춘 벌점 |
| beacon_structured_mappo_static_v3.yaml | structured_mappo | graph | static + 로봇 5대, 가장자리 근처 출발, 둑을 잡은 구조물만 진행도, 엔트로피 0.01 |
| beacon_structured_mappo_static_v4.yaml | structured_mappo | graph | v3 + 준비된 시작 상태 커리큘럼(7단계), 로컬 구조 cue, 올라서기 보상 |
| beacon_structured_mappo_static_v5.yaml | structured_mappo | graph (로봇별 가치) | v4 과제 + 교사 시연 BC 사전학습, 로봇별 가치, 확장 connectivity 입력, nn.GRU 시퀀스 학습, 병렬 환경 8개 (docs/accelerated_training.md) |
| beacon_structured_mappo_static_v6.yaml | structured_mappo | graph (로봇별 가치) | v5 + 설계와 같은 앞쪽 배 IR(0.21)·"앞발이 턱을 넘을 때만 둑 잡기" 규칙, BC 계수 0.05 유지, 3~5단계 위주 시연 300개 (docs/accelerated_training.md 6절) |
| beacon_structured_mappo_static_v7.yaml | structured_mappo | graph (로봇별 가치) | v6 + 로봇 10대, 5~7단계 목표 20(1줄 14~15 < 20 ≤ 2줄 28~30, 보강 필요), max_steps 1200, 연장 시 climb 보상 유지, entropy 0.03, 2줄 짓는 교사 (docs/accelerated_training.md 7절) |
| beacon_structured_mappo_static_v8.yaml | structured_mappo | graph (로봇별 가치) | v7 + 줄별 틈 가운데 하중(`row_midspan`), 덧댐 결합(`splice_grip`), 5~7단계 목표 18(1줄 12~15 < 18; 덧댐·2줄로 도달), entropy 0.015, 시연 절반은 덧댐 교사 (docs/accelerated_training.md 8절) |
| beacon_structured_mappo_static_v9.yaml | structured_mappo | graph (로봇별 가치) | v8 + 서 있는 로봇의 climb 보상 1대 상한, 9단계 커리큘럼(1.7 한 줄 → 준비된 줄 보강(`row_spanning`) → 짓기+보강 → 2.3 → 3.0), 교사 줄 간격 2.0×폭 (docs/accelerated_training.md 9절) |
| beacon_structured_mappo_static_v10.yaml | structured_mappo | graph (로봇별 가치) | v9 + 추락 벌점 0.5 → 2.0, v9 체크포인트 1500에서 `--resume`으로 이어서 3000까지 (docs/accelerated_training.md 10절) |
| beacon_structured_mappo_static_v11.yaml | structured_mappo | graph (로봇별 가치) | v10 + 모듈 16대, 에피소드마다 12~16대만 참여(`robot.active_min`), 출발 구역 1.6배, 추락 벌점 0.5로 복귀, v10 체크포인트 3000에서 이어서 4500까지 (docs/accelerated_training.md 11절) |

기본 Structured MAPPO를 학습한다.

    conda run -n bari2d python scripts/train.py \
      --config configs/structured_mappo.yaml --updates 1000

공정한 비교에서는 같은 seed 집합, update 수, 평가 stage, 환경 버전을 사용해야 한다. actor 구조만 비교하려면 baseline, gru_traffic, gru_traffic_connectivity은 모두 MLP critic을 사용한다. bio 계열은 actor 가지와 graph critic을 함께 바꾸므로, critic 효과를 따로 측정하려면 동일 actor에서 critic 키만 mlp 또는 graph로 바꾼 별도 실험을 추가한다.

상세 비교표와 권장 절제 순서는 docs/configurations.md에 있다.
