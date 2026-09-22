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
| bio_heterogeneity.yaml | bio_heterogeneity | graph | bio_film과 개체별 latent의 차이 |
| curriculum.yaml | gru | mlp | 난이도 무작위화가 켜진 기준선 |
| bio_policy.yaml | bio_film | graph | 이전 이름과의 호환용 alias |

예를 들어 bio_film을 학습한다.

    conda run -n bari2d python scripts/train.py \
      --config configs/bio_film.yaml --updates 1000

공정한 비교에서는 같은 seed 집합, update 수, 평가 stage, 환경 버전을 사용해야 한다. actor 구조만 비교하려면 baseline, gru_traffic, gru_traffic_connectivity은 모두 MLP critic을 사용한다. bio 계열은 actor 가지와 graph critic을 함께 바꾸므로, critic 효과를 따로 측정하려면 동일 actor에서 critic 키만 mlp 또는 graph로 바꾼 별도 실험을 추가한다.

상세 비교표와 권장 절제 순서는 docs/configurations.md에 있다.
