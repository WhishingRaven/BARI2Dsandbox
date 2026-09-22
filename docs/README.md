# BARI2D 문서

BARI2D는 약 20대의 동일한 크롤링 로봇이 국소 센서와 공유 정책만으로 무작위 간극을 가로지르는 하중 지지 구조물을 만들도록 학습시키는 연구 프레임워크다.

## 문서 구성

- [architecture.md](architecture.md): 전체 시스템 경계, 모듈과 데이터 흐름
- [simulation.md](simulation.md): 간극, 로봇, 센서, 접촉, 앵커와 하중 시험
- [training.md](training.md): 분산 행위자, 중앙 비평가, MAPPO와 커리큘럼
- [experiments.md](experiments.md): 검증, 절제 실험, 지표와 재현 절차
- [inference.md](inference.md): 체크포인트 선택과 그래픽 정책 추론
- [manual_control.md](manual_control.md): 키보드·마우스로 한 로봇씩 수동 조종
- [robot_interface.md](robot_interface.md): 로봇 사양, RobotState, actor 관측, 행동 마스크와 중앙 critic 상태
- [LaTex/bari2d_mathematical_formulation.tex](LaTex/bari2d_mathematical_formulation.tex): 상태, 관측, 동역학, 그래프 구조, 하중 용량, 보상과 MAPPO 목적의 수학적 정식화

## 핵심 연구 질문

각 로봇은 전체 지도, 다른 로봇의 전역 위치, 전체 접촉 그래프 또는 현재 구조 용량을 알지 못한다. 각자 관측할 수 있는 정보는 짧은 IR 및 변형률 이력, 내부 상태, 이전 행동과 공통 목표 하중뿐이다.

프로젝트가 다루는 질문은 다음과 같다.

> 국소 감각과 국소 기계 행동만으로 전역적인 연결성과 목표 하중을 만족하는 구조 형상이 창발할 수 있는가?

특정 트러스, 삼각형, 각도 또는 교량 토폴로지는 보상과 제어기에 인코딩하지 않는다. 성공은 오직 제방 간 연결, 구조 용량, 시간, 에너지, 사용 로봇 수와 안정성으로 판정한다.

## 빠른 실행

    conda activate bari2d
    python -m pip install -e '.[dev]'
    pytest
    python scripts/verify_scripted.py

기준 정책 훈련:

    conda run -n bari2d python scripts/train.py \
      --config configs/baseline.yaml --updates 1000

전체 생물 영감형 정책 훈련:

    conda run -n bari2d python scripts/train.py \
      --config configs/bio_policy.yaml --updates 1000

상세한 사용법과 현재 한계는 저장소 루트의 [README.md](../README.md)를 함께 참고한다.
