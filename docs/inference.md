# 그래픽 추론 뷰어

저장된 정책을 선택하고 실제 분산 추론을 애니메이션으로 보려면 저장소 루트에서 다음을 실행한다.

    conda run -n bari2d python infer.py \
      --checkpoint runs/baseline/checkpoint_001000.pt

체크포인트를 여러 개 지정하면 좌측 라디오 버튼에서 모델을 바꿀 수 있다. 기본적으로 runs 아래에서는 실행 디렉터리별 최신 체크포인트 하나를 자동으로 찾는다. 이전 체크포인트도 비교하려면 직접 지정한다.

    conda run -n bari2d python infer.py \
      --checkpoint runs/baseline/checkpoint_001000.pt \
      --checkpoint runs/bio_film/checkpoint_001000.pt

뷰어는 다음을 표시한다.

- 현재 환경의 제방, 간극, 로봇, 앵커와 기계 접촉
- 직전 스텝에서 각 로봇이 실행한 행동 약어
- 선택한 로봇의 다음 행동 확률 분포
- 목표 하중, 하중 용량, 횡단 진행도, 보상과 정책 엔트로피
- 생물 영감형 정책에서 사용할 수 있는 가지별 잠재값 평균

컨트롤:

- 모델 라디오 버튼: 체크포인트 교체 및 동일 시드 환경 재시작
- robot 슬라이더: 행동 확률을 확인할 로봇 선택
- target 슬라이더: 목표 하중 조건을 바꾸고 정책 입력 갱신
- stage 슬라이더: 커리큘럼 환경 단계 변경
- 재생, 1 step, 재설정: 에피소드 진행 제어
- stochastic: argmax 대신 정책 분포에서 행동 샘플링

창 없이 최종 프레임을 PNG로 저장하려면 다음을 사용한다.

    MPLBACKEND=Agg conda run -n bari2d python infer.py \
      --checkpoint runs/baseline/checkpoint_001000.pt \
      --no-show --steps 100 --output inference.png

체크포인트에 저장된 experiment_config를 우선 사용한다. 이 방식은 정책의 입력 차원과 아키텍처를 훈련 당시와 일치시킨다. 오래된 체크포인트에 설정이 없을 때만 --config의 설정을 사용한다.
