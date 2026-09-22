# 수동 로봇 조종

scripts/manual.py는 BARI2D 환경을 한 step씩 진행하면서 사람이 선택한 로봇 한 대에만 action을 보내는 Matplotlib 도구다. 나머지 로봇은 해당 step에 모두 IDLE을 받는다.

    conda run -n bari2d python scripts/manual.py

시작 시 주황색 원으로 선택 로봇을 표시한다. 로봇 본체색은 layer 0부터 10까지 달라지고, 보라색 아래쪽 삼각형은 하향 IR을 뜻한다. 로봇을 클릭하거나 좌우 화살표, [와 ] 키로 선택을 바꾼다. 화면 오른쪽에는 선택 로봇의 layer, anchor, strain, 낙하 상태, 직전 보상과 현재 유효 action이 표시된다.

| 입력 | action |
| --- | --- |
| W / S | 전진 / 후진 |
| A / D | 전진 좌회전 / 전진 우회전 |
| Z / C | 후진 좌회전 / 후진 우회전 |
| E | CLIMB |
| Q | ANCHOR |
| R | RELEASE |
| Space | IDLE |
| Home | 같은 seed로 에피소드 다시 시작 |
| Escape | 창 닫기 |

Matplotlib 기본 단축키를 수동 조종 창에서 끄므로 S는 figure 저장이 아니라 후진, Q는 종료가 아니라 ANCHOR로 동작한다. 화면 하단의 저장 버튼은 현재 프레임을 manual.png에 저장하며, --output 경로를 지정했다면 그 파일에 저장한다. 종료 버튼은 창을 닫는다.

로봇은 layer 0부터 layer 10까지 올라갈 수 있다. 별도 DESCEND action은 없지만, 앵커되지 않은 로봇이 바로 아래 layer의 지지 로봇 반경 1.125 밖으로 벗어나면 자동으로 한 layer씩 내려온다. 따라서 CLIMB 뒤 전진해 지지 로봇의 범위를 벗어나면 다시 낮아진다. layer는 2.5D 이산 높이 표현이며, 화면의 로봇 라벨 L0–L10으로 확인한다.

초기 화면을 PNG로 저장하거나 headless 환경에서 확인하려면 다음처럼 실행한다.

    MPLBACKEND=Agg conda run -n bari2d python scripts/manual.py \
      --stage 2 --target-load 6 --output manual.png --no-show

--config, --seed, --stage (1–4), --target-load 옵션으로 환경을 지정할 수 있다. action mask가 허용하지 않는 입력은 환경에서 IDLE로 바뀌며, 오른쪽 현재 허용 목록에서 확인할 수 있다.
