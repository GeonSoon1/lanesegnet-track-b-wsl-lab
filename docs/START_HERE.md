# 이 PC의 새 실행 안내 (2026-10-28)

현재 사용자 rjstns23의 경로와 새 실험은 [LOCAL_WINDOWS_WSL_gs.md](LOCAL_WINDOWS_WSL_gs.md)를 참고하세요.
아래 원작성자의 2026-10-24 결과는 과거 기록이며 이 PC의 새 측정 결과가 아닙니다.

```bash
cd /home/rjstns23/gg49-lanesegnet-lab_gs
python3 scripts/local_control.py status
python3 scripts/local_control.py dashboard
python3 scripts/local_control.py stop
python3 scripts/local_control.py resume
```

Windows에서 D:\mms_experiments\gg49-lanesegnet-lab_gs\OPEN_DASHBOARD_gs.cmd를 열면 대시보드만 실행합니다.
새 best/last는 runs/gs-20261028-{camera,fusion}-24e/에, 검증된 SHP ZIP은 각 run/qgis-export-v1.zip에 생성됩니다. 아직 파일이 없으면 해당 단계가 미완료입니다.

---

# 학습 현황 확인과 재실행

## 현황 보기

이 PC에서 http://127.0.0.1:8765 를 열면 현재 단계, 실험, epoch/step, 학습·검증 및 항목별 loss, GPU 메모리, 예측과 최근 오류 로그를 볼 수 있습니다. 브라우저를 닫아도 학습은 계속됩니다.

큰 결과는 모두 `D:/mms_experiments/gg49-lanesegnet-lab_gs`에 있습니다. 코드와 mms 환경 자체는 기존 WSL 위치에 유지합니다. 원본 D 데이터는 읽기만 합니다.

- `runs/pipeline-status.json`: 전체 진행.
- `runs/camera-24e`, `runs/fusion-24e`: 실제 지표, loss.png, history.csv, TensorBoard, best/last.pth, prediction.png.
- `reports/input-qa`, `reports/lidar-qa`: 입력 투영 진단.
- `reports/memory-selected.json`: 실제 검사를 통과한 메모리 조건.
- `reports/TRAINING_RESULTS.md`, `experiment-comparison.json`, `comparison.png`: 완료 후 생성하는 결과.
- `environment_backup/mms-before-training-20261024.tar`: 교체 전 mms 복구 사본. 실행 중인 환경에 덮어 풀지 마세요.

## 학습 순서

전체 GT와 입력 검사 → 두 모델의 16개 표본 100 optimizer-step 검증 → 영상 비교 24 epoch → RGB+LiDAR 24 epoch → 같은 validation/test에서 평가 순서로 진행합니다. 결합 모델의 개선 여부는 실제 비교 결과로 판단합니다.

공통 설정은 384×384, micro batch 2, gradient 누적 2, 유효 batch 4입니다. 영상 CNN은 AMP, BEV/head/loss는 FP32입니다. 새 실험은 공식 초기 가중치로 시작하고, 같은 실험을 재개할 때는 자체 last.pth를 사용합니다.

## 중복 실행 없이 재개

먼저 대시보드와 `runs/supervisor.pid`, `runs/active-worker.json`의 실제 프로세스를 확인하세요. 정상 실행 중이면 다시 기동하지 않습니다. 멈춘 경우 WSL에서 다음을 실행합니다.

```bash
cd /home/rjstns23/personal/mms_project/gg49-lanesegnet-lab_gs
bash scripts/run_training.sh
```

완료된 단계는 재사용하고 checkpoint에서 재개합니다. 설정이나 데이터 hash가 다르면 기존 실험과 섞지 않습니다. D가 마운트되지 않았으면 실행을 중단하며 C에 대체 저장하지 않습니다. 복구 이력은 `reports/repairs.jsonl`에 있습니다.

대시보드만 멈췄다면 다음을 실행합니다.

```bash
python3 /home/rjstns23/personal/mms_project/gg49-lanesegnet-lab_gs/scripts/dashboard_server.py
```

실측값이 생기기 전에는 그래프가 비어 있습니다. 이 실험은 로컬 GG49 공간 분할의 측정이며 공식 OpenLane-V2 성능 점수가 아닙니다. 설계와 완료 기준은 [PRD](/home/rjstns23/personal/mms_project/gg49-lanesegnet-lab_gs/docs/PRD.md), 실제 시도와 결과 해석은 [실험 회고](/home/rjstns23/personal/mms_project/gg49-lanesegnet-lab_gs/docs/mms_lanesegnet_training_experiment.md)를 참고하세요.

## 최종 상태 (2026-10-24)

영상 비교와 RGB+LiDAR 각24 epoch, best/last와 held-out 평가까지 완료했습니다. 두 best는 epoch6이며 test F1@1m은 모두0.284672로, 이번 실험에서 LiDAR 성능 향상은 입증되지 않았습니다. D의 `reports/TRAINING_RESULTS.md`에 상세 결과와 한계가 있습니다. 완료 점검 자동화 gg49는 삭제했고 절전 억제는 해제했습니다. 대시보드는 계속 열 수 있습니다. 지도 정합 P6는 후속 작업입니다.

## 컴퓨터를 껐다 켠 뒤 다시 보기

결과 보고서·그래프·best/last 가중치는 D드라이브 파일로 보존되므로 종료해도 사라지지 않습니다. 코드와 mms 환경은 WSL에 남습니다. 학습과 평가가 완료되어 컴퓨터를 종료해도 됩니다.

D드라이브를 연결하고 `D:\mms_experiments\gg49-lanesegnet-lab_gs\OPEN_DASHBOARD.cmd`를 더블클릭하세요. 필요하면 WSL에서 D를 마운트하고 대시보드 서버를 다시 시작한 뒤 브라우저를 엽니다. 이미 대시보드가 실행 중이면 재사용합니다. 이 실행 파일은 대시보드만 열고 학습은 시작하지 않습니다.

브라우저 주소만으로 접속되지 않으면 위 실행 파일을 먼저 여세요. 보고서는 대시보드 없이도 `D:\mms_experiments\gg49-lanesegnet-lab_gs\reports\TRAINING_RESULTS.md`, 그래프는 `runs\camera-24e\loss.png`와 `runs\fusion-24e\loss.png`를 직접 열 수 있습니다.

## Track B 실패·LiDAR·검출 기준 진단 보기

D드라이브를 연결하고 `D:\mms_experiments\gg49-lanesegnet-lab_gs\OPEN_DIAGNOSTICS.cmd`를 더블클릭하면 서버와 진단 화면을 함께 엽니다. 기존 `OPEN_DASHBOARD.cmd`로 서버를 켠 뒤 http://127.0.0.1:8765/diagnostics 를 열어도 됩니다. 첫 화면 상단의 **Track B 진단실 열기** 링크로도 이동할 수 있습니다.

- **실패 사례:** 시점을 선택하면 BEV 정답/두 모델 비교와 카메라 영상을 볼 수 있습니다. 카메라01~06을 바꾸고 정답·영상 예측·결합 예측·좌우 경계·LiDAR 셀을 개별 표시할 수 있습니다. BEV 그림과 6대 전체 이미지는 크게 열 수 있습니다.
- **LiDAR 비교:** 같은 fusion best에서 정상·0·다른 validation 시점 입력을 비교합니다. 표는 전체 validation, 비교 이미지는 선택한 시점의 결과입니다. 이는 입력 의존성 진단입니다.
- **검출 기준:** confidence 슬라이더와 허용거리 선택으로 precision/recall/F1을 확인합니다. 위 사례 이미지는confidence0.5 고정이며 슬라이더는 분석 표·그래프에 적용됩니다.

사용 데이터는 Track B validation47개이고 이번 진단에 test 입력은 사용하지 않습니다. 원본·학습 가중치를 보존하고 D의 `reports/diagnostics-track-b-v1`에 PNG·시점별JSON·threshold CSV·요약과 보고서를 저장합니다. 진행/오류는 `runs/diagnostics-status.json`, `logs/diagnostics.log`에서 확인합니다. 진단 결과와 해석은 실험 회고에 누적합니다.

필요할 때 진단만 실행하는 명령은 아래와 같습니다. 이미 실행 중인 진단 worker가 있으면 중복 실행하지 않습니다.

```bash
cd /home/rjstns23/personal/mms_project/gg49-lanesegnet-lab_gs
bash scripts/run_diagnostics.sh
```

## 실험 4·5와 팀 공유 화면

http://127.0.0.1:8765/experiments 를 열면 전체 흐름의 쉬운 설명, reference 전후 47개 사례와 두 20시점 구간의 8건 정합 비교를 볼 수 있습니다. D의 `OPEN_EXPERIMENTS.cmd`로 서버와 이 화면을 함께 열 수 있습니다.

QGIS에서 GeoPackage를 열면 확정 중심선·좌우 경계·원관측·평가 정답을 분리해 볼 수 있고, 지지수 조건은 불확실 레이어도 포함합니다. 결과는 D의 `reports/followup-45-v1`, 설명은 `docs/TEAM_BRIEF.md`와 실험 회고에 보존합니다. 이 단계는 재학습이 아닌 고정 가중치 추론·후처리 실험입니다.

## 녹화용 1 epoch 시연 (2026-10-26)

WSL에서 `bash scripts/run_demo.sh --check-only`로 D 마운트·353개 데이터 목록·초기 가중치·첫 캐시 파일을 확인한 뒤 `bash scripts/run_demo.sh`를 실행한다. mms Python을 직접 사용하며 고유한 demo-camera 이름으로 1 epoch를 새로 학습한다. 기존 camera-24e/fusion-24e를 덮어쓰지 않는다. 동시에 다른 학습을 실행하지 않는다. 전체 캐시 검사나 새로운 성능 검증을 대신하는 시연은 아니다.

Windows에 D가 보여도 WSL에는 마운트되지 않을 수 있다. 이때 Windows PowerShell에서 `wsl.exe -d Ubuntu -u root -- mount -t drvfs D: /mnt/d -o metadata,uid=1000,gid=1000`로 연결한 뒤 확인한다. 스크립트는 미연결 시 종료하므로 뒤 학습을 실행하지 않는다.
