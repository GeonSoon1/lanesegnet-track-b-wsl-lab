# rjstns23 Windows + WSL 실행 기록 (2026-10-28)

이 기록은 기존 작성자의 결과와 별개인 새 실행이다. 과거 camera/fusion 결과를 복사하지 않았다.

- 코드: /home/rjstns23/gg49-lanesegnet-lab_sj
- 새 환경: 위 폴더의 .venv-mms, Python 3.10.19. 기존 환경은 변경하지 않는다.
- 환경 복구: 기존 환경을 계속 사용하면 된다. 새 환경만 별도로 설치했다. 원본 코드 변경은 git diff로 확인한다.
- D: KIOXIA 약 2TB, WSL drvfs /mnt/d. 초기 여유 약 1.4TB.
- 결과: D:\mms_experiments\gg49-lanesegnet-lab_sj = /mnt/d/mms_experiments/gg49-lanesegnet-lab_sj
- 데이터: APR01 Track_B01~B06, Sphere EO 기준 pose, Track_B LAS 5개. 원본 읽기 전용 사용.
- SHP: 원본 ZIP에서 통합SHP만 data/shp/통합SHP로 추출.
- 전달된 repo 루트 가중치 SHA256: 9a31e3b0890ff6d9b6128fb6b470a2cb5850789103e2c309e46d6c443b460cda. SSD 복사본 일치.
- 최종 실험 접두사: sj-20261028-v2-. 초기 sj-20261028- 진단 실행은 별도로 보존한다. demo1epoch, pilot, camera/fusion 각각24epoch.
- TMPDIR, PIP_CACHE_DIR, TORCH_HOME, HF_HOME, XDG_CACHE_HOME, MPLCONFIGDIR 모두 D 실험 폴더.
- SSD의 임시 hard-link 연산이 지원되지 않아 uv 설치 방식을 중단하고 공식 python-build-standalone 3.10.19와 pip로 별도 환경을 만들었다. 임시 다운로드도 D를 사용.

## 명령
WSL에서: python3 scripts/local_control.py dashboard / prepare / start / status / stop / resume
로그: tail -f /mnt/d/mms_experiments/gg49-lanesegnet-lab_sj/logs/supervisor.log
시작/재개는 입력 QA_APPROVED.json 검사와 flock 잠금을 통과해야 한다.
중지는 STOP 파일로 요청하며, worker가 현재 optimizer update를 완료한 다음 상태를 저장한다. 즉시 강제 kill하지 않는다.
브라우저나 일반 터미널을 닫아도 setsid 분리 프로세스는 유지된다. 컴퓨터 종료/절전/WSL shutdown에서는 중단될 수 있다.
재부팅 뒤 SSD 마운트와 환경을 확인한 다음 resume한다. 마지막 유효 checkpoint 이후 저장하지 못한 step은 다시 수행한다.
checkpoint는 last에 모델/optimizer/scaler/step 기반 scheduler 정보/epoch/step/난수/order/position/settings/data hash를 저장한다. 25 optimizer step 간격과 검증 직전에 원자적 교체하고 재읽기를 검사한다.

## 이번 설계 보완
하드코딩 Python 경로를 설정에서 읽는다. 별도 시연단계, process 기반 상태 판정, 안전 중지, 체크포인트 readback, 로컬 전용 서버와 SHP ZIP 다운로드를 추가한다.
원본 엔진의 opt-in align_inference_reference=True를 두 모델에 동일 적용한다. 이전 문서의 성능 수치를 새 결과로 재사용하지 않는다. 이 보정만으로 성능 향상을 주장하지 않는다.
QGIS export는 새 best의 실제 test 평가 추론을 사용한다. raw pred_*와 GT 없는 merged_*를 분리하며 GT는 비교에만 사용한다. 건수/CRS/XYZ/속성과 ZIP 검사를 통과해야 완료다.

## 현재 상태
영상/결합 모델 각24epoch, best checkpoint test 평가와 SHP 재읽기 검증을 완료했다. 최종 전체 검증의 근거는 reports/LOCAL_COMPLETION_AUDIT.json이다. 아래 과정 기록과 최종 결과를 구분해 확인한다.

## 이번 실행에서 확인한 진행

CUDA forward/backward 성공, pip check 통과, 기존 계약 테스트 11개 통과. Track B 초기 split은 train191/val47/test57/buffer46/unlabelled12였고, 아래 v2에서 보완했다.
1 epoch 시연 완료: 48 optimizer steps, validation loss52.973691. Step13에서 안전 중지 후 상태 키를 재읽고 14 이후 진행을 검증했다. 이 수치는 최종 성능이 아니다.
메모리 probe에서 384px/micro batch2/유효batch4를 선택했고 fusion 실제 gradient와 가중치 변화가 확인됐다.
실제 프로세스는 새 세션, 제어 TTY 없음, stdout은 SSD 로그이다. 실행 명령을 시작한 셸이 끝난 뒤에도 진행됨을 확인했다. 컴퓨터 재부팅 시험은 수행하지 않았다.


## 2026-10-28 분할 보완 v2

초기 train191에는 양성 연결 정답이 0개였다. 기존 새 실행은 step146에서 안전 중지하고 checkpoint와 지표를 보존했다. 과거 저자 결과가 아니다.
입력 및 원래 test57을 그대로 유지하며 공간 X 구역을 다시 정했다. v2는 train228/val14/test57/buffer42/unlabelled12, 연결 정답 train24/val22/test48이다. 최소 구간 간 X 거리153.9m로 BEV ROI 겹침을 방지한다. 학습에는 새 공식 초기값을 사용한다.
접두사 sj-20261028-v2-. 원래 manifest는 artifacts/prepared/DATASET.initial-split-v1.json에 해시 그대로 보존한다. reports/split-policy-v2.json에 경계·근거가 있다. 검증14시점은 작고 인접해 있어 best 선택의 변동성과 일반화 한계가 있다. Test 모델 점수로 분할을 조정하지 않았다.


Windows SSD 루트의 OPEN_DASHBOARD_SJ.cmd는 대시보드만 열고, START_OR_RESUME_TRAINING_SJ.cmd는 학습 시작/재개, CHECK_TRAINING_SJ.cmd는 프로세스 상태, STOP_TRAINING_SJ.cmd는 안전 중지를 요청한다.
검증 명령: source scripts/local_env.sh 후 .venv-mms/bin/python -m pytest -q tests --capture=sys -o cache_dir=/mnt/d/mms_experiments/gg49-lanesegnet-lab_sj/cache/pytest. 11개 통과. 기본 fd 캡처는 이 SSD drvfs의 임시 파일 truncate에서 실패해 메모리 sys 캡처로 실행했다. 엔진 전체 수집이 아닌 lab 계약 테스트다.

좌표 및 시간 근거: 6개 EO 파일은 각353시점이며 같은 행의 촬영 시각 차이0초. LAS 헤더에는 개별 CRS 태그가 없지만 제공된 원본 CoordinateSystem.prj는 UTM52N/WGS84/EPSG32652이며 SHP와 일치한다. 좌표값과 영상 투영도 확인했다. 높이는 원본 Z를 보존하며 수직 기준면은 별도 측량 인증하지 않았다. 세부 보고서 reports/sensor-metadata-audit.json.


## 동일 PC의 새 환경을 다시 만드는 명령

원본 환경을 변경하는 작업이 아니다. 이미 실행 중인 .venv-mms를 덮어 만들지 않는다. 환경 복구가 필요할 때 학습을 안전 중지하고, 기존 환경 폴더를 백업한 뒤 동일한 .venv-mms 경로에 같은 버전을 복구한다. checkpoint는 환경 경로까지 검사하므로 다른 경로·설정을 사용하면 재개를 거부한다.

```bash
cd /home/rjstns23/gg49-lanesegnet-lab_sj
source scripts/local_env.sh
/home/rjstns23/.local/share/gg49-python310/python/bin/python3 -m venv .venv-mms
.venv-mms/bin/python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
bash scripts/install_local_dependencies.sh
python3 scripts/local_control.py status
python3 scripts/local_control.py resume
```

Python 3.10.19 런타임은 Python Build Standalone 20251014 배포본이며, 다운로드한 아카이브는 D의 cache/tmp/python310.tar.gz에 보존했다. SHA256: c0b09b744293f2aad85b1ef84544f8a7ba383675b29d1f7efd1e96bb9984399f
원래 패키지 버전은 reports/environment-local.lock.txt에 있다. SSD 드라이브 매핑과 CUDA backward 시험을 다시 통과시킨 후 재개한다.

대시보드만 중지: python3 scripts/local_control.py dashboard-stop 또는 SSD의 STOP_DASHBOARD_SJ.cmd. 학습에는 영향을 주지 않는다. 다시 dashboard/OPEN_DASHBOARD_SJ.cmd로 열 수 있다.


## SSD 상태 파일 교체 오류와 보완

v2 영상 모델 step157에서 status JSON 임시 파일 os.replace가 ENOENT로 실패했다. 정확한 OS 원인은 단정하지 않는다. 감독 프로세스가 마지막 checkpoint(step150)에서 자동 재개했고 저장되지 않은151~157은 다시 계산했다. 원본 metrics/log는 보존한다.
임시 파일 이름을 호출별 UUID로 바꾸고 fsync 및 제한된 재시도를 추가했다. 해당 오류와 동일 프로세스 동시 저장 회귀 테스트를 포함해13개 테스트를 통과했다. checkpoint도 새 임시 이름으로 원자 교체한다. Dashboard/그래프는 반복 step의 마지막 시도 값을 표시하며 원본 JSONL을 유지한다.
실제 본 학습을 안전 중지해 새 코드로 재개했다. 저장 step 397, epoch index 6, 데이터 위치 220. reports/main-v2-resume-saved.json 및 run의 resume-events.jsonl을 확인한다.

실행 중 프로세스의 TMPDIR/PIP/Torch/HF/XDG/Matplotlib 경로가 SSD 안임을 /proc로 재확인했다. 기존 home NVIDIA 캐시83,968,599bytes는 모두10월22일 자료로 보존했다. 이후 새 프로세스의 CUDA_CACHE_PATH도 SSD cache/cuda로 지정한다. /tmp는148KiB로 대용량 산출물을 경유하지 않았다. reports/storage-placement-audit.json.

대시보드를 실제 브라우저로 점검했다. 원본+EMA 곡선, 실험 선택, 1epoch완료와 SHP다운로드 링크를 확인했다. 완료된 실험의 경과시간은 completed_at에서 멈추며 재개 후 ETA는 원본 timestamp 기준이다. 상태는 실제 프로세스를 확인해 한국어로 표시한다.


## 재부팅 뒤 SSD 마운트 확인

Windows PowerShell에서 Get-Volume -DriveLetter D로 KIOXIA를 확인한다. WSL에서는 아래처럼 마운트를 먼저 확인한다. 자동 마운트가 되어 있으면 mount 명령은 실행하지 않는다.

```bash
mountpoint /mnt/d
findmnt -T /mnt/d
df -h /mnt/d
# D가 연결되어 있으나 WSL에 마운트되지 않았을 때만:
sudo mkdir -p /mnt/d
sudo mount -t drvfs D: /mnt/d -o metadata
mountpoint /mnt/d
cd /home/rjstns23/gg49-lanesegnet-lab_sj
source scripts/local_env.sh
.venv-mms/bin/python -m pip check
python3 scripts/local_control.py status
python3 scripts/local_control.py resume
```

마운트·쓰기·20GiB 여유·입력 QA 해시 검사가 실패하면 실행하지 않는다. 모델/optimizer/AMP/난수/데이터 위치는 last.pth에서 읽는다. 읽기 또는 설정 호환성 검사가 실패하면 중단하며 best만으로 optimizer를 복구했다고 처리하지 않는다.
저장 로그의 시각은 UTC이며 대시보드 표시는 한국 시각이다. 완료 시간은 completed_at에 고정한다.


v2 시연1epoch: 57updates, val loss56.292421. CPU로 실제14val시점을 추론해 confidence0.5 이상5개 차로 예측, 4개 XYZ SHP레이어와 ZIP의 HTTP 다운로드 해시까지 검증했다. 최종24epoch 결과와 별개다.
영상 모델24epoch/1368updates 완료. Best epoch7, val loss24.03852810178484. last의 optimizer/scheduler/RNG/순회 위치를 재읽어 확인했다. 이 시점에는 fusion 학습 중이며 test 평가 전이다.


## 추론 입력 재사용 보완

두24epoch 학습 후 fusion 검증 입력 ablation에서 원래 엔진의 singleton batch squeeze_가 호출자 이미지 차원을 바꿔 IndexError가 발생했다. predict가 입력 텐서와 메타데이터 사본을 사용하도록 보완했다. 학습된 best를 그대로 사용해 평가를 재실행하며, 첫 camera 평가 산출물은 attempt1 파일로 보존했다. 학습을 다시 시작한 것이 아니다.


# 이 PC의 새 Track B 학습 결과

실제 실행 완료 시점: 2026-10-28T14:32:35.716675+00:00

기존 작성자의 결과를 재사용하지 않은 sj-20261028-v2- 실험이다. 로컬 F1이며 공식 OpenLane-V2 점수가 아니다.

## sj-20261028-v2-camera-24e
- 학습: 24 epoch / 1368 optimizer updates
- Best: epoch 7, validation loss 24.03852810178484
- Test F1@1m: 0.2971014492753623
- Test F1@0.5m: 0.057971014492753624
- 원시 예측 76개 / 병합 12개. 병합 정확도 향상을 뜻하지 않는다.
- SHP: runs/sj-20261028-v2-camera-24e/qgis-export-v1/
- ZIP: runs/sj-20261028-v2-camera-24e/qgis-export-v1.zip

## sj-20261028-v2-fusion-24e
- 학습: 24 epoch / 1368 optimizer updates
- Best: epoch 8, validation loss 22.56091547012329
- Test F1@1m: 0.2846715328467153
- Test F1@0.5m: 0.06569343065693431
- 원시 예측 74개 / 병합 8개. 병합 정확도 향상을 뜻하지 않는다.
- SHP: runs/sj-20261028-v2-fusion-24e/qgis-export-v1/
- ZIP: runs/sj-20261028-v2-fusion-24e/qgis-export-v1.zip

두 모델 차이를 실제 test 지표로 비교해야 하며, loss 하락이나 학습 완료만으로 LiDAR 개선을 주장하지 않는다.
작은 Track B 공간 분할과 파생 GT라는 한계가 있다. 병합은 GT 없는 후처리이며 전체 도로망 연결 보증이 아니다.


## 실제 결과 해석

영상/결합 각각24epoch와 test57시점 평가를 완료했다. F1@1m은 영상0.297101, 결합0.284672이며 F1@0.5m은 영상0.057971, 결합0.065693이다. 이번 한 분할에서 LiDAR의 일관된 우위는 확인하지 못했다.

결합 모델은 실제 입력에 반응했다: validation B_0081의 관측455셀, 학습된 projection L2=0.107545, 반복 입력 최대 좌표차0m, LiDAR를0으로 바꾸면 전체query 최대 좌표차13.311667m. 이는 정확도 개선 증거와 별개다.

sj-20261028-v2-camera-24e: 예측→GT 거리 원시 0.778984m → 병합 2.418345m. 선 감소를 성능 개선으로 보고하지 않는다.

sj-20261028-v2-fusion-24e: 예측→GT 거리 원시 0.681617m → 병합 2.252726m. 선 감소를 성능 개선으로 보고하지 않는다.

추론 입력 재사용 회귀 테스트 포함14개 테스트 통과. 원본 ZIP과 최초 완료 audit는 initial-verified/initial 이름으로 보존했다.
