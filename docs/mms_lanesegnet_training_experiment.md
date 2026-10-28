# MMS LaneSegNet Training Experiment Log

## 목적

이 문서는 GG49 LaneSegNet의 데이터 준비, 영상 비교 모델과 RGB+LiDAR 결합 모델 학습, 오류 복구, 평가 결과와 미달성 목표를 누적 기록한다. [PRD](/home/imella0707/personal/mms_project/gg49-lanesegnet-lab_sj/docs/PRD.md)는 프로젝트 설계와 요구사항을, 이 문서는 실제 실행과 회고를 담당한다. 대시보드와 재실행 방법은 [START_HERE](/home/imella0707/personal/mms_project/gg49-lanesegnet-lab_sj/docs/START_HERE.md)에 둔다.

기록 형식은 [MMS 라벨링 실험 노트](/home/imella0707/personal/mms_project/labeling-studio/notes/mms_data_labeling_experiment.md)를 참고했다. 그 문서의 다른 프로젝트 수치나 성과 주장을 이번 실험의 근거로 사용하지 않는다.

## 기록 원칙

- 실험별로 목적 → 조건과 실행 → 관찰 수치 → 해석 → 한계 → 다음 실험을 기록한다.
- 오류와 복구를 해당 실험 안에 함께 남기고, 결과를 좋게 보이도록 실패를 지우지 않는다.
- 실제 로그·checkpoint·평가 파일과 가설을 구분한다. 학습 완료를 성능 목표 달성으로 표현하지 않는다.
- 시각은 KST로 표기한다. 원본 JSON/JSONL의 UTC 시각은 그대로 보존한다. 과거 실행은 저장된 근거를 바탕으로 사후 정리했음을 명시한다.
- 후속 실험은 새 회고와 run 이름으로 추가한다. 기존 test를 반복적으로 보며 설정을 맞추지 않는다.

## 현재 결론

**첫 비교 실험의 학습·평가는 완료했지만, LiDAR 추가를 통한 test 성능 개선은 확인하지 못했다.** Test F1@1m은 두 모델이 같고, F1@0.5m은 RGB+LiDAR가 낮았다. 이는 현재 결합 방식과 한 트랙·한 seed의 결과이며, LiDAR 자체가 도움이 없다는 결론은 아니다. 여러 시점의 예측 차선을 정합하는 프로젝트 최종 목표(P6)는 아직 수행하지 않았다.

---

## 회고 1: 2026-10-24 05:12 KST - Track B 영상/RGB+LiDAR 첫 비교 학습

### 문제 상황과 실험 목적

공식 LaneSegNet R50의 초기 가중치와 GG49 MMS 데이터·SHP 정답이 있었지만, 바로 학습 가능한 데이터 연결과 RGB+LiDAR 입력 경로는 완성되어 있지 않았다. 사용자는 `mms` 환경, 원본 보존, C드라이브 여유 부족에 따른 D드라이브 사용, 실제 학습과 현황 대시보드 구성을 요청했다.

실험 목적은 두 가지였다. 첫째, GG49 데이터로 실제 학습·평가가 가능한 파이프라인을 만든다. 둘째, 동일 조건의 영상 전용 비교 모델보다 RGB+LiDAR 결합 모델의 성능을 높인다. **첫째는 완료했지만 둘째는 이번 실험에서 달성하지 못했다.**

이 항목은 초기 조사부터 최종 평가까지를 사후 정리한 기록이며, 제목 시각은 평가 완료 시각이다.

### 실행 환경과 저장 위치

| 항목 | 실제 조건 |
|---|---|
| 코드·문서 | `/home/imella0707/personal/mms_project/gg49-lanesegnet-lab_sj` |
| 원본 센서 입력 | `D:/2026-APR-01_GG803SEC012` (읽기 전용 사용) |
| 원본 정답 | `D:/SHP데이터/통합SHP` (읽기 전용 사용) |
| 대용량 결과 | `D:/mms_experiments/gg49-lanesegnet-lab_sj` / WSL `/mnt/d/mms_experiments/gg49-lanesegnet-lab_sj` |
| 가상환경 | WSL2 Ubuntu, `/home/imella0707/miniconda3/envs/mms` |
| 학습 스택 | Python 3.10.21, PyTorch 1.13.1+cu117, torchvision 0.14.1+cu117, MMCV-full 1.7.0, MMDetection 2.26.0, NumPy 1.26.4 |
| GPU 예산 | 전체 사용량 10.5 GiB (10,752 MiB) 이내 |
| 환경 근거 | `configs/requirements-mms.lock.txt`, D의 `reports/memory-selected.json` |
| 코드 기준 | 독립 `.git` 메타데이터가 없는 실험 폴더. 이번 작업에서 별도 branch/commit을 생성하지 않음 |

공식 모델 기준 commit은 `9f12fe72a95dd991b3ec0a01cfbae5f3e9e961e0`, 멘토 GT builder 기준 commit은 `6b3671c29a66aebd6e6246dc31a5946dbcb0958f`이다. 두 원본 저장소는 수정하지 않고 실험 폴더에서 연결했다. 교체 전 `mms`는 D의 `environment_backup/mms-before-training-20261024.tar`에 보존했다. 사용하지 않는 구 Python 3.12 패키지는 복구 사본·활성 인터프리터·실행 프로세스의 미사용 상태를 확인한 뒤 정리했다. WSL 가상 디스크 압축은 수행하지 않았다.

데이터·GT·캐시·checkpoint·로그·그래프·환경 복구 사본은 D에 저장했다. 코드와 환경 자체는 WSL에 남겼다. 초기 복사 검증용 Track B 약 3.9 GB도 D에 보존했으며, 준비 파이프라인은 원본 D 입력을 읽었다. D 미연결 시 C에 대체 저장하지 않도록 검사했다.

### 데이터 선택과 입력·정답 구성

Track B는 특별히 모델 성능이 잘 나오는 트랙이라 선택한 것이 아니다. 이미 압축 해제된 데이터에서 6대 카메라·EO가 연결되고 SHP와 공간적으로 겹쳐 첫 실험을 구성하기에 적합했다. 다른 트랙에서도 입력과 정답 조건이 충족되면 실험할 수 있다.

| 항목 | 실제 준비 내용 |
|---|---|
| 원본 시점 | Track B 353개 |
| 영상 | 카메라 01~06, 보정 후 384×384 |
| 분할 | train 191 / validation 47 / test 57 / buffer 46 / unlabelled 12 |
| 분할 방식 | 지도 X 위치를 기준으로 연속 구간을 나누고 ±75 m buffer 적용, BEV 영역의 공간 비겹침 검사 |
| 비교의 표본 조건 | 두 모델 모두 동기 LiDAR가 있는 동일 eligible 시점·정답·분할 사용 |
| GT | 최종 semantic/topology NPZ의 중심선·좌우 경계, 유형, mask, 방향성 adjacency를 같은 인스턴스 순서로 사용 |
| 유형 처리 | unknown=-1은 유형 loss에서 제외, none=0은 학습. 혼합·기타 코드 223/224/999는 unknown 유지 |
| 좌표 계약 | `ego=(world-origin)@rotation`, `world=ego@rotation.T+origin` |
| BEV | X±51.2 m, Y±25.6 m, Z[-4,2] m, 100×200 grid |
| LiDAR | 촬영 시각 ±0.1초의 LAS 점군을 ego BEV로 변환. 누적 지도 대체 입력을 쓰지 않음 |
| LiDAR 5채널 | count, mean intensity, minimum Z, maximum Z, observed mask (채널별 정규화는 prepare 코드 기준) |

입력·정답 시각 투영 12개와 LiDAR 투영 표본을 보존해 큰 축 반전 등을 확인했다. 이는 정밀 측량 수준의 보정이나 시각 동기 정확도를 인증한 검사가 아니다. 시간 창 내 LiDAR는 희소한 띠 형태로 관측되므로, 화면 전체에 조밀한 점군이 입력됐다고 표현하지 않는다.

Prepared dataset SHA256:

```text
8339f7cdf7194a883e6a52ce89d3ee9119b5f5e0ba13e836c98a30aebd09915f
```

### 모델 연결과 공통 학습 조건

공식 R50 모델은 영상 기반이며 `use_lidar=False`이다. 이 비교에서는 영상 모델을 camera baseline으로 두고, 별도의 `LiDARFusion` CNN(5→64→64→256) 출력을 영상 BEV에 잔차로 더하는 fusion 모델을 구현했다. 마지막 projection은 0으로 초기화하고 observed mask로 잔차를 제한했다. 단순히 설정의 `False`를 `True`로 바꾼 실험이 아니다.

로컬 멘토 자료에는 완전한 `fusion_train.py`와 그 학습 환경이 없었다. **이번 fusion 구현은 멘토의 RGB+LiDAR 학습을 그대로 재현한 모델이 아니다.** 이미지 backbone은 ResNet-50이며 전체 모델은 LaneSegNet의 BEV·head·topology를 함께 사용한다.

| 항목 | camera-24e / fusion-24e 공통 |
|---|---|
| 초기 가중치 | `lanesegnet_r50_8x1_24e_olv2_subset_A.pth`의 호환 가중치 |
| 불일치 처리 | 7→6 camera embedding 및 class 수가 다른 head key 등은 제외·새 초기화하고 로드 근거 보존 |
| 시작 방식 | 각 모델을 공식 가중치에서 별도로 시작. pilot 또는 학습된 camera 가중치를 fusion에 이어 쓰지 않음 |
| 학습 길이 | 실제 24 epoch, 각 1,152 optimizer step |
| batch | micro batch 2 × 누적 2 = 유효 batch 4 |
| optimizer | AdamW, 전체 LR 2e-5 / 이미지 backbone LR 2e-6, weight decay 0.01 |
| 스케줄·안정화 | 100-step warmup 후 cosine LR, gradient clip 5 |
| seed | 42 (한 seed 실험) |
| 정밀도 | 이미지 CNN은 AMP, BEV/head/기하 assignment/loss는 FP32 |
| best 선택 | 고정 validation total loss 최저 checkpoint; test로 선택하지 않음 |
| resume | model·optimizer·scaler·난수·표본 순서/진행 위치와 동일 설정·data hash 검사 |

초기 가중치 SHA256은 공식 배포 파일과 일치했다.

```text
9a31e3b0890ff6d9b6128fb6b470a2cb5850789103e2c309e46d6c443b460cda
```

### 실행 순서와 실제 명령

경로·설정 정리 → 복구 가능한 mms 환경 구성 → 대시보드 → GT/영상/LiDAR 준비 → 투영 검수 → 실제 GPU probe → 16개 표본 100-step pilot 두 건 → camera 24 epoch → fusion 24 epoch → 동일 validation/test 평가 순으로 실행했다. probe에서 공통 384×384·micro batch 2 설정을 선택했고, 검사와 본 학습 점검에서 전체 GPU 약 8.3~8.6 GiB가 관측됐다. 두 pilot에서 실제 loss 감소와 가중치 갱신을 확인했으나, 이를 일반화 성능으로 해석하지 않았다.

전체 파이프라인 진입점은 다음과 같다. 실행 전 정상 worker가 있는지 확인해야 하며, 완료된 실험을 새 실험으로 덮어쓰기 위한 명령은 아니다.

```bash
cd /home/imella0707/personal/mms_project/gg49-lanesegnet-lab_sj
bash scripts/run_training.sh
```

launcher는 D mount, `LD_LIBRARY_PATH=/usr/lib/wsl/lib`, D의 TMPDIR·캐시 경로를 설정하고 `mms` Python으로 supervisor를 실행했다. supervisor가 순차 기동한 본 학습 worker의 인자는 다음과 같다.

```text
-m gg49_lab.train train --mode camera --resolution 384 --batch 2 --run camera-24e --epochs 24
-m gg49_lab.train train --mode fusion --resolution 384 --batch 2 --run fusion-24e --epochs 24
```

KST 기준 camera는 03:10:59~04:11:49, fusion은 04:11:55~05:11:17에 실행됐고 전체 평가는 05:12:02에 완료됐다. 정상 supervisor/worker를 중복 실행하지 않았으며 최종 완료 후 10분 간격 후속 점검을 삭제했다. 작업 전용 절전 억제도 해제했다.

### 발생한 오류, 원인과 복구

| 관찰한 문제 | 적용한 조치 | 실제 검증·보존 근거 |
|---|---|---|
| WSL에서 구 PyTorch/cuDNN이 `libcuda.so`를 찾지 못함 | launcher에 WSL GPU 드라이버 경로 지정 | 실제 CUDA Conv2D forward/backward 통과 |
| AMP 기하 assignment에 유효하지 않은 값이 들어가 Hungarian matching 실패 | 이미지 CNN만 AMP로 두고 BEV/head/assignment/loss를 FP32로 제한 | fusion 실제 backward와 encoder/projection gradient·가중치 변화 확인 |
| 공식 head의 eval 경로가 validation loss에 필요한 segmentation 출력을 생략 | eval/dropout 상태를 유지한 채 전체 decoder/mask 출력을 만드는 명시적 분기 추가 | validation 47개 전체에서 유한 loss, eval 유지·가중치 불변 확인 |
| 검증 오류가 epoch 실행을 잃게 할 가능성 | 검증 직전 checkpoint도 저장하고 재개 상태 보존 | 실제 checkpoint model·optimizer·난수 상태 복원 검사 통과 |
| 대시보드 브라우저 자동 검수 실행 실패 | HTTP/API·Python/JavaScript 구문 및 저장 PNG 검수 수행 | 전체 브라우저 화면 시각 검수는 미완료로 명시 |

복구 원문은 [repairs.jsonl](D:/mms_experiments/gg49-lanesegnet-lab_sj/reports/repairs.jsonl)에 있다. validation 실패 시도는 `runs/failed-camera-validation-20261024`에 보존했다. 계약 테스트 3개, 두 모델의 실제 GPU probe·pilot, 47-frame validation smoke와 checkpoint 복원 검사를 수행했다. 가짜 loss나 가짜 CUDA 연산으로 실행을 통과시키지 않았다.

### 측정 결과

아래 값은 각 모델의 **best checkpoint(epoch 6)**로 평가한 결과다. `last.pth`에는 epoch 24·step 1,152가 저장되어 있다. CPU에서 두 best/last를 직접 열고 각 24개 epoch 기록, 동일한 공통 설정과 data hash를 검사했다.

| 항목 | 영상 비교 (camera-24e) | RGB+LiDAR (fusion-24e) |
|---|---:|---:|
| 완료 epoch / optimizer step | 24 / 1,152 | 24 / 1,152 |
| Best epoch | 6 | 6 |
| Best validation total loss | 22.991598 | 22.151032 |
| 마지막 epoch train total loss | 5.965745 | 5.901508 |
| 마지막 epoch validation total loss | 25.617467 | 25.381096 |
| Validation F1@1m | 0.110701 | 0.119850 |
| Test F1@1m | 0.284672 | 0.284672 |
| Test F1@0.5m | 0.058394 | 0.043796 |
| Test precision / recall @1m | 0.527027 / 0.195000 | 0.527027 / 0.195000 |
| Test TP / 예측 수 / GT 수 @1m | 39 / 74 / 200 | 39 / 74 / 200 |
| 매칭된 test 중심선 평균 XYZ 오차 | 0.645024 m | 0.671380 m |

표의 예측 수·GT 수는 시점별 차선 인스턴스를 합산한 것이며, 중복을 제거한 전체 지도 차선 개수가 아니다.

평가는 validation 47개·test 57개 시점, confidence≥0.5에서 수행했다. 중심선의 순서가 있는 10개 표본의 평균 3D 거리로 Hungarian 대응을 구하고 0.5 m/1 m 이내의 대응을 TP로 계산했다. 오차 평균은 1 m 이내에 매칭된 표본에만 해당한다. **공식 OpenLane-V2 평가 점수가 아니다.**

유형 정확도 1.0은 1 m 이내에 매칭된 알려진 유형 표본(test 각 71개)에 한정된다. 전체 차선 검출이 100% 정확하다는 뜻이 아니다. topology는 매칭된 차선 쌍만 평가했고 TP/FP/FN이 모두 0이었다. 이 값으로 연결 관계 성능을 입증할 수 없다.

- [영상 loss 그래프](D:/mms_experiments/gg49-lanesegnet-lab_sj/runs/camera-24e/loss.png)
- [RGB+LiDAR loss 그래프](D:/mms_experiments/gg49-lanesegnet-lab_sj/runs/fusion-24e/loss.png)
- [F1 비교 그래프](D:/mms_experiments/gg49-lanesegnet-lab_sj/reports/comparison.png)
- [평가 원문](D:/mms_experiments/gg49-lanesegnet-lab_sj/reports/experiment-comparison.json)
- [결과 보고서](D:/mms_experiments/gg49-lanesegnet-lab_sj/reports/TRAINING_RESULTS.md)
- [최종 checkpoint·epoch 검증](D:/mms_experiments/gg49-lanesegnet-lab_sj/reports/completion-audit.json)

### 결과 해석과 목표 달성 여부

| 목표 | 판정 | 근거 |
|---|---|---|
| GG49 실제 입력·정답으로 학습/평가 가능 | 완료 | 두 모델의 24 epoch, 실제 가중치 갱신, best/last와 평가 산출물 |
| LiDAR 추가로 영상 모델보다 test 성능 개선 | 미달성 | F1@1m 동일, F1@0.5m과 매칭 오차는 fusion이 불리 |
| 시점 간 차선 중복 정합·병합 | 미착수 | 이번 결과는 시점별 예측과 world export이며 지도 정합 P6는 후속 |

validation loss와 validation F1이 소폭 좋아졌다는 사실만으로 LiDAR 결합이 성공했다고 말할 수 없다. 실제 test에서는 개선이 없었고 recall도 낮았다. train loss는 계속 줄지만 validation loss는 epoch 6 이후 개선되지 않아 과적합 가능성을 관찰했다. 이것은 곡선에서 얻은 해석이며, 성능 저하 원인을 확정한 것은 아니다.

학습의 의미는 GG49에서 비교 가능한 실행 기준과 실패 사례를 얻었다는 데 있다. 성능 목표를 달성했다고 포장해서는 안 되며, 반대로 이 한 번의 결과만으로 LiDAR가 무용하거나 추가 학습이 모두 무의미하다고 단정해서도 안 된다.

### 아직 확인하지 못한 원인과 한계

다음은 **검증 전 가설**이다. 현재 로그만으로 어느 항목이 원인인지 순위를 확정할 수 없다.

- ±0.1초 입력이 희소하므로 필요한 공간 정보를 충분히 담지 못했을 수 있다. 시간 누적은 아직 비교하지 않았다.
- 좌표·보정·LAS/카메라 시간 기준의 작은 오차가 잔차 특징의 유효성을 낮췄을 수 있다. 육안 투영 검사만으로 정밀 일치를 보장하지 않는다.
- 새로 구현한 잔차 결합 구조가 LiDAR를 충분히 활용하지 못했을 수 있다. 실제 gradient·입력 의존성은 확인했지만, 활용의 효용까지 증명한 것은 아니다.
- 한 트랙의 작은 공간 분할과 한 seed 결과이므로 다른 구간·트랙·seed에 대한 결론은 내리지 않는다.
- unknown 처리와 GT 기하·방향·가상 경계 구성의 품질을 더 검토할 필요가 있다. 빈 정답·의미가 불명확한 표본을 임의의 negative로 만들지 않았다.

### 다음 실험의 순서와 판정 기준

이 항목은 제안이며 아직 실행한 실험이 아니다. 단순히 epoch만 늘리는 것을 첫 조치로 삼지 않는다.

1. validation 대표 실패 사례에서 GT·pose·영상·LiDAR의 좌표와 시각 기준을 수치로 검토한다.
2. 같은 fusion checkpoint에서 정상 LiDAR, 0 입력, 시간/프레임을 섞은 입력의 차이를 validation에서 비교해 실제 의존성을 진단한다. 이는 재학습 효과의 인과 입증과는 구분한다.
3. validation에서 confidence별 precision/recall과 사례를 분석한다. test를 보면서 threshold를 맞추지 않는다.
4. 희소 LiDAR 누적 또는 결합 방식 중 한 조건을 바꾼 새 실험을 설계한다. 시간 누적은 pose 보정·split 경계·미래 관측 사용 여부를 명시하고 누수를 검사한다.
5. 새 조건을 정한 뒤 여러 seed와 새로 확보한 독립 구간에서 재현성을 확인한다. 이미 관찰한 현재 test는 개발 의사결정 이후 완전히 새로운 최종 검증 자료로 취급하지 않는다.
6. 시점별 예측 품질과 별도로 지도 정합 P6의 중복·오병합·위치오차·연결 보존을 평가한다.

### 주장 가능한 것과 주장하면 안 되는 것

- 가능: “동일한 GG49 공간 분할과 초기 가중치로 영상/RGB+LiDAR 각 24 epoch 비교 실험을 수행했고, 이번 test에서는 LiDAR 개선을 확인하지 못했다.”
- 불가: “LiDAR를 넣어 정확도가 개선됐다”, “멘토의 fusion 모델을 그대로 재현했다”, “한국 도로 전체에서 성능을 검증했다”, “여러 시점 차선 정합까지 완료했다.”

---

## 회고 2: 2026-10-24 10:35 KST - 종료 후 결과 보존과 대시보드 재실행 검증

### 문제 상황과 조치

학습이 완료된 뒤 사용자가 외출을 위해 컴퓨터를 종료해도 결과가 남고 다시 열 수 있는지 확인을 요청했다. 보고서·그래프·가중치는 D의 파일이므로 보존되지만, 재부팅 뒤에는 대시보드 서버를 다시 시작해야 한다.

D에 [OPEN_DASHBOARD.cmd](D:/mms_experiments/gg49-lanesegnet-lab_sj/OPEN_DASHBOARD.cmd)를 만들었다. `scripts/open_dashboard.ps1`과 `scripts/ensure_dashboard.sh`가 D 연결·WSL mount·기존 서버를 확인하고 필요할 때 서버를 기동한다. 정상 서버는 재사용하고 이 실행 파일에서 학습을 시작하지 않는다.

### 검증 결과와 한계

검증된 dashboard 프로세스만 실제 종료하고 Windows launcher의 `-NoBrowser` 경로로 재기동했다. API에서 두 모델의 completed·epoch24 상태, 모델별 24개 epoch 지표와 결과 보고서 재열림을 확인했다. 근거는 [dashboard-restart-check.json](D:/mms_experiments/gg49-lanesegnet-lab_sj/reports/dashboard-restart-check.json)이다.

**실제 Windows 재부팅 시험은 하지 않았다.** 서버 종료·재시작과 파일 보존을 확인한 것이며, D 드라이브 문자·WSL 배포판 이름·프로젝트 경로가 바뀌면 launcher 수정이 필요하다. 사용 방법은 START_HERE에 유지한다.

---

## 회고 3: 2026-10-24 15:48 KST - 실패 사례·LiDAR 기여도·검출 기준 진단

### 목적과 실행 조건

사용자가 승인한 세 진단을 기존 best(epoch6)로 수행했다. 전체188건(4조건×47시점)의 예측을 D에 보존하고, 기하 분절 진단을 보강할 때는 같은 캐시를 재사용했다. 모델·원본·이전 평가 결과는 수정하지 않았다. 새 학습이나 test 재평가는 수행하지 않았다.

```bash
cd /home/imella0707/personal/mms_project/gg49-lanesegnet-lab_sj
bash scripts/run_diagnostics.sh
```



### 실패 사례 시각화에서 발견한 점

`B_0082`에서 GT는6개 짧은 구간(약19.9~46.5m), fusion 예측은2개 긴 선(약94.6m,97.6m)이었다. 기존의 순서가 있는10개 표본 평균 거리로는 TP0/FP2/FN6이지만, GT 길이의99.60%가 예측 선과1m 이내였다. GT에서 예측으로의 길이 가중 평균 최근접3D 거리는 약0.288m였다.

즉, 기존 F1이 낮다는 사실만으로 선이 전부 도로에서 멀리 벗어났다고 해석할 수 없다. 선의 방향·범위·분절 정의가 정답과 다르면 위치가 가까워도 인스턴스 대응에 실패한다. 이는 기존 지표를 임의로 바꾸거나 기존 결과를 취소했다는 뜻이 아니다. LaneSegNet의 차로 인스턴스/연결 관계를 평가하려면 분절 재현도 여전히 검토해야 한다.

원래 지표는 그대로 보존하고, 방향·1:1 대응·topology를 평가하지 않는 **보조 기하 근접도**를 별도로 계산했다. 각 중심선을 길이를 따라120개 중간 위치로 샘플링하고 다른 선의 연속 선분까지의 최근접 XYZ 거리를 구해 길이로 가중했다. 시점별 길이를 합산했으며 중복 제거한 지도 길이가 아니다. 아래 비율은 기존 F1 또는 새 정확도 점수가 아니다.

| 조건 | 정답 길이 중 예측과1m 이내인 비율 | 예측 길이 중 정답과1m 이내인 비율 |
|---|---:|---:|
| 영상 전용 | 78.71% | 90.07% |
| 결합·정상 LiDAR | 75.16% | 89.08% |
| 결합·LiDAR 0 | 72.08% | 89.18% |
| 결합·다른 시점 LiDAR | 74.10% | 89.03% |


### LiDAR 기여도 검사

confidence0.5, 기존 ordered 중심선 거리1m 기준이다. 정상·0·교체는 모두 **동일한 fusion checkpoint**로 비교했다. 교체 donor는 validation 안의 seed42 고정 순열이며 자기 자신은 없다.

| 조건 | TP / FP / FN | Precision | Recall | F1@1m |
|---|---:|---:|---:|---:|
| 영상 전용 | 15 / 68 / 173 | 0.180723 | 0.079787 | 0.110701 |
| 결합·정상 LiDAR | 16 / 63 / 172 | 0.202532 | 0.085106 | 0.119850 |
| 결합·LiDAR 0 | 15 / 61 / 173 | 0.197368 | 0.079787 | 0.113636 |
| 결합·다른 시점 LiDAR | 16 / 62 / 172 | 0.205128 | 0.085106 | 0.120301 |

- LiDAR를0으로 바꾸면 동일 query의 중심선은 프레임 평균 약0.1993m 달라졌고, 다른 시점으로 교체하면 약0.0526m 달라졌다. 이 평균에는 낮은 점수의 query도 포함된다.
- 정상 입력을3개 시점에서 반복 실행했을 때 최대 좌표 차이는0m였다. 따라서 관측된 입력 변경 효과는 이번 반복 추론의 수치 차이보다 컸다.
- 정상 LiDAR는0 입력보다 기존 F1이 조금 높지만, 교체 입력과 F1이 거의 같다. 교체는 TP가 같고 FP가1개 적은 정도다. 이 한 번의 교체 결과로 센서 정보의 효용을 확정하지 않는다. 같은 주행 구간의 LiDAR들이 유사할 수 있다는 한계도 있다.
- 관측 셀 비율은 전체 BEV20,000셀의 프레임 평균 **2.39%**다. 입력이 희소하다는 관찰이며, 희소성이 성능 제한의 확정 원인이라는 뜻은 아니다.

**판정:** 모델이 LiDAR 입력에 반응하는 것은 확인했다. 올바른 시점의 LiDAR가 유용한 정보를 충분히 제공한다는 증거는 아직 약하며, 재학습에 따른 성능 이득의 인과 입증과도 구분해야 한다.

### Confidence 기준 분석

| 조건 | confidence0.5 F1 | validation 탐색 최고 기준 | 탐색 최고 F1 |
|---|---:|---:|---:|
| 영상 전용 | 0.110701 | 0.30 | 0.128617 |
| 결합·정상 LiDAR | 0.119850 | 0.30 | 0.145215 |
| 결합·LiDAR 0 | 0.113636 | 0.30 | 0.132450 |
| 결합·다른 시점 LiDAR | 0.120301 | 0.30 | 0.150820 |


두 기본 모델 모두 탐색 grid 안에서는0.3이 최고였다. fusion은 TP16→22, FP63→93으로 바뀐다. 검출 수 증가와 함께 오탐도 늘었으며, threshold만 낮춰 문제가 해결됐다고 결론 내리지 않는다. 이 값은 같은 validation에서 고른 탐색 결과이고 독립 test 성능이 아니다. 기본 운영 설정0.5나 기존 보고서를 자동 변경하지 않았다. 0.5m·1m·진단용2m 표와 그래프는 대시보드에서 선택한다.

### 화면과 근거

- 대시보드: http://127.0.0.1:8765/diagnostics
- 시점별 BEV 비교, GT 구간 끝점 표시, 정상/0/교체 비교, 카메라6대 이미지와 개별 레이어: `cases/B_XXXX/`
- 원문: `summary.json`, `manifest.json`, `thresholds.csv`, `DIAGNOSTICS_REPORT.md`
- 보정 영상 위 LiDAR는 원시 LAS가 아니라 BEV셀 중심과 최저 높이를 사용한 근사 투영이다. 정밀 보정 인증은 하지 않았다.
- 기존 confidence0.5 camera/fusion validation TP·예측 수·GT 수가 첫 평가와 일치하는지 검사했다. 입력·checkpoint hash, validation key와 donor 매핑을 보존했다.
- 반복 추론 중 공식 feature 추출의 `img.squeeze_()`가 입력 tensor 차원을 바꾸는 문제를 발견했다. 진단마다 이미지 tensor·메타데이터를 복사해 해결하고, 유효한 camera 캐시에서 재개했다. 실패 traceback은 `logs/diagnostics.log`에 보존했다.
- 전체 브라우저 자동조작 QA는 호스트 sandbox 초기화 오류로 실행하지 못했다. HTTP/API·JavaScript 구문 검사와 생성된 PNG 시각 검수를 수행했다.

### 다음 판단

우선 SHP의 차로 구간/방향 정의와 모델이 출력하는 긴 차로의 인스턴스 정의를 맞추고, 분절·방향을 요구하는 지표와 지도 기하 근접도를 구분해 평가해야 한다. 그다음 실제 LiDAR 시간·pose·누적 방식 및 독립적인 조건 비교를 검토한다. 이미 관찰한 test로 threshold나 모델 조건을 맞추지 않는다. 이번 진단만으로 추가24 epoch 학습이나 지도 정합 P6를 자동 시작하지 않았다.


진단 결과와 이미지는 D의 `reports/diagnostics-track-b-v1`에 저장했다. [진단 해석 보고서](D:/mms_experiments/gg49-lanesegnet-lab_sj/reports/diagnostics-track-b-v1/FINDINGS.md)를 함께 참고한다.

최종 검증: 계약 테스트7개 통과,188개 유한 예측·425개 이미지와47개 사례 API 확인, 기존 두 run의 best/last SHA256 불변을 검사했다. 근거는 [진단 검증 JSON](D:/mms_experiments/gg49-lanesegnet-lab_sj/reports/diagnostics-track-b-v1/verification.json)이다.

---

## 회고 4: 2026-10-24 16:10 KST - 학습/추론 reference 한 조건 대조



### 가설과 고정 조건

공식 로컬 decoder는 각 층의 회귀 결과를 reference에 반영해 저장한다. 학습 head는 마지막 층의 보정 전 reference를 사용하는 반면 기존 fast 추론은 보정 후 reference에 마지막 회귀를 다시 더하는 차이가 있었다. 이것이 긴 선 예측의 원인인지 검증했다.

데이터 Track B validation47개·정답·영상384×384·LiDAR·모델별 epoch6 best·confidence0.5를 고정했다. 바꾼 것은 마지막 reference 선택뿐이다. `align_inference_reference=True`는 실험 opt-in이며 기존 기본값과 이전 결과는 그대로 보존했다. 두 모델의 첫3개 시점(총6건)에서 기존 캐시 재현 및 수정 fast와 실제 학습 경로의 마지막 출력 동등성을 확인했다.

### 결과

| 모델 | 기존 F1@1m | 학습 정렬 F1@1m | 기존/정렬 F1@0.5m |
|---|---:|---:|---:|
| camera | 0.110701 | 0.110701 | 0.044280 / 0.036900 |
| fusion | 0.119850 | 0.119850 | 0.044944 / 0.044944 |


검사6건의 정렬 fast vs 학습 경로 최대 좌표/score 차이는0이었다. 전체47개에서 classification score는 기존과 같았다. 하지만 F1@1m은 두 모델 모두 변화가 없었고 영상 모델의0.5m 지표는 소폭 낮아졌다. `B_0082`의 긴 두 예측도 약95~98m로 유지됐다. 따라서 reference 차이는 확인됐으나, 이 변경이 분절 길이 문제를 해결하거나 성능을 개선했다는 근거는 없다. 코드 경로의 일관성 확인과 성능 향상을 구분한다.

### GT source 계약 검수

47개 시점의 `source_link_ids`를 원본 `NT2_LINK.shp`의ID로 대조했다. 누락 sourceID는 없었고, ego→world로 되돌린 GT 표본에서 해당 원본 링크까지의 최대 거리는 0.103879m였다. 이는 GT 표본이 원본 링크 기하를 따르는지의 검사이지 센서의 절대 정밀도·영상 보정 인증이 아니다. 점수를 올리려고 GT를 임의 병합하거나 정답 의미를 변경하지 않았다. 학습된 모델이 왜 긴 구간을 유지하는지에 대한 학습 원인까지 확정한 것은 아니다.



## 회고 5: 2026-10-24 16:10 KST - 두 구간 지도 정합과 관측 지지수 대조

 GT 없는 대응·병합과 GT 보조 평가

### 사전에 고정한 범위와 방법

- A: B_0082~B_0101, B: B_0102~B_0121, 각20개 연속 validation 시점. 서로 독립적인 새 test로 주장하지 않는다.
- 두 모델의 학습 정렬 추론, confidence0.5 고정. 추론 경로의 학습 일관성 때문에 선택했으며 GT 점수 최대 조건을 골라 사용하지 않았다.
- 시점 예측을 EPSG:32652로 변환하고 STRtree 공간색인으로 후보를 검색했다. SHP는 대응·가중평균 함수에 전달하지 않았다.
- 같은 방향(각도15도 이내), 겹침5m 이상/짧은 선의25% 이상, 겹침 구간 중앙거리0.65m/P95거리1m 이내, 높이 차0.5m 이내, 알려진 경계 유형 일치를 사용했다.
- 같은 시점의 다른 인스턴스는 합치지 않았다. 그룹에 합칠 때 겹치는 다른 멤버와도 모순이 없는지 검사해 연쇄 드리프트를 제한했다.
- 공통 station1m에서 confidence 가중 평균으로 중심선·좌우 경계를 함께 합쳤다. 관측이 없는 공간을 임의로 연결하지 않았다. 모든 source 관측 id와 채택·거부 이유를 보존했다.

### 사전 판정 기준과 추가 조건

중복 후보쌍 감소, GT 근접 길이 감소2%p 이내, 예측→GT P95 증가0.10m 이내, source 추적 보존과 병합 가드 위반0을 기준으로 고정했다. 조건은 `protocol.json`, `map-acceptance.json`에 저장했다.

기본 평균의 B구간이 거리 기준을 통과하지 못한 뒤, **한 조건만 추가**해 `관측2회 이상 지지되는 선분만 확정 지도에 표시`했다. 나머지는 지우지 않고 별도 불확실 레이어에 보존했다. 모델·대응 임계값·가중평균·판정 기준은 바꾸지 않았다. 새 조건의 이유와 규칙은 실행 전 `support-protocol.json`에 기록했다. GT를 보며 지지수 임계값을 탐색하지 않았다.

### 실제 결과

| 방식 | 구간/모델 | 관측→출력 | 중복 후보쌍 전→후 | GT 근접 길이 전→후 | 예측→GT P95 전→후 | 기준 |
|---|---|---:|---:|---:|---:|---|
| 평균 | A/camera | 33→5 | 162→2 | 93.89%→93.89% | 3.958→3.783m | 통과 |
| 평균 | A/fusion | 30→5 | 127→1 | 94.13%→94.13% | 4.066→3.890m | 통과 |
| 평균 | B/camera | 36→9 | 181→4 | 95.97%→95.97% | 0.863→1.515m | 미통과 |
| 평균 | B/fusion | 36→7 | 198→2 | 96.01%→96.01% | 0.846→1.228m | 미통과 |
| 관측2회 확정 | A/camera | 33→2 | 162→0 | 93.89%→83.84% | 3.958→4.253m | 미통과 |
| 관측2회 확정 | A/fusion | 30→2 | 127→0 | 94.13%→81.82% | 4.066→4.227m | 미통과 |
| 관측2회 확정 | B/camera | 36→7 | 181→4 | 95.97%→86.42% | 0.863→0.783m | 미통과 |
| 관측2회 확정 | B/fusion | 36→6 | 198→3 | 96.01%→80.08% | 0.846→1.904m | 미통과 |


기본 평균은 A의 두 모델만 통과했다. B는 중복이 줄었지만 예측→GT P95 증가 기준을 넘었다. 관측2회 확정 조건은 중복 그림을 단순화하더라도 GT 근접 길이가 약9~16%p 감소해4건 모두 통과하지 못했다. **보기 좋은 선 개수 감소를 성능 성공으로 처리하지 않았다.**

공통 GT 위치에서의 평균 최근접 오차도 함께 제공한다. B/camera는0.299→0.346m, B/fusion은0.292→0.332m다. 예측→GT P95는 반복 관측 각각을 세는 상태와 병합선 한 번을 세는 상태의 길이 가중치가 달라 분포 구성의 영향도 받는다. 이 한계를 기록하고 사전 미통과 판정은 유지했다.

### 오병합·연결 검사의 범위

인접 차선·반대 방향·교차·같은 시점·연쇄 드리프트·관측 공백·불확실 꼬리 보존은 합성 테스트로 확인했다. 실제 병합 가드 위반은0이었다. GT 보조 중간 지점 검사에서 판단 가능한 겹침쌍의 다른 차선 오병합 의심은0이었지만, 기본 A/camera17쌍·A/fusion14쌍은GT1m 안에서 판단하지 못했다. 이를 전체 의미/연결 정확도 보장으로 해석하지 않는다.

각 구간에는 GT adjacency에 따른 연결 지점 probe2개가 있었고 기본 평균 전후 모두2/2가 근접 지원됐다. 이는 연결 지점의 기하 지원이지 방향성 topology edge 정확도나 교차로 검증이 아니다. support 조건의 연결 지점 지원도 원문에 보존했다. 전체 Track·분기·합류·공유 경계의 전역 일관성은 후속 검증 대상이다. pose/카메라 extrinsic 자체를 새로 보정하는 ICP 실험은 수행하지 않았다.

### 산출물

`maps/A_camera` 등의 각 폴더에는 전후 PNG, 그룹 흔들림 PNG, `map.gpkg`, WGS84 `merged-wgs84.geojson`, 원본 관측/결합JSON, 판정 지표와 association ledger를 저장했다. GeoPackage는 EPSG:32652의3D LineString이며 observations/merged_centers/merged_left/merged_right/gt_reference 레이어를 포함한다. 지지수 조건은 uncertain_centers를 추가했다. Fiona1.9.6으로 다시 열어 레이어·CRS·건수를 검사했다.

## 결론과 다음 단계

1. 실험4는 코드 경로 차이를 확인했지만 지표 개선은 얻지 못했다. 새24 epoch 학습을 했다고 보고하지 않는다.
2. 실험5는 실제 두 구간에서 동작하는 대응·병합과 GIS 출력까지 구현했다. 기본 평균2/4, 추가 지지수 조건0/4 통과로 전 구간 성공은 아니다.
3. 다음 개선은 지도 구간의 분절/방향 계약과 길이 가중 평가를 정리하고, GT를 쓰지 않는 불확실도/편향 제어와 교차로 보존을 검증하는 것이다. 현재 test를 조건 선택에 쓰지 않는다.
4. 팀 설명은 `TEAM_BRIEF.md`와 http://127.0.0.1:8765/experiments 에서 제공한다. 과거 학습·진단 결과도 원래 화면과 원문에 그대로 남겼다.

## 실행·검증 근거

- 환경: WSL Ubuntu/mms, 기존 PyTorch1.13.1+cu117 스택 유지. GIS 표준 드라이버를 위해 Fiona1.9.6과 attrs/click-plugins/cligj만 추가했다. 환경 lock 갱신 및 pip check를 수행한다.
- GPU: reference 실험 중 점검한 전체 최대 2590MiB, 예산10752MiB 이내. 지도 병합은 CPU 작업이다.
- 원문: `reference-results.json`, `gt-contract-audit.json`, `map-results.json`, `map-supported-results.json`, `summary.json`, `verification.json`.
- 로그: D의 `logs/followup-reference.log`, `followup-map.log`, `followup-support.log`, `followup-visuals.log`.
- 표준 재실행: 실험 폴더의 `scripts/run_followup.sh`. 기존결과/캐시를 보존하고 정상 worker 중복 실행을 막는다.
- 브라우저 자동조작 QA는 호스트 sandbox 오류로 사용할 수 없어 전체 interaction QA를 주장하지 않는다. HTTP/API·스크립트 구문·생성 PNG·GIS 파일과 실제 수치를 검증한다.


팀 설명은 [TEAM_BRIEF](/home/imella0707/personal/mms_project/gg49-lanesegnet-lab_sj/docs/TEAM_BRIEF.md)에 정리했다.

최종 검증: 계약 테스트 11개 통과, 고정 가중치의 GPU 동등성 6건, 새로운 예측 94건·이미지 204개·GeoPackage 8개를 확인했다. 원래 두 모델의 best/last와 prepared dataset 해시가 유지됐고 pip check도 통과했다. [검증 원문](D:/mms_experiments/gg49-lanesegnet-lab_sj/reports/followup-45-v1/verification.json)을 보존했다.


## 2026-10-24 팀 공유용 CONFIG 정리·SHP 변환

실제 model.py/train.py/dataset.py/model_modes.json 및 두 run summary의 설정을 CONFIG_SUMMARY.md에 정리했다. 기존 followup-45-v1 지도 8건을 ESRI Shapefile 44레이어(합계 506피처)로 변환했다. 모든 레이어를 다시 열어 EPSG:32652, 3D LineString, 건수, XYZ 좌표(차이 1e-9m 이하), 속성을 원본 GPKG와 비교해 통과했다. ZIP 무결성 검사도 통과했다. 추가 학습·추론이나 원본 데이터 변경은 없다.

배포: D:/mms_experiments/gg49-lanesegnet-lab_sj/exports/team-share-v1.zip (3,881,734 bytes). 설정·발표 설명·정합 전후 이미지·추적 JSON·원본 GPKG·README·검증 기록을 동봉했다. 모두 Track B validation의 두 20시점 구간이며 전체 Track 결과가 아니다. 기본 평균 2/4 통과 및 supported 0/4 통과라는 한계를 그대로 설명했다. 재현 스크립트: scripts/export_team_shp.py.


## 2026-10-26 시연 시작 실패 복구

Windows KIOXIA D는 연결돼 있었으나 WSL /mnt/d가 마운트되지 않았다. 사용자가 단계별 명령을 계속 입력해 DATASET.json 읽기에서 KeyError frames, 실패 기록 저장 시 D unavailable 오류가 이어졌다. 학습은 시작되지 않았다. drvfs로 D를 마운트한 뒤 기존 353 frame manifest를 읽어 검증했다. dataset 로더의 마운트·manifest 검증, train 진입점의 마운트 사전 검사, 한 명령의 run_demo.sh를 추가했다. bash 문법과 실제 preflight를 통과했고, 모의 미연결·빈 manifest 두 조건이 명확한 오류로 차단됨을 검증했다. 본 학습은 재실행하지 않았다.


## 2026-10-26 시연 가중치의 실제 SHP 출력

demo-camera-20261026-224854-5342는 1 epoch/48 step 완료, validation loss 52.952475였다. 사용자 요청으로 이 run의 best.pth를 strict load해 첫 validation 20시점 B_0082~B_0101을 실제 추론했다. 기존 24 epoch 가중치나 예측은 사용하지 않았다. confidence 0.5 이상 예측은 5개였고, pred_centers/left/right 및 gt_reference SHP를 run/qgis-export-v1에 저장했다. 같은 시연 preview의 기본 legacy 추론 경로를 사용했고, 좌표는 차량→EPSG:32652 세계좌표로 변환했다. 정합하지 않은 시점별 예측이며 지도 완성이나 성능 개선을 뜻하지 않는다. 각 SHP 재개방으로 건수·3D geometry·CRS·XYZ·속성을 검증했고 가중치 해시 불변과 ZIP 검사를 통과했다. provenance와 frame별 건수는 export-summary.json에 있다. 재현 모듈은 src/gg49_lab/demo_export.py이다.


## 2026-10-28 새 PC 실행 (rjstns23)

기존 실험 결과와 분리한다. 실제 경로·환경 변경·실행 명령과 진행 상태는 LOCAL_WINDOWS_WSL_SJ.md 및 D의 runs/reports를 참조한다. 새 결과는 아직 측정 전이다.


## 2026-10-28 분할 보완 v2

초기 train191에는 양성 연결 정답이 0개였다. 기존 새 실행은 step146에서 안전 중지하고 checkpoint와 지표를 보존했다. 과거 저자 결과가 아니다.
입력 및 원래 test57을 그대로 유지하며 공간 X 구역을 다시 정했다. v2는 train228/val14/test57/buffer42/unlabelled12, 연결 정답 train24/val22/test48이다. 최소 구간 간 X 거리153.9m로 BEV ROI 겹침을 방지한다. 학습에는 새 공식 초기값을 사용한다.
접두사 sj-20261028-v2-. 원래 manifest는 artifacts/prepared/DATASET.initial-split-v1.json에 해시 그대로 보존한다. reports/split-policy-v2.json에 경계·근거가 있다. 검증14시점은 작고 인접해 있어 best 선택의 변동성과 일반화 한계가 있다. Test 모델 점수로 분할을 조정하지 않았다.


## SSD 상태 파일 교체 오류와 보완

v2 영상 모델 step157에서 status JSON 임시 파일 os.replace가 ENOENT로 실패했다. 정확한 OS 원인은 단정하지 않는다. 감독 프로세스가 마지막 checkpoint(step150)에서 자동 재개했고 저장되지 않은151~157은 다시 계산했다. 원본 metrics/log는 보존한다.
임시 파일 이름을 호출별 UUID로 바꾸고 fsync 및 제한된 재시도를 추가했다. 해당 오류와 동일 프로세스 동시 저장 회귀 테스트를 포함해13개 테스트를 통과했다. checkpoint도 새 임시 이름으로 원자 교체한다. Dashboard/그래프는 반복 step의 마지막 시도 값을 표시하며 원본 JSONL을 유지한다.
실제 본 학습을 안전 중지해 새 코드로 재개했다. 저장 step 397, epoch index 6, 데이터 위치 220. reports/main-v2-resume-saved.json 및 run의 resume-events.jsonl을 확인한다.


## 추론 입력 재사용 보완

두24epoch 학습 후 fusion 검증 입력 ablation에서 원래 엔진의 singleton batch squeeze_가 호출자 이미지 차원을 바꿔 IndexError가 발생했다. predict가 입력 텐서와 메타데이터 사본을 사용하도록 보완했다. 학습된 best를 그대로 사용해 평가를 재실행하며, 첫 camera 평가 산출물은 attempt1 파일로 보존했다. 학습을 다시 시작한 것이 아니다.


## 새 PC 실행 최종 검증

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
