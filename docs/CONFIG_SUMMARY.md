> 2026-10-28 이 PC의 새 실험은 `sj-20261028-v2-`입니다. 아래 기존 저자의 결과와 구분하세요. 실제 경로·분할·환경은 [LOCAL_WINDOWS_WSL_SJ.md](LOCAL_WINDOWS_WSL_SJ.md), 현재 완료 여부는 SSD의 runs/pipeline-status.json을 확인하세요.

# GG49 실제 CONFIG 설정 — 팀 공유용

## 설명할 한 문장
공식 LaneSegNet R50 전체 체크포인트에서 시작해 Track B의 카메라 6대와 SHP 정답으로 영상 모델을 학습하고, LiDAR BEV 결합 모델도 같은 조건으로 각각 24 epoch 학습해 비교했습니다. LiDAR를 추가했지만 성능 개선을 입증하지는 못했습니다.

## 실제 학습 설정
| 항목 | 사용한 값 | 의미 |
|---|---|---|
| 모델 | LaneSegNet + ResNet-50 + FPN | R50은 영상 특징을 뽑는 backbone이며 전체 모델 이름은 LaneSegNet |
| 초기 가중치 | lanesegnet_r50_8x1_24e_olv2_subset_A.pth | 호환되는 이름·shape만 로드. 카메라 임베딩과 다른 크기 출력은 새로 초기화 |
| 카메라 | 01~06, 고정 순서 6대 | Sphere 영상은 입력하지 않음 |
| 영상 | 카메라별 RGB 384×384 | 왜곡 보정과 크기 변경에 투영행렬도 맞춤 |
| 정규화 | mean [123.675,116.28,103.53], std [58.395,57.12,57.375] | 캐시 RGB에 적용; 공식 AV2 augmentation을 그대로 실행한 것은 아님 |
| 데이터 | Track B; train 191 / val 47 / test 57 | 공간 분할, buffer 46 및 unlabelled 12 제외. 두 모델은 동일 분할 사용 |
| 정답 | SHP에서 변환한 차선 geometry·경계 type·연결 관계 | 차선 클래스 1개. 타입 none/solid/dashed; unknown -1은 type loss에서 제외 |
| BEV | 100×200, 특징 차원 256 | 차량 주변을 위에서 보는 특징 지도 |
| 공간 범위 | x [-51.2,51.2], y [-25.6,25.6], z [-4,2] m | 차량 기준 좌표 |
| 출력 | query 200개; 차선별 중앙·좌·우 각각 XYZ 10점 | 차선당 90개 좌표 값 |
| 학습량 | 각 24 epoch, 실제 optimizer update 1,152회 | 두 모델 각각 독립 학습 |
| 배치 | micro batch 2 × 누적 2회 = 유효 배치 4 | 마지막 묶음은 작을 수 있음 |
| optimizer | AdamW, weight decay 0.01 | base LR 0.00002; backbone LR 0.000002 |
| LR schedule | 첫 100 update 선형 warmup, 이후 cosine | 최소 비율 0.001 |
| 안정화 | gradient norm clip 5; activation checkpointing | 영상 CNN AMP, BEV/head·geometry/loss FP32 |
| seed | 42 | 같은 데이터·초기값 조건 비교 |
| backbone 동결 | frozen_stages=1, BN 동결 | backbone 전체를 다 업데이트한 것은 아님 |
| 환경 | mms, RTX 3060 12GB | GPU 사용 예산 10.5GiB |
| checkpoint 선택 | 매 epoch validation total loss가 가장 낮은 best | 두 모델 모두 best epoch 6; last는 epoch 24 |

## 영상 모델과 LiDAR 결합 모델 차이
| 항목 | camera | fusion |
|---|---|---|
| use_camera | true | true |
| 실제 use_lidar | false | true |
| 입력 | 카메라 6대 | 동일 카메라 + 촬영 시각 ±0.1초 LiDAR |
| LiDAR 처리 | 사용하지 않음 | 5채널 BEV → CNN 5→64→64→256 → 영상 BEV에 잔차 합산 |

LiDAR 5채널은 점 개수·평균 intensity·최소 높이·최대 높이·관측 mask입니다. 마지막 projection은 0으로 초기화하고, 관측 mask를 곱해 영상 BEV에 더합니다. 시간 누적 지도는 넣지 않았습니다. camera 학습 결과를 이어받아 fusion을 시작한 것이 아니라, 두 모델 모두 같은 공식 초기 가중치에서 시작했습니다.

공식 config의 input_modality.use_lidar=False만 true로 바꾼 것이 아닙니다. 실제 실행에서는 model_modes.json과 model.py의 LiDARFusion, 데이터 로더가 함께 작동합니다. 멘토의 완전한 fusion 구현을 받은 것이 아니므로 멘토 모델의 정확한 재현이라고 말하지 않습니다. 체크포인트 이름 8x1은 공식 GPU×배치 설정이며 카메라 8대라는 뜻이 아닙니다.

## Loss 설정
| 항목 | 함수 | 가중치 |
|---|---|---|
| 차선 클래스 | sigmoid Focal (gamma 2, alpha 0.25) | 1.5 |
| 차선 좌표 | L1 | 0.025 |
| 차선 mask | sigmoid Cross Entropy | 3 |
| 차선 mask 형태 | Dice | 3 |
| 경계 타입 | Masked categorical Cross Entropy | 0.1 |
| 연결 관계 | Focal | 5 |

6개 decoder layer의 loss와 topology loss를 합산합니다. 경계 타입은 실제 F.cross_entropy 구현을 따르며, unknown/background를 제외합니다.

## 평가와 후처리 설정은 학습 설정과 구분
- 첫 held-out test의 confidence는 0.5, 10점 XYZ 대응 평균 거리 기준으로 평가했습니다. F1@1m은 두 모델 모두 0.284672였습니다. 공식 OpenLane-V2 benchmark 점수가 아닌 로컬 평가입니다.
- 이후 confidence 0.3 분석은 validation 진단입니다. 공식 운영 기준을 바꾸거나 test 개선을 증명한 것이 아닙니다.
- 후속 실험 4는 같은 best checkpoint에서 추론 좌표 reference만 바꾼 비교입니다. 추가 학습은 하지 않았고 F1@1m도 개선되지 않았습니다.
- SHP로 제공하는 실험 5는 validation의 두 연속 구간입니다. A=B_0082~B_0101, B=B_0102~B_0121입니다. 여기서 A/B는 Track A/B가 아니라 모두 Track B 안의 구간 이름입니다.
- 정합 기준: confidence 0.5, 방향차 15도 이내, median XY 0.65m / p95 XY 1m / median Z 0.5m 이내, 겹침 최소 5m 및 비율 25%, 샘플 간격 1m. GT는 평가에만 사용했습니다.
- 기본 평균 병합은 4건 중 A 구간 두 모델만 사전 기준을 통과했습니다. B 구간은 위치오차 증가로 실패했고, 2회 관측 확정 변형은 누락 증가로 4건 모두 실패했습니다. 전체 Track 완성 지도가 아닙니다.

## 실제 값의 출처
- engine/projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py: 공식 구조·기본 loss
- src/gg49_lab/model.py: 카메라 수·공간 범위·클래스·fusion·타입 loss override
- src/gg49_lab/train.py: optimizer·batch 누적·LR·학습 loop
- src/gg49_lab/dataset.py: 입력과 정규화
- configs/model_modes.json: 실제 모델 모드
- D:/mms_experiments/gg49-lanesegnet-lab_sj/runs/{camera,fusion}-24e/summary.json: 실행된 설정·24 epoch 완료 기록
- configs/experiment.local.json은 planning/preflight용이며 일부 과거 항목이 남아 있습니다. 실제 실행 config로 인용하지 마세요.

설계는 PRD.md, 실측 결과와 실패 원인은 mms_lanesegnet_training_experiment.md, 발표 흐름은 TEAM_BRIEF.md를 참고합니다.
