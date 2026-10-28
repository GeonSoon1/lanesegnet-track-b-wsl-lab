# LaneSegNet Track B · Windows + WSL

카메라 6대와 촬영 시각에 맞춘 LiDAR로 Track B를 학습하고, 로컬 대시보드로 확인하며 QGIS용 SHP를 생성하는 실험 코드입니다.

이 저장소에는 코드·대시보드·문서만 있습니다. 원본 데이터·가중치·캐시·학습 로그·추론 산출물은 SSD에서 별도로 관리합니다. 출처와 로컬 보완 사항은 [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)를 확인하세요.

- [이 PC 실행·중지·재개 및 결과 안내](docs/LOCAL_WINDOWS_WSL_SJ.md)
- [쉬운 시작 안내](docs/START_HERE.md)
- [PRD](docs/PRD.md)
- [학습 실험 기록·회고](docs/mms_lanesegnet_training_experiment.md)
- [실험4·5 / 지도 정합 / 팀 설명](http://127.0.0.1:8765/experiments)
- [팀 공유용 쉬운 설명](docs/TEAM_BRIEF.md)
- [Track B 진단실](http://127.0.0.1:8765/diagnostics)
- [실시간 현황과 loss 그래프](http://127.0.0.1:8765)

코드와 문서는 현재 WSL 폴더에 유지합니다. 데이터·캐시·checkpoint·로그·그래프는 `D:/mms_experiments/gg49-lanesegnet-lab_sj`에 저장하고 원본 데이터는 수정하지 않습니다. 이 PC의 새 가상환경은 `.venv-mms`이며, 기존 환경은 유지했습니다.

실행 상태는 D의 `runs/pipeline-status.json`과 모델별 `summary.json`에서 확인합니다. 두 모델은 각각 최대 24 epoch이며, 실제 checkpoint와 평가 보고서까지 생성해야 완료입니다.


## 팀원이 복제한 뒤 확인할 사항

```bash
git clone https://github.com/GeonSoon1/lanesegnet-track-b-wsl-lab.git gg49-lanesegnet-lab_sj
cd gg49-lanesegnet-lab_sj
```

이 저장소는 실험 코드·문서·dashboard와 수정한 LaneSegNet 엔진을 포함합니다. **원본 데이터, 초기/학습 가중치, 캐시, 실행 결과, 가상환경은 포함하지 않습니다.** 초기 `lanesegnet_r50_8x1_24e_olv2_subset_A.pth`와 팀 SSD 자료는 별도로 준비해야 합니다.

1. [실제 CONFIG](docs/CONFIG_SUMMARY.md), [시작 안내](docs/START_HERE.md), [실험 기록](docs/mms_lanesegnet_training_experiment.md)을 읽으세요. 과거 결과가 clone만으로 생기는 것은 아닙니다.
2. GT 준비는 별도 [멘토 GT builder](https://github.com/justin95214/gg49-lanesegnet-gt-builder)가 이 저장소의 형제 폴더에 필요합니다. `prepare.py`가 해당 저장소의 scripts/build_gt.py와 configs/gg49-codebook.json을 참조합니다.
3. 원래 저장소에는 기존 작성자의 WSL 사용자·mms 환경·데이터 경로가 있었습니다. 이 PC는 LOCAL_WINDOWS_WSL_SJ.md와 storage.local.json에 실제 경로를 기록했고, 원래 계획 설정은 experiment.author-history.json에 보존했습니다. `configs/storage.local.json`과 소스/스크립트의 절대 경로를 본인 PC에 맞추고, 실제 자료와 환경을 사전 확인하세요. clone 직후 다른 PC에서 무설정 실행되는 배포판은 아닙니다. `setup_mms.sh`도 경로를 검토한 뒤 사용하세요.
4. 대용량 데이터·캐시·로그·checkpoint·SHP는 SSD 실험 폴더에 저장하세요. 기본값은 Windows D:/mms_experiments/gg49-lanesegnet-lab_sj, WSL /mnt/d/mms_experiments/gg49-lanesegnet-lab_sj입니다. SSD 미마운트 시 C로 대체 저장하지 마세요.
5. 학습 산출물은 .pth이며, QGIS용 SHP는 해당 가중치로 추론·지도 좌표 변환·내보내기를 수행해 따로 생성합니다. 1 epoch 시연의 결과와 24 epoch 본 실험을 구분하세요.
6. localhost 대시보드는 해당 PC에서 서버를 실행한 후 이용합니다. GitHub에서 링크만 누른다고 원 작성자의 결과를 볼 수 있는 것은 아닙니다.

외부 코드 출처와 수정 파일은 [THIRD_PARTY.md](THIRD_PARTY.md)를 참고하세요.
