#!/usr/bin/env bash
source "$(dirname "$0")/local_env.sh"
"$PY" -m pip install 'numpy==1.26.4' 'setuptools<81' wheel
"$PY" -m pip install 'mmcv-full==1.7.0' --only-binary mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
"$PY" -m pip install 'mmdet==2.26.0' 'scipy==1.11.4' 'opencv-python==4.10.0.84' 'pyshp==2.3.1' 'pyproj==3.6.1' 'shapely==2.0.6' 'laspy==2.5.4' 'matplotlib==3.9.4' 'tensorboard==2.17.1' 'pytest==8.3.5' 'psutil==6.1.1' 'einops==0.8.1' 'fiona==1.9.6' 'yapf==0.40.1'
"$PY" -m pip check
"$PY" -m pip freeze > "$STORE/reports/environment-local.lock.txt"
"$PY" -c 'import torch,mmcv,mmcv.ops; print(torch.__version__,torch.cuda.is_available()); x=torch.randn(1,3,16,16,device="cuda"); m=torch.nn.Conv2d(3,4,3).cuda(); m(x).sum().backward(); print("CUDA_FORWARD_BACKWARD_OK")'
