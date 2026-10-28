# Third-party code

`engine/projects/` is a local adaptation of LaneSegNet and its bundled BEVFormer/third-party modules. The upstream Apache-2.0 license is preserved in `engine/LICENSE`; existing source attribution headers are retained. See the upstream LaneSegNet project: https://github.com/OpenDriveLab/LaneSegNet .

GG49 modifications relative to the local upstream source copy (ignoring line-ending changes):
- `engine/projects/lanesegnet/__init__.py`
- `engine/projects/lanesegnet/thirdparty/maptr_head.py`
- `engine/projects/lanesegnet/models/detectors/lanesegnet_mapele_bucket.py`
- `engine/projects/lanesegnet/models/detectors/lanesegnet.py`
- `engine/projects/lanesegnet/models/detectors/__init__.py`
- `engine/projects/lanesegnet/models/dense_heads/laneseg_head.py`

The lab adds the separate RGB+LiDAR adapter, dataset pipeline, training/evaluation, diagnostics, dashboard and GIS export under `src/`, `scripts/`, and `dashboard/`. Original datasets and checkpoint binaries are not distributed in this repository.

GT preparation depends on the separate mentor repository https://github.com/justin95214/gg49-lanesegnet-gt-builder , expected beside this checkout. It is not vendored or silently downloaded here.
