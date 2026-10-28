from mmdet.models.detectors.base import BaseDetector
from mmdet.models.builder import build_backbone,build_neck
class CameraDetectorBase(BaseDetector):
    """Actual camera branch initialization; unused 3D detector dependencies omitted."""
    def __init__(self,img_backbone=None,img_neck=None,train_cfg=None,test_cfg=None,pretrained=None,init_cfg=None,**kwargs):
        super().__init__(init_cfg=init_cfg)
        self.img_backbone=build_backbone(img_backbone)
        if img_neck is not None:self.img_neck=build_neck(img_neck)
        self.train_cfg=train_cfg;self.test_cfg=test_cfg
    @property
    def with_img_neck(self):return hasattr(self,'img_neck') and self.img_neck is not None
    def aug_test(self,*args,**kwargs):raise NotImplementedError('Single-scale GG49 inference only')
