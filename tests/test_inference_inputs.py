"""Prediction inputs remain reusable even with the legacy singleton squeeze."""
from types import SimpleNamespace
import torch
from gg49_lab.model import LabModel

def test_repeated_prediction_preserves_callers_batch_shape_and_metadata():
    class Head:
        def __call__(self,*args):return {'history_states':torch.zeros(1)}
        def get_lanes(self,*args):return []
    class Model:
        net=SimpleNamespace(pts_bbox_head=Head(),lclc_head=SimpleNamespace(get_relationship=lambda *args:[]))
        def features(self,img,metas,lidar):
            assert img.ndim==5
            img.squeeze_()
            metas[0]['nested']['changed']=True
            return [],None
    image=torch.zeros(1,6,3,2,2);metadata=[{'nested':{'changed':False}}]
    for _ in range(2):LabModel.predict(Model(),image,metadata)
    assert image.shape==(1,6,3,2,2)
    assert metadata==[{'nested':{'changed':False}}]
