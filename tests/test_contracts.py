import torch
from gg49_lab.model import MaskedLaneTypeLoss,LiDARFusion

def test_unknown_is_masked_but_none_is_supervised():
    x=torch.randn(4,3,requires_grad=True);loss=MaskedLaneTypeLoss()(x,torch.tensor([-1,0,1,2]));loss.backward()
    assert x.grad[0].abs().sum()==0
    assert (x.grad[1:].abs().sum(1)>0).all()

def test_lidar_residual_identity_then_real_sensor_dependency():
    m=LiDARFusion();rgb=torch.randn(1,20000,256);lidar=torch.rand(1,5,100,200);lidar[:,4]=1
    assert torch.equal(m(rgb,lidar),rgb)
    torch.nn.init.normal_(m.projection.weight,std=.01)
    changed=m(rgb,lidar);assert not torch.allclose(changed,rgb)
    empty=lidar.clone();empty[:,4]=0
    assert torch.equal(m(rgb,empty),rgb)
    changed.square().mean().backward()
    assert m.encoder[0].weight.grad.abs().sum()>0


def test_storage_loss_never_falls_back_to_c(monkeypatch):
    from gg49_lab import common
    import pytest
    destination=common.STORE/'runs/storage-guard-test.json'
    assert not destination.exists()
    monkeypatch.setattr(common.os.path,'ismount',lambda _:False)
    with pytest.raises(RuntimeError,match='no fallback'):
        common.write_json(destination,{'should_not_write':True})
    assert not destination.exists()
