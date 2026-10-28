import numpy as np
from gg49_lab.diagnostic_metrics import measure,combine,derangement

def lane(y=0.,reverse=False):
    x=np.linspace(-5,5,10)
    a=np.column_stack([x,np.full(10,y),np.zeros(10)])
    if reverse:a=a[::-1]
    return np.stack([a,a+np.array([0,1,0]),a-np.array([0,1,0])])

def test_confidence_filter_changes_counts_without_counting_rejected_prediction():
    gt=np.stack([lane(0),lane(5)])
    pred=np.stack([lane(.2),lane(5.2),lane(15)])
    high=measure(pred,np.array([.9,.2,.8]),gt,.5,1.)
    assert (high['tp'],high['fp'],high['fn'])==(1,1,1)
    low=measure(pred,np.array([.9,.2,.8]),gt,.1,1.)
    assert (low['tp'],low['fp'],low['fn'])==(2,1,0)
    assert combine([high,low])['f1']==6/9

def test_empty_predictions_and_ordered_direction_are_not_silent_true_positives():
    gt=np.stack([lane()])
    empty=measure(np.empty((0,3,10,3)),np.empty(0),gt)
    assert (empty['tp'],empty['fn'],empty['f1'])==(0,1,0)
    reversed_lane=measure(np.stack([lane(reverse=True)]),np.array([.9]),gt)
    assert reversed_lane['tp']==0
    assert measure(gt,np.array([.9]),gt)['tp']==1

def test_donor_mapping_is_deterministic_bijective_and_validation_only():
    ids=list(range(82,129));a=derangement(ids)
    assert a==derangement(ids)
    assert sorted(a)==ids and sorted(a.values())==ids
    assert all(i!=j for i,j in a.items())


def test_split_gt_can_be_geometrically_close_while_instance_distance_rejects_it():
    from gg49_lab.diagnostic_metrics import direction_free_geometry
    full=lane()
    left=lane();right=lane()
    left[:,:,0]=np.linspace(-5,0,10);right[:,:,0]=np.linspace(0,5,10)
    gt=np.stack([left,right]);pred=np.stack([full])
    assert measure(pred,np.array([.9]),gt)['tp']==0
    proximity=direction_free_geometry(pred,gt)
    assert proximity['gt_supported_fraction']==1
    assert proximity['pred_supported_fraction']==1
    assert proximity['gt_mean_nearest_distance_m']<1e-6
