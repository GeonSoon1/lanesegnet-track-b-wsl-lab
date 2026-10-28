import numpy as np
from gg49_lab.stitching import stitch,pair_stats
P={'max_angle_deg':15,'median_distance_m':.65,'p95_distance_m':1.,'max_z_m':.5,'min_overlap_m':5.,'min_overlap_ratio':.25,'station_step_m':1.,'max_unobserved_gap_m':1.5,'min_length_m':5.}
def obs(id,frame,start=0,end=20,y=0,reverse=False):
 c=np.column_stack([np.linspace(start,end,10),np.full(10,y),np.zeros(10)])
 if reverse:c=c[::-1]
 return dict(id=id,frame=frame,lanes=np.stack([c,c+[0,1.7,0],c-[0,1.7,0]]),score=.8,left_type=1,right_type=1)
def test_repeated_observations_merge_and_preserve_provenance_and_boundaries():
 result,ledger=stitch([obs('a','f1'),obs('b','f2',10,30,.1)],P)
 assert len(result)==1 and set(result[0]['source_ids'])=={'a','b'}
 assert result[0]['lanes'].shape[0]==3 and max(result[0]['support_per_vertex'])==2
 assert np.isclose(result[0]['lanes'][0,0,0],0) and np.isclose(result[0]['lanes'][0,-1,0],30)
def test_adjacent_opposite_direction_crossing_and_same_frame_are_not_merged():
 a=obs('a','f1')
 for b in [obs('b','f2',y=3),obs('b','f2',reverse=True),obs('b','f1')]:assert len(stitch([a,b],P)[0])==2
 b=obs('b','f2');b['lanes']=b['lanes'][:,:,[1,0,2]];assert len(stitch([a,b],P)[0])==2
def test_transitive_drift_does_not_merge_all_and_gaps_are_not_invented():
 result,_=stitch([obs('a','f1',y=0),obs('b','f2',y=.5),obs('c','f3',y=1.)],P)
 assert len(result)==2
 assert len(stitch([obs('a','f1',0,10),obs('b','f2',20,30)],P)[0])==2


def test_support_filter_preserves_uncertain_tails_instead_of_inventing_evidence():
    from gg49_lab.stitching import split_by_support,length
    groups,_=stitch([obs('a','f1'),obs('b','f2',10,30,.1)],P)
    confirmed,uncertain=split_by_support(groups)
    assert confirmed and uncertain
    assert all(min(r['support_per_vertex'])>=2 for r in confirmed)
    total=sum(length(r['lanes'][0]) for r in confirmed+uncertain)
    assert np.isclose(total,length(groups[0]['lanes'][0]))
    assert set(s for r in confirmed+uncertain for s in r['source_ids'])=={'a','b'}
