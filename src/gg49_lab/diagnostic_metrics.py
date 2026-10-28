"""Validation diagnostics with the same ordered-distance assignment as evaluate.py."""
import numpy as np
from scipy.optimize import linear_sum_assignment

THRESHOLDS=[0.,.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]

def derangement(ids,seed=42):
    ids=list(ids)
    if len(ids)<2:raise ValueError('Need at least two validation frames')
    rng=np.random.RandomState(seed)
    for _ in range(10000):
        donor=rng.permutation(ids).tolist()
        if all(a!=b for a,b in zip(ids,donor)):return dict(zip(ids,donor))
    raise RuntimeError('Could not construct a derangement')

def measure(lanes,scores,gt,confidence=.5,tolerance=1.):
    keep=np.flatnonzero(scores>=confidence);selected=lanes[keep]
    if len(selected) and len(gt):
        distance=np.linalg.norm(selected[:,None,0]-gt[None,:,0],axis=-1).mean(-1)
        pi,gi=linear_sum_assignment(distance)
        accepted=[(int(keep[a]),int(b),float(distance[a,b])) for a,b in zip(pi,gi) if distance[a,b]<=tolerance]
    else:accepted=[]
    tp=len(accepted);n=len(keep);g=len(gt)
    p=tp/n if n else 0.;r=tp/g if g else 0.
    return {'tp':tp,'predicted':n,'gt':g,'fp':n-tp,'fn':g-tp,'precision':p,'recall':r,'f1':2*tp/(n+g) if n+g else 0.,'matches':accepted}

def combine(rows):
    sums={k:sum(r[k] for r in rows) for k in ['tp','predicted','gt','fp','fn']}
    t,n,g=sums['tp'],sums['predicted'],sums['gt']
    sums.update(precision=t/n if n else 0.,recall=t/g if g else 0.,f1=2*t/(n+g) if n+g else 0.)
    return sums


def direction_free_geometry(lanes,gt,tolerance=1.,samples=120):
    # Length-weighted proximity, without direction or 1:1 association. Not an instance F1.
    def sample(curve):
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
        if distance[-1]<=1e-9:return np.repeat(curve[:1],samples,axis=0),0.
        valid=np.r_[True,np.diff(distance)>1e-9]
        q=(np.arange(samples)+.5)/samples*distance[-1]
        return np.column_stack([np.interp(q,distance[valid],curve[valid,k]) for k in range(3)]),float(distance[-1])
    def evaluate(source,target):
        length=supported=distance_total=0.
        if len(target):
            a=target[:,0,:-1].reshape(-1,3);v=np.diff(target[:,0],axis=1).reshape(-1,3)
            vnorm=np.maximum((v*v).sum(-1),1e-12)
        for lane in source:
            points,n=sample(lane[0]);length+=n
            if not len(target):continue
            delta=points[:,None,:]-a[None];t=np.clip((delta*v).sum(-1)/vnorm,0,1)
            distances=np.linalg.norm(delta-t[...,None]*v,axis=-1).min(-1)
            supported+=float(np.mean(distances<=tolerance))*n
            distance_total+=float(distances.mean())*n
        return length,supported,distance_total/length if length and len(target) else None
    gl,gs,gm=evaluate(gt,lanes);pl,ps,pm=evaluate(lanes,gt)
    return {'gt_length_m':gl,'gt_supported_length_m':gs,'gt_supported_fraction':gs/gl if gl else None,'pred_length_m':pl,'pred_supported_length_m':ps,'pred_supported_fraction':ps/pl if pl else None,'gt_mean_nearest_distance_m':gm,'pred_mean_nearest_distance_m':pm}
