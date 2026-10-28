"""Local spatial split, audited for train/validation positive topology labels.

The held-out test region remains the original author's upper-X region. No test
prediction/metric is used to choose these boundaries. Cache coordinates stay fixed.
"""
import collections
import numpy as np

def apply_local_split(rows):
    xs=np.array([r['origin'][0] for r in rows])
    original_test_min=float(np.percentile(xs,80)+75)
    for row in rows:
        x=row['origin'][0]
        row['split']=('unlabelled' if not row['joint_eligible'] else
                      'test' if x>original_test_min else
                      'train' if x<253040 else
                      'val' if 253190<=x<=253285 else 'buffer')
    counts=dict(collections.Counter(r['split'] for r in rows))
    assert min(counts.get(k,0) for k in ['train','val','test'])>=8
    gaps={}
    for a,b in [('train','val'),('val','test'),('train','test')]:
        one=np.array([r['origin'][0] for r in rows if r['split']==a])
        two=np.array([r['origin'][0] for r in rows if r['split']==b])
        gap=float(np.abs(one[:,None]-two[None,:]).min())
        assert gap>=150, 'Spatial separation must exceed two 57.25m ROI radii'
        gaps[a+'_'+b]=gap
    labels={}
    for split in ['train','val','test']:
        edges=0;lanes=0
        for row in rows:
            if row['split']!=split:continue
            with np.load(row['gt_path']) as gt:
                edges+=int((gt['adjacency']>0).sum());lanes+=len(gt['lanes'])
        labels[split]={'directed_positive_edges':edges,'lane_instances':lanes}
    assert labels['train']['directed_positive_edges']>0
    assert labels['val']['directed_positive_edges']>0
    return {'policy':'gs-spatial-topology-v2','split_counts':counts,
            'train_x_max_exclusive':253040,'val_x_interval_inclusive':[253190,253285],
            'test_x_min_exclusive':original_test_min,'minimum_cross_split_x_gaps_m':gaps,
            'label_counts':labels,'test_region_unchanged':True,
            'limitations':['Validation has only 14 nearby frames; best selection may be noisy.',
                          'Spatial separation is not an independent drive or weather holdout.',
                          'Label coverage audit is not test model tuning.']}
