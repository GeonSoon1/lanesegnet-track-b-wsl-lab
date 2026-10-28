"""Check real geometry smoke NPZ and draw a BEV overview (not camera overlay)."""
import argparse, collections, json
from pathlib import Path
import cv2
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default='track-b-smoke12')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    work=root/'artifacts'/args.run
    manifest=json.loads((work/'MANIFEST.json').read_text())
    staging=json.loads((root/'data'/args.run/'stage-manifest.json').read_text())
    rows=manifest['frames']
    summary={'run':args.run,'status':'geometry_only','training_ready':False,'frames':len(rows),'eligible_frames':sum(r['supervised_eligible'] for r in rows),'split_counts_for_smoke_only':dict(collections.Counter(r['split'] for r in rows)),'lane_instances':0,'frames_detail':[],'limits':['Sampled EO only: image paths in generated manifest are not training paths.','Geometry type 0 is a placeholder, not semantic supervision.','No RGB projection, semantic GT, model forward or training was tested.','Sample-derived spatial split is not an approved training/evaluation split.']}
    width,cellw,cellh=1290,430,270
    canvas=np.full((100+cellh*((len(rows)+2)//3),width,3),250,np.uint8)
    cv2.putText(canvas,'Track B | SHP-derived geometry smoke | 12 sampled timepoints',(22,32),cv2.FONT_HERSHEY_SIMPLEX,.75,(32,32,32),2)
    cv2.putText(canvas,'Center: purple   Left: orange   Right: green   Ego +X: right, +Y: up',(22,62),cv2.FONT_HERSHEY_SIMPLEX,.55,(65,65,65),1)
    cv2.putText(canvas,'BEV geometry only; not RGB overlay, predictions, or training results',(22,84),cv2.FONT_HERSHEY_SIMPLEX,.5,(65,65,65),1)
    for idx,row in enumerate(rows):
        with np.load(work/'gt'/(row['key']+'.npz'),allow_pickle=False) as d:
            lanes=d['lanes'];n=len(lanes)
            assert lanes.shape==(n,3,10,3)
            assert d['masks'].shape==(n,100,200)
            assert np.isfinite(lanes).all()
            assert np.isin(d['masks'],[0,1]).all()
            assert len(d['source_link_ids'])==n
            origin=np.asarray(row['origin']);rotation=np.asarray(row['rotation'])
            assert np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
            if n:
                world=lanes@rotation.T+origin
                assert np.allclose((world-origin)@rotation,lanes,atol=1e-6)
            detail={'key':row['key'],'original_index_zero_based':staging['indices'][idx]['original_index_zero_based'],'lane_instances':n,'nearest_shp_m':row['GT_nearest_m'],'supervised_eligible':row['supervised_eligible'],'z_min_m':float(lanes[:,:,:,2].min()) if n else None,'z_max_m':float(lanes[:,:,:,2].max()) if n else None}
            summary['frames_detail'].append(detail);summary['lane_instances']+=n
            x=(idx%3)*cellw+15;y=100+(idx//3)*cellh
            cv2.putText(canvas,f"{row['key']} | source #{detail['original_index_zero_based']} | lanes={n}",(x,y+19),cv2.FONT_HERSHEY_SIMPLEX,.48,(35,35,35),1)
            x0,y0=x,y+35;pw,ph=400,200
            def uv(a):
                return np.rint(np.column_stack(((a[:,0]+51.2)/102.4*pw+x0,(25.6-a[:,1])/51.2*ph+y0))).astype(np.int32)
            cv2.rectangle(canvas,(x0,y0),(x0+pw,y0+ph),(210,210,210),1)
            for gx in [-40,-20,0,20,40]:
                pts=uv(np.array([[gx,-25.6],[gx,25.6]]));cv2.line(canvas,tuple(pts[0]),tuple(pts[1]),(232,232,232),1)
            for gy in [-20,0,20]:
                pts=uv(np.array([[-51.2,gy],[51.2,gy]]));cv2.line(canvas,tuple(pts[0]),tuple(pts[1]),(232,232,232),1)
            for lane in lanes:
                for k,color in enumerate([(155,65,130),(0,150,225),(60,150,65)]):
                    cv2.polylines(canvas,[uv(lane[k,:,:2])],False,color,2,cv2.LINE_AA)
            cv2.drawMarker(canvas,(x0+pw//2,y0+ph//2),(20,20,20),cv2.MARKER_CROSS,9,1)
            flag='eligible' if row['supervised_eligible'] else 'unlabelled: do not train as negative'
            cv2.putText(canvas,flag,(x,y+253),cv2.FONT_HERSHEY_SIMPLEX,.42,(65,90,50) if row['supervised_eligible'] else (45,60,190),1)
    out=root/'reports';out.mkdir(exist_ok=True)
    (out/'gt-smoke-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    assert cv2.imwrite(str(out/'gt-smoke-bev.png'),canvas)
    print(json.dumps({k:v for k,v in summary.items() if k!='frames_detail'},indent=2))

if __name__=='__main__':
    main()
