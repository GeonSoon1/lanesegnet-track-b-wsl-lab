"""Durable visual evidence for controlled reference and map experiments."""
import numpy as np,cv2
from .common import STORE,read_json,write_json,now
from .dataset import load_frames
from .diagnostics import load_gt,project_segments,draw_lanes
from .reference_experiment import OUT,state

def main():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 reference=read_json(OUT/'reference-results.json');frames={r['key']:r for r in load_frames()}
 for i,case in enumerate(reference['cases']):
  key=case['key'];row=frames[key];folder=OUT/'cases'/key;folder.mkdir(parents=True,exist_ok=True);gt=load_gt(row)
  with np.load(STORE/'cache/rgb'/(key+'.npz')) as d:rgb=d['rgb'];mats=d['matrices']
  for mode in ['camera','fusion']:
   legacy_path=STORE/'reports/diagnostics-track-b-v1/predictions'/('camera' if mode=='camera' else 'fusion_normal')/(key+'.npz')
   with np.load(legacy_path) as a:legacy=a['lanes'][a['scores']>=.5]
   with np.load(OUT/'predictions'/f'{mode}_aligned'/(key+'.npz')) as a:aligned=a['lanes'][a['scores']>=.5]
   fig,axes=plt.subplots(1,2,figsize=(14,5.2))
   for ax,lanes,label,color in zip(axes,[legacy,aligned],['legacy','aligned'],['#b74f4a','#2879c8']):
    draw_lanes(ax,gt,'#25865a','SHP GT');draw_lanes(ax,lanes,color,label);m=case['metrics'][mode+'_'+label]
    ax.set(title=f'{label} / TP{m["tp"]} FP{m["fp"]} FN{m["fn"]}',xlim=(-51.2,51.2),ylim=(-25.6,25.6),xlabel='Ego X (m)',ylabel='Ego Y (m)');ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
   fig.suptitle(f'{key} / {mode} / fixed best weights, image, GT, score0.5; only final reference differs');fig.tight_layout();fig.savefig(folder/f'{mode}-reference.png',dpi=140);plt.close(fig)
   pictures=[]
   for cam,(image,P) in enumerate(zip(rgb,mats)):
    view=image[:,:,::-1].copy()
    for lanes,color in [(gt,(90,134,37)),(legacy,(74,79,183)),(aligned,(200,121,40))]:
     for part,x1,y1,x2,y2 in project_segments(lanes,P):
      if part==0:cv2.line(view,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)
    cv2.rectangle(view,(0,0),(150,27),(25,35,45),-1);cv2.putText(view,f'{key} / CAM{cam+1:02}',(6,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);pictures.append(view)
   cv2.imwrite(str(folder/f'{mode}-cameras.jpg'),np.vstack([np.hstack(pictures[:3]),np.hstack(pictures[3:])]),[cv2.IMWRITE_JPEG_QUALITY,94])
  case['images']={m:{'reference':str((folder/f'{m}-reference.png').relative_to(STORE)),'cameras':str((folder/f'{m}-cameras.jpg').relative_to(STORE))} for m in ['camera','fusion']}
  write_json(folder/'case.json',case);state('followup_render',i+1,47,frame=key)
 reference['cases']=reference['cases'];write_json(OUT/'reference-results.json',reference)
 maps=read_json(OUT/'map-results.json');support=read_json(OUT/'map-supported-results.json')
 result={'status':'completed','completed_at':now(),'reference':reference,'map_mean':maps,'map_supported':support,'gt_audit':{k:v for k,v in read_json(OUT/'gt-contract-audit.json').items() if k!='rows'},'folder':str(OUT.relative_to(STORE))}
 write_json(OUT/'summary.json',result);write_json(STORE/'runs/followup-status.json',{'status':'completed','phase':'done','completed':47,'total':47,'updated_at':now(),'output':str(OUT.relative_to(STORE))});print('FOLLOWUP_VISUALS_COMPLETE',flush=True)
if __name__=='__main__':main()
