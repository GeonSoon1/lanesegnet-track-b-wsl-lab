import cv2,json,numpy as np
from .common import STORE,write_json

def main():
    rows=json.loads((STORE/'artifacts/prepared/DATASET.json').read_text())['frames'];out=STORE/'reports/lidar-qa';out.mkdir(exist_ok=True);reports=[]
    for key in ['B_0095','B_0250']:
        row=next(r for r in rows if r['key']==key)
        with np.load(STORE/'cache/rgb'/(key+'.npz')) as d:rgb=d['rgb'];matrices=d['matrices']
        with np.load(STORE/'cache/lidar'/(key+'.npz')) as d:bev=d['bev'].astype(np.float32)
        iy,ix=np.nonzero(bev[4]>0);pts=np.c_[-51.2+(ix+.5)*.512,-25.6+(iy+.5)*.512,bev[2,iy,ix]*6,np.ones(len(ix))]
        counts=[]
        for P in matrices:
            a=pts@P.T;u=a[:,:2]/np.maximum(a[:,2:3],.001);counts.append(int(((a[:,2]>.1)&(u[:,0]>=0)&(u[:,0]<384)&(u[:,1]>=0)&(u[:,1]<384)).sum()))
        cam=int(np.argmax(counts));P=matrices[cam]
        proj=pts@P.T;good=proj[:,2]>.1;uv=np.rint(proj[:,:2]/np.maximum(proj[:,2:3],.001)).astype(int)
        good&=(uv[:,0]>=0)&(uv[:,0]<384)&(uv[:,1]>=0)&(uv[:,1]<384)
        original=rgb[cam][:,:,::-1].copy();overlay=original.copy();colors=cv2.applyColorMap(np.clip(bev[1,iy,ix]*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO).reshape(-1,3)
        for p,c in zip(uv[good],colors[good]):cv2.circle(overlay,tuple(p),1,tuple(map(int,c)),-1)
        heat=cv2.applyColorMap(np.clip(bev[0]*255,0,255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)
        heat[bev[4]==0]=(245,245,245)
        with np.load(row['gt_path']) as d:
            for lane in d['lanes']:
                for k,col in enumerate([(180,60,180),(0,150,240),(80,190,70)]):
                    q=np.rint((lane[k,:,:2]+[51.2,25.6])/[.512,.512]).astype(np.int32);cv2.polylines(heat,[q],False,col,1)
        heat=cv2.resize(heat,(768,384),interpolation=cv2.INTER_NEAREST)
        top=np.hstack([original,overlay]);canvas=np.vstack([top,heat]);cv2.putText(canvas,key+f' | CAM{cam+1}: RGB / time-local LiDAR cell representatives',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.putText(canvas,'BEV count + SHP: tensor grid X right, Y down; not dense raw-point projection',(10,406),cv2.FONT_HERSHEY_SIMPLEX,.46,(20,30,40),1)
        cv2.imwrite(str(out/(key+'.jpg')),canvas)
        reports.append({'key':key,'observed_cells':len(ix),'selected_camera':cam+1,'projected_cells_by_camera':counts,'projected_selected_cells':int(good.sum()),'half_window_seconds':.1,'reconstruction':'BEV cell center and per-cell minimum height, approximate visualization only'})
    write_json(STORE/'reports/lidar-projection-qa.json',{'examples':reports,'clock_accuracy_not_certified':True,'spatial_projection_is_diagnostic':True})
if __name__=='__main__':main()
