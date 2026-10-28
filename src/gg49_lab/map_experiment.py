"""Two fixed validation windows: GT-free association, map export and independent GT audit."""
import json,math,traceback
from collections import Counter
from pathlib import Path
import numpy as np,shapefile,fiona
from shapely.geometry import LineString,Polygon,mapping
from shapely.ops import unary_union
from pyproj import Transformer
from .common import STORE,ROOT,SHP,read_json,write_json,now
from .dataset import load_frames
from .stitching import stitch,length,direction,sample_line,nearest_distance,geometry_quality,duplicate_pairs,pair_stats
from .reference_experiment import OUT,state


def raw_links():
 path=SHP/'NT2_LINK.shp';reader=shapefile.Reader(str(path),encoding='cp949');result={}
 for row in reader.iterShapeRecords():
  record=row.record.as_dict();points=np.c_[np.asarray(row.shape.points),np.asarray(row.shape.z)]
  result[record['ID']]={'xyz':points,'attributes':record}
 return result

def gt_reference(rows,links):
 polygons=[];ids=set();probes={};audit=[]
 for row in rows:
  R=np.array(row['rotation']);origin=np.array(row['origin']);corners=np.array([[-51.2,-25.6,0],[51.2,-25.6,0],[51.2,25.6,0],[-51.2,25.6,0]])@R.T+origin;polygons.append(Polygon(corners[:,:2]))
  with np.load(row['gt_path']) as d:
   lanes=d['lanes']@R.T+origin;source=d['source_link_ids'].tolist();ids.update(source)
   for i,id in enumerate(source):
    assert id in links,id;distance=nearest_distance(lanes[i,0],[links[id]['xyz']]);audit.append({'frame':row['key'],'link':id,'mean_source_distance_m':float(distance.mean()),'max_source_distance_m':float(distance.max()),'gt_length_m':length(lanes[i,0]),'original_link_length_m':length(links[id]['xyz'])})
   for i,j in zip(*np.where(d['adjacency']>0)):probes[source[i]+'->'+source[j]]=(lanes[i,0,-1]+lanes[j,0,0])/2
 footprint=unary_union(polygons);truth=[]
 for id in sorted(ids):
  clipped=LineString(links[id]['xyz']).intersection(footprint)
  pieces=[clipped] if clipped.geom_type=='LineString' else list(clipped.geoms) if hasattr(clipped,'geoms') else []
  for piece in pieces:
   if piece.geom_type=='LineString' and len(piece.coords)>1 and piece.length>.1:truth.append({'id':id,'xyz':np.asarray(piece.coords)})
 return truth,probes,audit

def observations(rows,mode):
 result=[]
 for row in rows:
  R=np.array(row['rotation']);origin=np.array(row['origin'])
  with np.load(OUT/'predictions'/f'{mode}_aligned'/(row['key']+'.npz')) as d:
   for rank in np.flatnonzero(d['scores']>=.5):
    result.append({'id':f'{mode}:{row["key"]}:rank{rank}','frame':row['key'],'timestamp':row['timestamp'],'rank':int(rank),'lanes':d['lanes'][rank]@R.T+origin,'score':float(d['scores'][rank]),'left_type':int(d['left_types'][rank]),'right_type':int(d['right_types'][rank])})
 return result

def nearest_label(point,truth):
 best=None
 for r in truth:
  xyz=r['xyz'];a=xyz[:-1];v=np.diff(xyz,axis=0);t=np.clip(((point-a)*v).sum(-1)/np.maximum((v*v).sum(-1),1e-12),0,1);q=a+t[:,None]*v;dist=np.linalg.norm(q-point,axis=1);i=int(dist.argmin())
  if best is None or dist[i]<best[0]:best=(float(dist[i]),r['id'],q[i])
 return best

def merge_audit(raw,merged,truth,p):
 byid={r['id']:r for r in raw};assessed=unknown=suspect=0;examples=[];guard=0
 for group in merged:
  members=[byid[id] for id in group['source_ids']]
  for i,a in enumerate(members):
   for b in members[i+1:]:
    test=pair_stats(a,b,p)
    if test['reason'] not in ['insufficient_overlap'] and not test['compatible']:guard+=1
    if not test['compatible']:continue
    axis=direction(a['lanes'][0]);origin=a['lanes'][0,0,:2];ta=(a['lanes'][0,:,:2]-origin)@axis;tb=(b['lanes'][0,:,:2]-origin)@axis;station=(max(ta.min(),tb.min())+min(ta.max(),tb.max()))/2
    pa=np.array([np.interp(station,ta,a['lanes'][0,:,k]) for k in range(3)]);pb=np.array([np.interp(station,tb,b['lanes'][0,:,k]) for k in range(3)])
    la=nearest_label(pa,truth);lb=nearest_label(pb,truth)
    if la[0]>1 or lb[0]>1:unknown+=1;continue
    assessed+=1
    if la[1]!=lb[1] and np.linalg.norm(la[2]-lb[2])>1.5:
     suspect+=1;examples.append({'a':a['id'],'b':b['id'],'gt_a':la[1],'gt_b':lb[1]})
 return {'guard_violations':guard,'gt_assessed_overlap_pairs':assessed,'gt_unassessed_overlap_pairs':unknown,'gt_suspected_cross_lane_pairs':suspect,'examples':examples,'limitation':'GT-assisted midpoint proxy only; does not certify all semantic/topological merges.'}

def export(folder,raw,merged,truth):
 gpkg=folder/'map.gpkg';schema={'geometry':'3D LineString','properties':{'id':'str','score':'float','sources':'int','left_type':'int','right_type':'int'}}
 for layer,items,part in [('observations',raw,0),('merged_centers',merged,0),('merged_left',merged,1),('merged_right',merged,2)]:
  with fiona.open(gpkg,'w',driver='GPKG',layer=layer,crs='EPSG:32652',schema=schema) as sink:
   for r in items:sink.write({'geometry':{'type':'LineString','coordinates':r['lanes'][part].tolist()},'properties':{'id':r['id'],'score':r['score'],'sources':r.get('source_count',1),'left_type':r.get('left_type',-1),'right_type':r.get('right_type',-1)}})
 with fiona.open(gpkg,'w',driver='GPKG',layer='gt_reference',crs='EPSG:32652',schema={'geometry':'3D LineString','properties':{'id':'str'}}) as sink:
  for r in truth:sink.write({'geometry':{'type':'LineString','coordinates':r['xyz'].tolist()},'properties':{'id':r['id']}})
 assert set(fiona.listlayers(gpkg))=={'observations','merged_centers','merged_left','merged_right','gt_reference'}
 with fiona.open(gpkg,layer='merged_centers') as src:assert len(src)==len(merged) and src.crs.to_epsg()==32652
 transformer=Transformer.from_crs(32652,4326,always_xy=True);features=[]
 for r in merged:
  lon,lat=transformer.transform(r['lanes'][0,:,0],r['lanes'][0,:,1]);features.append({'type':'Feature','geometry':{'type':'LineString','coordinates':np.column_stack([lon,lat]).tolist()},'properties':{'id':r['id'],'source_ids':r['source_ids'],'score':r['score']}})
 write_json(folder/'merged-wgs84.geojson',{'type':'FeatureCollection','features':features})


def figure(folder,raw,merged,truth,title):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 points=np.concatenate([r['lanes'][0] for r in raw]+[r['xyz'] for r in truth]);origin=points[:,:2].mean(0);lo=points[:,:2].min(0)-5;hi=points[:,:2].max(0)+5
 fig,axes=plt.subplots(2,1,figsize=(15,5))
 for ax,items,name in zip(axes,[raw,merged],['Before: multi-frame observations','After: associated and fused']):
  for r in truth:ax.plot(r['xyz'][:,0]-origin[0],r['xyz'][:,1]-origin[1],color='#4d6357',linestyle='--',lw=1.1,alpha=.6)
  for i,r in enumerate(items):
   lane=r['lanes'][0];ax.plot(lane[:,0]-origin[0],lane[:,1]-origin[1],color=plt.cm.tab10(i%10),lw=.9 if name.startswith('Before') else 2.2,alpha=.5 if name.startswith('Before') else .95)
  ax.set_title(f'{name}\n{len(items)} lane observations/outputs');ax.set_xlim(lo[0]-origin[0],hi[0]-origin[0]);ax.set_ylim(lo[1]-origin[1],hi[1]-origin[1]);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('Easting offset (m)');ax.set_ylabel('Northing offset (m)')
 fig.suptitle(title+f' | EPSG:32652 | origin=({origin[0]:.2f}, {origin[1]:.2f})\nDashed green: SHP for evaluation only; matching uses predictions only',fontsize=11);fig.tight_layout();fig.savefig(folder/'before-after.png',dpi=160,bbox_inches='tight');plt.close(fig)
 # Detailed centerline spread of overlapping observations in the same first cluster.
 if merged:
  group=max(merged,key=lambda r:r['source_count']);members=[r for r in raw if r['id'] in group['source_ids']];center=group['lanes'][0];axis=direction(center);normal=np.array([-axis[1],axis[0]]);origin2=center[0,:2]
  fig,ax=plt.subplots(figsize=(12,3.8))
  for r in members:
   xy=r['lanes'][0,:,:2]-origin2;ax.plot(xy@axis,xy@normal,alpha=.3,lw=.9,color='#467fb0')
  xy=center[:,:2]-origin2;ax.plot(xy@axis,xy@normal,color='#d58224',lw=2,label='Confidence-weighted fusion')
  ax.set(xlabel='Along-lane station (m)',ylabel='Lateral position (m)',title='Largest group: multi-view spread and fused line');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(folder/'spread.png',dpi=150);plt.close(fig)

def main():
 protocol=read_json(OUT/'protocol.json');accept=read_json(OUT/'map-acceptance.json');frames={r['key']:r for r in load_frames()};links=raw_links();results=[];all_gt_audits=[]
 for window,keys in protocol['map_windows'].items():
  rows=[frames[key] for key in keys];assert len(rows)==20 and all(r['split']=='val' for r in rows);truth,probes,gt_audit=gt_reference(rows,links);all_gt_audits+=gt_audit
  for mode in ['camera','fusion']:
   state('map_stitching',len(results),4,window=window,mode=mode);folder=OUT/'maps'/f'{window}_{mode}';folder.mkdir(parents=True,exist_ok=True);raw=observations(rows,mode)
   merged,ledger=stitch(raw,protocol['association']);rawlines=[r['lanes'][0] for r in raw];newlines=[r['lanes'][0] for r in merged];gtlines=[r['xyz'] for r in truth]
   before=geometry_quality(rawlines,gtlines);after=geometry_quality(newlines,gtlines);audit=merge_audit(raw,merged,truth,protocol['association'])
   source_ids=set(id for r in merged for id in r['source_ids']);retained=source_ids=={r['id'] for r in raw};assert retained and audit['guard_violations']==0
   bdup=duplicate_pairs(raw,protocol['association']);adup=duplicate_pairs(merged,protocol['association'])
   junction={name:int((nearest_distance(np.array(list(probes.values())),lines)<=1).sum()) if probes else 0 for name,lines in [('before',rawlines),('after',newlines)]}
   result={'window':window,'mode':mode,'keys':keys,'observations':len(raw),'outputs':len(merged),'duplicate_compatible_pairs_before':bdup,'duplicate_compatible_pairs_after':adup,'before':before,'after':after,'audit':audit,'all_source_ids_retained':retained,'gt_link_count':len(set(r['id'] for r in truth)),'gt_connection_probes':len(probes),'connection_probe_support':junction,'connection_scope':'Geometry coverage at known directed-GT junction probes, not topology edge accuracy. No new network connectivity is claimed.','groups':[{'id':r['id'],'source_count':r['source_count'],'source_ids':r['source_ids'],'mean_support':float(np.mean(r['support_per_vertex']))} for r in merged]}
   result['acceptance_passed']=bool(adup<bdup and after['gt_to_prediction']['supported_fraction']>=before['gt_to_prediction']['supported_fraction']-accept['gt_supported_fraction_max_drop'] and after['prediction_to_gt']['p95_distance_m']<=before['prediction_to_gt']['p95_distance_m']+accept['prediction_to_gt_p95_max_increase_m'] and retained and audit['guard_violations']==0 and audit['gt_suspected_cross_lane_pairs']==0)
   def serial(r):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()}
   write_json(folder/'observations.json',{'crs':'EPSG:32652','items':[serial(r) for r in raw]});write_json(folder/'merged.json',{'crs':'EPSG:32652','items':[serial(r) for r in merged]});write_json(folder/'association-ledger.json',ledger);write_json(folder/'metrics.json',result)
   export(folder,raw,merged,truth);figure(folder,raw,merged,truth,f'Window{window}: {keys[0]}..{keys[-1]} / {mode} aligned-reference / score>=0.5')
   result['image']=str((folder/'before-after.png').relative_to(STORE));result['spread_image']=str((folder/'spread.png').relative_to(STORE));result['folder']=str(folder.relative_to(STORE));write_json(folder/'metrics.json',result);results.append(result)
 write_json(OUT/'gt-contract-audit.json',{'source':'Original NT2_LINK.shp matched by source_link_ids; clip geometry using union of fixed-window BEV footprints. GT never used by stitch().','rows':all_gt_audits,'max_gt_to_original_link_distance_m':max(r['max_source_distance_m'] for r in all_gt_audits),'adjacency_scope':'NPZ adjacency retained as directed junction geometry probes; no global topology claim'})
 write_json(OUT/'map-results.json',{'status':'completed','completed_at':now(),'protocol':protocol,'acceptance':accept,'runs':results})
 state('map_completed',4,4);print('MAP_EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write_json(STORE/'runs/followup-status.json',{'status':'failed','phase':'map','error':str(e),'traceback':traceback.format_exc(),'updated_at':now()});raise
