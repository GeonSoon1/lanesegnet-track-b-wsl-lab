import json,traceback
import numpy as np,fiona
from .common import STORE,read_json,write_json,now
from .dataset import load_frames
from .reference_experiment import OUT,state
from .map_experiment import raw_links,gt_reference,export,figure
from .stitching import split_by_support,geometry_quality,duplicate_pairs,nearest_distance

def main():
 old=read_json(OUT/'map-results.json');protocol=old['protocol'];accept=old['acceptance'];frames={r['key']:r for r in load_frames()};links=raw_links();results=[]
 for original in old['runs']:
  window=original['window'];mode=original['mode'];state('support_experiment',len(results),4,window=window,mode=mode)
  source=STORE/original['folder'];raw=read_json(source/'observations.json')['items'];groups=read_json(source/'merged.json')['items']
  for r in raw+groups:r['lanes']=np.asarray(r['lanes'])
  confirmed,uncertain=split_by_support(groups);truth,probes,audit=gt_reference([frames[k] for k in original['keys']],links);gtlines=[r['xyz'] for r in truth];lines=[r['lanes'][0] for r in confirmed]
  folder=OUT/'maps'/f'{window}_{mode}_supported';folder.mkdir(exist_ok=True);quality=geometry_quality(lines,gtlines);all_sources=set(s for r in confirmed+uncertain for s in r['source_ids']);assert all_sources=={r['id'] for r in raw}
  dups=duplicate_pairs(confirmed,protocol['association']);result={**original,'variant':'support>=2','outputs':len(confirmed),'uncertain_outputs':len(uncertain),'after':quality,'duplicate_compatible_pairs_after':dups,'all_source_ids_retained':True,'retention_scope':'All observations retained across confirmed+uncertain layers, not all kept in main map.','connection_probe_support':{'before':original['connection_probe_support']['before'],'after':int((nearest_distance(np.array(list(probes.values())),lines)<=1).sum()) if probes else 0}}
  result['acceptance_passed']=bool(dups<original['duplicate_compatible_pairs_before'] and quality['gt_to_prediction']['supported_fraction']>=original['before']['gt_to_prediction']['supported_fraction']-accept['gt_supported_fraction_max_drop'] and quality['prediction_to_gt']['p95_distance_m'] is not None and quality['prediction_to_gt']['p95_distance_m']<=original['before']['prediction_to_gt']['p95_distance_m']+accept['prediction_to_gt_p95_max_increase_m'])
  def serial(r):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()}
  write_json(folder/'merged.json',{'crs':'EPSG:32652','confirmed':[serial(r) for r in confirmed],'uncertain':[serial(r) for r in uncertain]});write_json(folder/'metrics.json',result)
  export(folder,raw,confirmed,truth)
  with fiona.open(folder/'map.gpkg','w',driver='GPKG',layer='uncertain_centers',crs='EPSG:32652',schema={'geometry':'3D LineString','properties':{'id':'str','sources':'int'}}) as sink:
   for r in uncertain:sink.write({'geometry':{'type':'LineString','coordinates':r['lanes'][0].tolist()},'properties':{'id':r['id'],'sources':r['source_count']}})
  figure(folder,raw,confirmed,truth,f'Window{window} / {mode} / support>=2 confirmed map; uncertain tails retained separately')
  result.update(folder=str(folder.relative_to(STORE)),image=str((folder/'before-after.png').relative_to(STORE)),spread_image=str((folder/'spread.png').relative_to(STORE)))
  write_json(folder/'metrics.json',result);results.append(result)
 # Audit the entire47-frame validation source geometry, independently of stitching.
 _,_,all_audit=gt_reference([frames[k] for k in protocol['frame_keys']],links)
 write_json(OUT/'gt-contract-audit.json',{'source':'Original NT2_LINK.shp matched by source_link_ids. No synthetic GT resegmentation.','frames':47,'rows':all_audit,'max_gt_to_original_link_distance_m':max(r['max_source_distance_m'] for r in all_audit),'missing_source_ids':[]})
 write_json(OUT/'map-supported-results.json',{'status':'completed','completed_at':now(),'support_protocol':read_json(OUT/'support-protocol.json'),'runs':results});state('support_completed',4,4)
if __name__=='__main__':
 try:main()
 except Exception as e:write_json(STORE/'runs/followup-status.json',{'status':'failed','phase':'support','error':str(e),'traceback':traceback.format_exc(),'updated_at':now()});raise
