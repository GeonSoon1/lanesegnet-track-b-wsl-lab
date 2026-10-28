"""Conservative GT-free association and confidence-weighted straight-corridor fusion."""
import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree

def length(line):return float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum())
def direction(line):
 d=line[-1,:2]-line[0,:2];return d/max(np.linalg.norm(d),1e-9)
def sample_line(line,n=120):
 t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))];valid=np.r_[True,np.diff(t)>1e-9]
 if t[-1]<1e-8:return np.repeat(line[:1],n,axis=0)
 q=(np.arange(n)+.5)/n*t[-1]
 return np.column_stack([np.interp(q,t[valid],line[valid,k]) for k in range(3)])
def nearest_distance(points,lines):
 if not len(lines):return np.full(len(points),np.inf)
 a=np.concatenate([l[:-1] for l in lines]);v=np.concatenate([np.diff(l,axis=0) for l in lines]);delta=points[:,None]-a[None];t=np.clip((delta*v).sum(-1)/np.maximum((v*v).sum(-1),1e-12),0,1)
 return np.linalg.norm(delta-t[...,None]*v,axis=-1).min(-1)

def pair_stats(a,b,p,allow_same_frame=False):
 result={'compatible':False,'reason':''}
 if not allow_same_frame and a['frame']==b['frame']:return dict(result,reason='same_frame')
 ca=np.asarray(a['lanes'])[0];cb=np.asarray(b['lanes'])[0];axis=direction(ca);cos=float(axis@direction(cb));result['cosine']=cos
 if cos<np.cos(np.deg2rad(p['max_angle_deg'])):return dict(result,reason='direction_or_angle')
 for field in ['left_type','right_type']:
  if a.get(field,-1)>0 and b.get(field,-1)>0 and a[field]!=b[field]:return dict(result,reason='known_type_conflict')
 origin=ca[0,:2];ta=(ca[:,:2]-origin)@axis;tb=(cb[:,:2]-origin)@axis
 if np.any(np.diff(ta)<-1e-4) or np.any(np.diff(tb)<-1e-4):return dict(result,reason='nonmonotonic_curve')
 lo=max(ta.min(),tb.min());hi=min(ta.max(),tb.max());overlap=max(0,float(hi-lo));result['overlap_m']=overlap
 if min(length(ca),length(cb))<p['min_length_m'] or overlap<p['min_overlap_m'] or overlap/max(1e-9,min(ta.ptp(),tb.ptp()))<p['min_overlap_ratio']:return dict(result,reason='insufficient_overlap')
 q=np.linspace(lo,hi,max(6,int(np.ceil(overlap))+1));pa=np.column_stack([np.interp(q,ta,ca[:,k]) for k in range(3)]);pb=np.column_stack([np.interp(q,tb,cb[:,k]) for k in range(3)])
 distances=np.linalg.norm(pa[:,:2]-pb[:,:2],axis=1);result.update(median_distance_m=float(np.median(distances)),p95_distance_m=float(np.quantile(distances,.95)),median_z_m=float(np.median(abs(pa[:,2]-pb[:,2]))))
 if result['median_distance_m']>p['median_distance_m'] or result['p95_distance_m']>p['p95_distance_m'] or result['median_z_m']>p['max_z_m']:return dict(result,reason='distance_or_height')
 return dict(result,compatible=True,reason='compatible')

def stitch(observations,p):
 if not observations:return [],[]
 lines=[LineString(np.asarray(o['lanes'])[0,:,:2]) for o in observations];tree=STRtree(lines);candidates=[];ledger=[];stats={}
 for i,line in enumerate(lines):
  for j in tree.query(line.buffer(p['p95_distance_m'])):
   j=int(j)
   if j<=i:continue
   test=pair_stats(observations[i],observations[j],p);stats[i,j]=test
   item={'a':observations[i]['id'],'b':observations[j]['id'],**test};ledger.append(item)
   if test['compatible']:candidates.append((test['median_distance_m'],i,j,item))
 groups={i:{i} for i in range(len(observations))};owner=list(range(len(observations)))
 for _,i,j,item in sorted(candidates,key=lambda q:(q[0],q[1],q[2])):
  ga,gb=owner[i],owner[j]
  if ga==gb:item['action']='already_connected';continue
  rejected=None
  for a in groups[ga]:
   for b in groups[gb]:
    if observations[a]['frame']==observations[b]['frame']:rejected='cluster_same_frame';break
    key=tuple(sorted([a,b]));test=stats.get(key)
    if test is None:test=pair_stats(observations[a],observations[b],p)
    if test['reason'] not in ['insufficient_overlap'] and not test['compatible']:rejected='cluster_pair_conflict';break
   if rejected:break
  if rejected:item['action']=rejected;continue
  groups[ga]|=groups.pop(gb)
  for index in groups[ga]:owner[index]=ga
  item['action']='merged'
 merged=[]
 for indices in groups.values():
  members=[observations[i] for i in sorted(indices)];anchor=max(members,key=lambda o:length(np.asarray(o['lanes'])[0]));origin=np.asarray(anchor['lanes'])[0,0,:2];axis=direction(np.asarray(anchor['lanes'])[0]);curves=[]
  for o in members:
   lane=np.asarray(o['lanes']);t=(lane[0,:,:2]-origin)@axis;curves.append((t,lane,float(o['score'])))
  lo=min(t[0] for t,l,w in curves);hi=max(t[-1] for t,l,w in curves);grid=np.unique(np.r_[lo,np.arange(np.ceil(lo/p['station_step_m'])*p['station_step_m'],hi,p['station_step_m']),hi]);points=[];support=[];parts=[]
  for station in grid:
   active=[(t,l,w) for t,l,w in curves if t[0]-1e-6<=station<=t[-1]+1e-6]
   if not active:
    if len(points)>1:parts.append((points,support))
    points=[];support=[];continue
   values=[];weights=[]
   for t,l,w in active:
    values.append(np.array([[np.interp(station,t,l[k,:,xyz]) for xyz in range(3)] for k in range(3)]));weights.append(w)
   points.append(np.average(values,axis=0,weights=weights));support.append(len(active))
  if len(points)>1:parts.append((points,support))
  for points,support in parts:
   lanes=np.asarray(points).transpose(1,0,2)
   def vote(field):
    votes={}
    for o in members:
     value=o.get(field,-1);votes[value]=votes.get(value,0)+o['score']
    return max(votes,key=votes.get)
   merged.append({'id':f'merged_{len(merged):03}','frame':f'merged_{len(merged):03}','lanes':lanes,'score':float(np.average([o['score'] for o in members])),'left_type':vote('left_type'),'right_type':vote('right_type'),'source_ids':[o['id'] for o in members],'support_per_vertex':support,'source_count':len(members)})
 return merged,ledger

def duplicate_pairs(observations,p):
 return sum(pair_stats(a,b,p,True)['compatible'] for i,a in enumerate(observations) for b in observations[i+1:])

def geometry_quality(lines,truth,tolerance=1.):
 def one(source,target):
  total=supported=0.;distances=[];weights=[]
  for line in source:
   n=length(line);points=sample_line(line);d=nearest_distance(points,target);total+=n;supported+=float((d<=tolerance).mean())*n;distances.extend(d.tolist());weights.extend([n/len(d)]*len(d))
  if not distances or not target:return {'length_m':total,'supported_fraction':0 if total else None,'mean_distance_m':None,'p95_distance_m':None}
  d=np.asarray(distances);w=np.asarray(weights);order=np.argsort(d);q=np.cumsum(w[order])/w.sum()
  return {'length_m':total,'supported_fraction':supported/total if total else None,'mean_distance_m':float(np.average(d,weights=w)),'p95_distance_m':float(np.interp(.95,q,d[order]))}
 return {'gt_to_prediction':one(truth,lines),'prediction_to_gt':one(lines,truth)}


def split_by_support(groups,minimum=2):
    confirmed=[];uncertain=[]
    for group in groups:
        mask=np.asarray(group['support_per_vertex'])>=minimum
        if len(mask)<2:continue
        edges=mask[:-1]&mask[1:];start=0
        for stop in range(1,len(edges)+1):
            if stop<len(edges) and edges[stop]==edges[start]:continue
            item=dict(group);item['id']=group['id']+('_confirmed_' if edges[start] else '_uncertain_')+str(start)
            item['frame']=item['id'];item['lanes']=group['lanes'][:,start:stop+1].copy();item['support_per_vertex']=group['support_per_vertex'][start:stop+1]
            item['disposition']='confirmed' if edges[start] else 'uncertain'
            (confirmed if edges[start] else uncertain).append(item);start=stop
    return confirmed,uncertain
