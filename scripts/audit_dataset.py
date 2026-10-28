#!/usr/bin/env python3
"""Read-only MMS/SHP audit; standard library only. Writes one JSON report.
XY proximity is coverage, NOT training eligibility. Source files are never changed.
"""
import argparse, collections, csv, hashlib, json, math, re, statistics, struct
from pathlib import Path
from datetime import datetime, timezone

def fingerprint(path):
    return {"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def dbf_audit(path):
    data=path.read_bytes()
    count=struct.unpack_from("<I",data,4)[0]
    hlen,rlen=struct.unpack_from("<HH",data,8)
    fields=[]; offset=1
    for pos in range(32,hlen-1,32):
        h=data[pos:pos+32]
        if h[0]==13: break
        name=h[:11].split(b"\0")[0].decode("ascii")
        fields.append((name,offset,h[16])); offset+=h[16]
    rows=[]
    for i in range(count):
        row=data[hlen+i*rlen:hlen+(i+1)*rlen]
        if row[:1]!=b"*":
            rows.append({name:row[start:start+size].strip() for name,start,size in fields})
    encodings={}
    for enc in ("utf-8","cp949"):
        errors=collections.Counter()
        for row in rows:
            for name,value in row.items():
                try: value.decode(enc)
                except UnicodeDecodeError: errors[name]+=1
        encodings[enc]={"undecodable_values":sum(errors.values()),"errors_by_field":dict(errors)}
    codes={}
    for name in ("LineType","LineKind","Direction","LinkType"):
        if rows and name in rows[0]:
            codes[name]=dict(collections.Counter(r[name].decode("ascii") for r in rows))
    return {"records":len(rows),"fields":[x[0] for x in fields],"encoding_checks":encodings,"codes":codes}

def shp_audit(path,index=False):
    data=path.read_bytes(); pos=100; segments=[]; grid=collections.defaultdict(list); records=0
    bbox=struct.unpack_from("<dddddddd",data,36)
    info={"shape_type":struct.unpack_from("<i",data,32)[0],
          "bbox":dict(zip(("xmin","ymin","xmax","ymax","zmin","zmax","mmin","mmax"),bbox))}
    while pos+8<=len(data):
        _,length=struct.unpack_from(">ii",data,pos); pos+=8
        body=memoryview(data)[pos:pos+length*2]; pos+=length*2
        if struct.unpack_from("<i",body)[0] not in (3,13,23): continue
        records+=1
        if not index: continue
        nparts,npts=struct.unpack_from("<ii",body,36)
        parts=list(struct.unpack_from("<"+"i"*nparts,body,44))+[npts]; xyoff=44+nparts*4
        points=[struct.unpack_from("<dd",body,xyoff+i*16) for i in range(npts)]
        for st,en in zip(parts,parts[1:]):
            for a,b in zip(points[st:en-1],points[st+1:en]):
                sid=len(segments); segments.append((*a,*b))
                for ix in range(math.floor(min(a[0],b[0])/50),math.floor(max(a[0],b[0])/50)+1):
                    for iy in range(math.floor(min(a[1],b[1])/50),math.floor(max(a[1],b[1])/50)+1):
                        grid[ix,iy].append(sid)
    info.update(records=records,indexed_line_segments=len(segments))
    return info,segments,grid

def nearest_within_57m(x,y,segments,grid):
    # All cells intersecting the 57m square are searched; distance uses exact segments.
    ids=set()
    for ix in range(math.floor((x-57)/50),math.floor((x+57)/50)+1):
        for iy in range(math.floor((y-57)/50),math.floor((y+57)/50)+1):
            ids.update(grid.get((ix,iy),()))
    best=57.0**2; found=False
    for sid in ids:
        ax,ay,bx,by=segments[sid]; vx,vy=bx-ax,by-ay; vv=vx*vx+vy*vy
        t=max(0.0,min(1.0,((x-ax)*vx+(y-ay)*vy)/vv)) if vv else 0.0
        d=(x-ax-t*vx)**2+(y-ay-t*vy)**2
        if d<=best: best,found=d,True
    return math.sqrt(best) if found else None

def read_eo(path):
    if not path.is_file(): return []
    with path.open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.reader(f,delimiter=";"))
    for row in rows:
        if len(row)!=17 or not all(math.isfinite(float(v)) for v in row[1:]):
            raise ValueError("Unexpected EO schema: "+str(path))
    return rows

def las_audit(folder):
    tracks={}
    for path in sorted(folder.glob("*.las")):
        match=re.match(r"Track_([A-Z]+)_",path.name); track=match.group(1) if match else "other"
        item=tracks.setdefault(track,{"files":0,"bytes":0,"points":0,"versions":set(),
                                     "point_formats":set(),"global_encoding_flags":set(),"sample_times":[]})
        with path.open("rb") as f:
            h=f.read(375)
            if h[:4]!=b"LASF": raise ValueError("Invalid LAS signature: "+str(path))
            minor=h[25]; po=struct.unpack_from("<I",h,96)[0]; fmt=h[104]&63
            plen=struct.unpack_from("<H",h,105)[0]; count=struct.unpack_from("<I",h,107)[0]
            if minor>=4: count=struct.unpack_from("<Q",h,247)[0] or count
            gps_off=22 if fmt>=6 else 20 if fmt in (1,3,4,5) else None
            if count and gps_off is not None:
                for i in range(17):
                    f.seek(po+(i*(count-1)//16)*plen); point=f.read(plen)
                    item["sample_times"].append(struct.unpack_from("<d",point,gps_off)[0])
        item["files"]+=1; item["bytes"]+=path.stat().st_size; item["points"]+=count
        item["versions"].add(f"{h[24]}.{minor}"); item["point_formats"].add(fmt)
        item["global_encoding_flags"].add(struct.unpack_from("<H",h,6)[0])
    for item in tracks.values():
        times=item.pop("sample_times")
        item["gps_time_sample_count"]=len(times)
        item["sampled_gps_time_range"]=[min(times),max(times)] if times else None
        for key in ("versions","point_formats","global_encoding_flags"): item[key]=sorted(item[key])
    return {"tracks":tracks,"total_files":sum(t["files"] for t in tracks.values()),
            "total_bytes":sum(t["bytes"] for t in tracks.values()),
            "total_points":sum(t["points"] for t in tracks.values()),
            "time_note":"17 evenly spaced point records per LAS. Sampled ranges are not full time bounds; spatial chunks are not assumed time-sorted."}

def audit(raw,shp):
    rm_info,segments,grid=shp_audit(shp/"RM1_LANELINE.shp",True)
    report={"schema_version":1,"generated_at_utc":datetime.now(timezone.utc).isoformat(),
            "raw_root":str(raw),"shp_root":str(shp),
            "method":{"source_access":"read-only","dependencies":"Python standard library",
                      "coverage_origin":"XY average of CAM1 through CAM6 for each EO row",
                      "coverage_geometry":"Exact XY point-to-RM1-segment distance within 57m using 50m cells; Z ignored",
                      "coverage_is_training_eligibility":False,
                      "limits":["Proximity does not prove valid paired lane corridors or usable GT.",
                                "Timestamp equality does not verify clock accuracy, calibration or drift.",
                                "Image existence does not verify that every image decodes correctly.",
                                "No complete LAS scan, image projection, LiDAR alignment or model run is performed."]},
            "shapefiles":{},"tracks":[]}
    for name in ("RM1_LANELINE","NT2_LINK"):
        info=rm_info if name=="RM1_LANELINE" else shp_audit(shp/(name+".shp"))[0]
        info["dbf"]=dbf_audit(shp/(name+".dbf"))
        info["sidecars"]={ext:fingerprint(shp/(name+ext)) for ext in (".shp",".shx",".dbf",".prj",".cpg")}
        info["cpg_label"]=(shp/(name+".cpg")).read_text(encoding="ascii").strip()
        info["prj_wkt"]=(shp/(name+".prj")).read_text(encoding="ascii").strip()
        report["shapefiles"][name]=info
    raw_prj=raw/"CoordinateSystem.prj"
    report["raw_coordinate_system_wkt"]=raw_prj.read_text(encoding="ascii").strip() if raw_prj.is_file() else None
    names=sorted({m.group(1) for p in raw.iterdir() if p.is_dir()
                  for m in [re.fullmatch(r"Track_([A-Z]+)(?:\d\d|Sphere)",p.name)] if m})
    cameras=("01","02","03","04","05","06")
    for track in names:
        sensors={}; allrows={}
        for sensor in ("Sphere",*cameras):
            folder=raw/("Track_"+track+sensor); path=folder/"External Orientation.csv"
            rows=read_eo(path); allrows[sensor]=rows
            sensors[sensor]={"eo_exists":path.is_file(),"eo_rows":len(rows),
                             "jpg_files":len(list(folder.glob("*.jpg"))),
                             "eo_referenced_image_missing":sum(not (folder/r[0]).is_file() for r in rows),
                             "internal_orientation_exists":(folder/"Internal Orientation.txt").is_file()}
            if path.is_file(): sensors[sensor]["eo_fingerprint"]=fingerprint(path)
        counts=[len(rows) for rows in allrows.values()]; aligned=bool(counts[0]) and len(set(counts))==1
        entry={"track":track,"sensors":sensors,"equal_nonzero_eo_row_counts":aligned}
        if aligned:
            sphere=allrows["Sphere"]
            entry["max_camera_vs_sphere_time_delta_seconds"]=max(abs(float(allrows[s][i][1])-float(sphere[i][1]))
                                                                 for s in cameras for i in range(len(sphere)))
            coords=[[sum(float(allrows[s][i][a]) for s in cameras)/6 for a in (2,3,4)] for i in range(len(sphere))]
            entry["camera_mean_xyz_range"]=[[min(p[k] for p in coords),max(p[k] for p in coords)] for k in range(3)]
            distances=[nearest_within_57m(p[0],p[1],segments,grid) for p in coords]
            near=[d for d in distances if d is not None]
            entry["RM1_XY_proximity"]={"total_frames":len(coords),"within_10m":sum(d<=10 for d in near),
                                       "within_57m":len(near),"outside_57m":len(coords)-len(near),
                                       "median_distance_among_within_57m_m":statistics.median(near) if near else None,
                                       "is_training_eligibility":False}
        report["tracks"].append(entry)
    report["las"]=las_audit(raw/"Scanfiles")
    calib=raw/"Calibration"
    report["calibration"]={"ort_files":len(list(calib.glob("*.ort"))),
                           "orttrj_files":len(list(calib.glob("*.orttrj"))),
                           "calib_files":len(list(calib.glob("*.calib"))),
                           "certificate_pdf_files":len(list(calib.glob("*.pdf")))}
    report["recommendation"]={"first_track":"B",
                              "reason":"Complete six-camera EO/image references and broad SHP proximity; confirm valid corridors in a separate GT smoke run.",
                              "first_gt_encoding":"cp949",
                              "encoding_note":"Validate actual DBF bytes; .cpg can disagree with non-ASCII fields.",
                              "spatial_split_note":"B/C and F/H show overlapping routes; different track names do not establish independent evaluation."}
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root",required=True,type=Path)
    parser.add_argument("--shp-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(); result=audit(args.raw_root,args.shp_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"tracks":[{"track":t["track"],"coverage":t.get("RM1_XY_proximity")}
                                                      for t in result["tracks"]],
                      "las_files":result["las"]["total_files"]},ensure_ascii=True))

if __name__=="__main__": main()
