"""Freeze 100 independent ETH3D image pairs for quantization regression tests."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from scripts.matching_metrics import quaternion_rotation,sampson_distance_px
from scripts.sample_data_io import write_json

ROOT=Path(__file__).resolve().parents[1]
SEED=20261117
# The indoor/outdoor labels follow https://www.eth3d.net/datasets.
SCENES={'exhibition_hall':('indoor',10),'lecture_room':('indoor',10),
        'living_room':('indoor',10),'lounge':('indoor',10),
        'boulders':('outdoor',10),'observatory':('outdoor',20),'terrace_2':('outdoor',10)}


def pixel_hash(path):
    with Image.open(path) as image:
        image=image.convert('RGB')
        return hashlib.sha256(f'{image.width}x{image.height}:RGB:'.encode()+image.tobytes()).hexdigest()


def scene_data(root,scene):
    folder=root/scene
    preparation=json.loads((folder/'preparation.json').read_text())
    assert preparation['split']=='test'
    cameras={}
    for line in (folder/'dslr_calibration_undistorted/cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        p=line.split();assert p[1]=='PINHOLE'
        fx,fy,cx,cy=map(float,p[4:8]);cameras[int(p[0])]=dict(K=[[fx,0,cx],[0,fy,cy],[0,0,1]],width=int(p[2]),height=int(p[3]))
    frames={};tracks={}
    with (folder/'dslr_calibration_undistorted/images.txt').open() as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            p=line.split();name=p[9];point_line=f.readline().split()
            points={int(point_line[i+2]):[float(point_line[i]),float(point_line[i+1])]
                    for i in range(0,len(point_line),3) if int(point_line[i+2])>=0}
            w2c=np.eye(4);w2c[:3,:3]=quaternion_rotation(list(map(float,p[1:5])));w2c[:3,3]=list(map(float,p[5:8]))
            path=folder/'images'/name;assert path.is_file()
            frames[name]={**cameras[int(p[8])], 'camera_to_world':np.linalg.inv(w2c).tolist(),
                'camera_model':'PINHOLE','camera_convention':'OpenCV','scene':scene,
                'environment':SCENES[scene][0],'dataset':'eth3d','source_path':str(path),
                'source_url':preparation['source_url'],'source_split':'ETH3D high-res test images; supplied reference calibration'}
            tracks[name]=points
    return frames,tracks


def choose_pairs(frames,tracks,count,seed):
    candidates=[]
    for a,b in itertools.combinations(sorted(frames),2):
        common=tracks[a].keys()&tracks[b].keys()
        if len(common)<50:continue
        score=.5*(len(common)/len(tracks[a])+len(common)/len(tracks[b]))
        if not .15<=score<=.9:continue
        x=np.asarray(frames[a]['camera_to_world']);y=np.asarray(frames[b]['camera_to_world'])
        if np.linalg.norm(x[:3,3]-y[:3,3])<1e-3:continue
        ids=sorted(common)
        e=sampson_distance_px([tracks[a][i] for i in ids],[tracks[b][i] for i in ids],frames[a],frames[b])
        if np.median(e)>1.5 or np.mean(e<=3)<.9:continue
        candidates.append((a,b,score,len(common),float(np.median(e))))
    if len(candidates)<count:raise ValueError(f'Only {len(candidates)} eligible pairs; need {count}')
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda p:p[2])
    # Fixed quantiles give a range of sparse overlap; no matcher outputs are used.
    chosen=[candidates[i] for i in np.linspace(0,len(candidates)-1,count,dtype=int)]
    random.Random(seed).shuffle(chosen)
    return chosen,len(candidates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,default=ROOT/'data/flexglue_ptq_100')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'data/flexglue_eval_100')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    old_images=json.loads((args.calibration/'images.json').read_text())
    old_files={m['sha256'] for m in old_images.values()}
    with ThreadPoolExecutor(8) as pool:old_pixels=set(pool.map(pixel_hash,[args.calibration/n for n in old_images]))
    old_scenes={(x['dataset'],s) for x in json.loads((args.calibration/'manifest.json').read_text())['datasets'] for s in x.get('scenes',[])}
    images={};pairs=[];scene_rows=[]
    def register(scene,name,meta):
        relative='images/'+scene+'/'+Path(name).name
        if relative in images:return relative
        source=Path(meta['source_path']);digest=hashlib.sha256(source.read_bytes()).hexdigest();pixels=pixel_hash(source)
        assert digest not in old_files and pixels not in old_pixels,'Calibration image leakage'
        target=args.output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        with Image.open(target) as im:
            im.load();assert im.size==(meta['width'],meta['height'])
        images[relative]={**meta,'sha256':digest,'decoded_rgb_sha256':pixels}
        return relative
    for i,(scene,(environment,count)) in enumerate(SCENES.items()):
        assert ('eth3d',scene) not in old_scenes
        frames,tracks=scene_data(args.source,scene)
        selected,eligible=choose_pairs(frames,tracks,count,SEED+i)
        for a,b,overlap,common,median in selected:
            image0=register(scene,a,frames[a]);image1=register(scene,b,frames[b])
            pairs.append(dict(image0=image0,image1=image1,scene=scene,dataset='eth3d',kind='positive',
                environment=environment,shared_reference_tracks=common,sparse_overlap=overlap,
                reference_track_median_sampson_px=median,label_source='shared supplied reference COLMAP tracks'))
        scene_rows.append(dict(scene=scene,environment=environment,pairs=count,eligible_pairs=eligible))
        print('SCENE',scene,count,'pairs from',eligible,flush=True)
    indoor=sorted(n for n,m in images.items() if m['environment']=='indoor')
    outdoor=sorted(n for n,m in images.items() if m['environment']=='outdoor')
    negative=list(itertools.product(indoor,outdoor));random.Random(SEED).shuffle(negative)
    used=Counter()
    for a,b in negative:
        if used[a]>=2 or used[b]>=2:continue
        used[a]+=1;used[b]+=1
        pairs.append(dict(image0=a,image1=b,scene=images[a]['scene']+'__'+images[b]['scene'],
            dataset='eth3d',kind='negative',environment='indoor__outdoor',
            label_source='constructed easy negative: different indoor/outdoor benchmark scenes; not a hard-negative benchmark'))
        if len(pairs)==100:break
    random.Random(SEED).shuffle(pairs)
    for i,row in enumerate(pairs):row['pair_id']=f'validation_{i:04d}'
    assert len(pairs)==100 and Counter(p['kind'] for p in pairs)=={'positive':80,'negative':20}
    content_pairs={tuple(sorted(images[p[k]]['sha256'] for k in ['image0','image1'])) for p in pairs}
    assert len(content_pairs)==100
    for n,m in images.items():assert hashlib.sha256((args.output/n).read_bytes()).hexdigest()==m['sha256']
    write_json(args.output/'images.json',images)
    (args.output/'pairs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in pairs))
    (args.output/'pairs.txt').write_text(''.join(f"{p['image0']} {p['image1']}\n" for p in pairs))
    manifest=dict(version=1,purpose='Frozen held-out validation only; never use for PTQ calibration/QAT or choosing scales',
        seed=SEED,pairs=100,positive_pairs=80,negative_pairs=20,images=len(images),scenes=scene_rows,
        calibration_manifest_sha256=hashlib.sha256((args.calibration/'manifest.json').read_bytes()).hexdigest(),
        pair_manifest_sha256=hashlib.sha256((args.output/'pairs.jsonl').read_bytes()).hexdigest(),
        image_manifest_sha256=hashlib.sha256((args.output/'images.json').read_bytes()).hexdigest(),
        scene_disjoint_from_calibration=True,file_hash_overlap_with_calibration=0,decoded_pixel_hash_overlap_with_calibration=0,
        reference='Official ETH3D test image calibration and supplied SfM tracks; not hidden laser depth ground truth',
        pair_selection='Source tracks/poses and fixed seed only; no FlexGlue outputs or quantization results used',
        source_url='https://www.eth3d.net/datasets',source_root=str(args.source),
        image_bytes=sum((args.output/n).stat().st_size for n in images))
    write_json(args.output/'manifest.json',manifest)
    write_json(args.output/'validation.json',dict(pair_count=100,unique_content_pairs=100,
        all_images_decoded=True,all_image_sha256_verified=True,calibration_file_hash_overlap=0,
        calibration_decoded_pixel_overlap=0,calibration_scene_overlap=0,
        max_positive_reference_median_sampson_px=max(p.get('reference_track_median_sampson_px',0) for p in pairs)))
    (args.output/'README.txt').write_text('''FlexGlue 独立验证集 v1：100 对（80 正样本 + 20 容易负样本对照）
正样本来自 ETH3D high-res test 的 7 个独立场景：40 对室内、40 对室外。
每对至少共享 50 条参考稀疏轨迹，并检查参考相机的极线一致性。
负样本由不同室内/室外场景组合而成，不等同于 Doppelgangers 困难负样本。
全部图像与现有 800 对校准候选的文件哈希和解码 RGB 像素哈希无交集；场景也不重叠。
这是自建配对子集；相机/轨迹来自官方 test 图像标定，不是 ETH3D 官方成对或激光深度评测。
选对过程不使用模型输出或量化结果。固定此集合，只用于验证，不用于 PTQ 校准、QAT 或选择缩放范围。
pairs.jsonl：图像路径相对本目录，含正负类别、场景和参考轨迹信息。
images.json：OpenCV camera_to_world、K、像素尺寸和来源。
manifest.json / validation.json：固定种子、清单指纹及与校准集的隔离检查。
评估应同时记录匹配几何质量、相对位姿误差、正样本拒绝率、负样本误接受率、匹配耗时和早退层数。
''')
    print('COMPLETE',json.dumps(manifest),flush=True)


if __name__=='__main__':main()
