"""Prepare small, traceable real-image subsets for FlexGlue quantization studies."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image

from scripts.sample_data_io import RemoteZip, write_json, fetch

ROOT = Path(__file__).resolve().parents[1]
SOURCES = Path(os.environ.get('FLEXGLUE_DATA_SOURCES', ROOT/'dataset_sources'))
SEED = 20261116


def archive_from_index(name):
    info = json.loads((SOURCES/f'{name}_index.json').read_text())
    archive = RemoteZip.__new__(RemoteZip)
    archive.urls, archive.sizes, archive.entries = info['urls'], info['sizes'], info['entries']
    archive.starts = [0]
    for n in archive.sizes: archive.starts.append(archive.starts[-1]+n)
    archive.by_name = {e['path']: e for e in archive.entries}
    return archive


class SampleSet:
    def __init__(self, root, name):
        self.root = Path(root)/name
        self.root.mkdir(parents=True, exist_ok=True)
        self.name, self.images, self.pairs = name, {}, []
        self.report = dict(dataset=name, state='preparing', seed=SEED)
        self.save_status()

    def save_status(self):
        write_json(self.root/'status.json', {**self.report, 'pairs':len(self.pairs), 'images':len(self.images)})

    def image(self, relative, payload=None, local=None, **metadata):
        relative = 'images/'+relative
        if relative in self.images: return relative
        target = self.root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if local is not None:
                try: os.link(local, target)
                except OSError: shutil.copy2(local, target)
            else:
                temp=target.with_suffix(target.suffix+'.part');temp.write_bytes(payload);temp.replace(target)
        with Image.open(target) as image:
            image.load()
            width, height = image.size
            if image.mode not in ('RGB','RGBA'): raise ValueError(f'Unexpected image mode {image.mode}')
        self.images[relative] = dict(width=width, height=height,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(), **metadata)
        return relative

    def pair(self, a, b, scene, **metadata):
        if a==b: raise ValueError('Self pair')
        self.pairs.append(dict(pair_id=f'{self.name}_{len(self.pairs):04d}',
                               image0=a, image1=b, scene=scene, **metadata))

    def complete(self):
        assert len(self.pairs)==100
        assert len({tuple(sorted([p['image0'],p['image1']])) for p in self.pairs})==100
        assert all(p[k] in self.images for p in self.pairs for k in ('image0','image1'))
        write_json(self.root/'images.json',self.images)
        (self.root/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in self.pairs))
        (self.root/'pairs.txt').write_text(''.join(f"{p['image0']} {p['image1']}\n" for p in self.pairs))
        self.report.update(state='complete', scenes=sorted({p['scene'] for p in self.pairs}),
            image_bytes=sum((self.root/p).stat().st_size for p in self.images),
            purpose='Small calibration/quantization-regression sample; not a full training or official evaluation split.')
        self.save_status()
        print('COMPLETE',self.name,len(self.pairs),'pairs',len(self.images),'images',flush=True)


def select_pairs(candidates, count, seed=SEED, score_key=None):
    candidates=list(candidates)
    random.Random(seed).shuffle(candidates)
    # Distribute across overlap scores when an actual score is available.
    if score_key:
        candidates.sort(key=score_key)
        indices=np.linspace(0,len(candidates)-1,min(count*4,len(candidates)),dtype=int)
        candidates=[candidates[i] for i in indices]
        random.Random(seed).shuffle(candidates)
    selected, seen=[],set()
    for p in candidates:
        key=tuple(sorted([str(p[0]),str(p[1])]))
        if p[0]==p[1] or key in seen: continue
        seen.add(key);selected.append(p)
        if len(selected)==count:return selected
    raise ValueError(f'Only {len(selected)} distinct candidate pairs')


def scannetpp(sample):
    base=getattr(sample, 'scannetpp_source', None)
    if base is None:
        raise ValueError('ScanNet++ requires --scannetpp-source pointing to the local DSLR scene directory')
    meta=json.loads((base/'nerfstudio/transforms_undistorted.json').read_text())
    frames={Path(f['file_path']).name:f for f in meta['frames'] if not f.get('is_bad',False)}
    tracks={}
    lines=(base/'colmap/images.txt').read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line or line.startswith('#'):continue
        parts=line.split();points=lines[i].split();i+=1
        name=Path(parts[9]).name
        if name in frames:tracks[name]={int(x) for x in points[2::3] if int(x)>=0}
    candidates=[]
    for a,b in itertools.combinations(sorted(frames),2):
        if a not in tracks or b not in tracks:continue
        common=len(tracks[a]&tracks[b]);den=min(len(tracks[a]),len(tracks[b]))
        overlap=common/max(den,1)
        if common>=30 and .05<=overlap<=.8:candidates.append((a,b,overlap,common))
    selected=select_pairs(candidates,100,score_key=lambda p:p[2])
    K=[[meta['fl_x'],0,meta['cx']],[0,meta['fl_y'],meta['cy']],[0,0,1]]
    for a,b,overlap,common in selected:
        paths=[]
        for name in (a,b):
            c2w=np.array(frames[name]['transform_matrix'],dtype=float);c2w[:3,1:3]*=-1
            paths.append(sample.image('8c3169b088/'+name,local=base/'resized_undistorted_images'/name,
                K=K,camera_to_world=c2w.tolist(),camera_convention='OpenCV',
                source_path=str(base/'resized_undistorted_images'/name)))
        sample.pair(*paths,'8c3169b088',source_split='train frames, is_bad excluded',
                    overlap=overlap,overlap_type='sparse shared COLMAP tracks / min track count',shared_points=common)
    sample.report.update(source='Existing local ScanNet++ v2 DSLR subset',
        source_url='https://huggingface.co/datasets/yifei-liu/scannetppv2',
        selection='100 distinct pairs from non-bad train frames, >=30 shared reference tracks; overlap-stratified',
        calibration='Undistorted pinhole images with laser-aligned reference poses',ground_truth_depth=False)
    sample.complete()


def mapfree(sample):
    archive=archive_from_index('mapfree')
    scenes=sorted({p.split('/')[1] for p in archive.by_name if p.endswith('/overlaps.npz')}
                  -set(getattr(sample,'exclude_scenes',())))
    chosen=random.Random(SEED).sample(scenes,5)
    for scene_index,scene in enumerate(chosen):
        prefix='train/'+scene+'/'
        metadata={n:archive.read(prefix+n) for n in ('overlaps.npz','poses.txt','intrinsics.txt')}
        for name,data in metadata.items():
            target=sample.root/'metadata'/scene/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        with np.load(io.BytesIO(metadata['overlaps.npz']),allow_pickle=False) as overlaps:
            idxs,scores=overlaps['idxs'],overlaps['overlaps']
            candidates=[]
            for row,score in zip(idxs,scores):
                score=float(score)
                a=f'seq{int(row[0])}/frame_{int(row[1]):05d}.jpg';b=f'seq{int(row[2])}/frame_{int(row[3]):05d}.jpg'
                if .05<=score<=.8 and prefix+a in archive.by_name and prefix+b in archive.by_name:
                    candidates.append((a,b,score))
        selected=select_pairs(candidates,20,seed=SEED+scene_index,score_key=lambda x:x[2])
        intrinsics={}
        for line in metadata['intrinsics.txt'].decode().splitlines():
            if not line.strip() or line.startswith('#'):continue
            p=line.split();fx,fy,cx,cy=map(float,p[1:5]);intrinsics[p[0]]=[[fx,0,cx],[0,fy,cy],[0,0,1]]
        poses={}
        for line in metadata['poses.txt'].decode().splitlines():
            if not line.strip() or line.startswith('#'):continue
            p=line.split();poses[p[0]]=list(map(float,p[1:]))
        names=sorted({n for p in selected for n in p[:2]})
        def download(name):
            q=poses[name]
            return name,sample.image(scene+'/'+name,payload=archive.read(prefix+name),K=intrinsics[name],
                world_to_camera_quaternion_wxyz=q[:4],world_to_camera_translation=q[4:],
                source_url=archive.urls[0],source_member=prefix+name)
        with ThreadPoolExecutor(8) as pool:paths=dict(pool.map(download,names))
        for a,b,overlap in selected:
            sample.pair(paths[a],paths[b],scene,source_split='train',overlap=overlap,overlap_type='official Map-free SfM covisibility')
        sample.save_status();print('MAPFREE',scene,len(sample.pairs),flush=True)
    sample.report.update(source_url=archive.urls[0],selection='5 train scenes x 20 unique overlap-stratified pairs, overlap [0.05,0.8]',
                         ground_truth_depth=False,ground_truth_pose=True)
    sample.complete()


def blendedmvs(sample):
    archive=archive_from_index('blended')
    available={p.split('/')[1] for p in archive.by_name if p.endswith('/cams/pair.txt')}
    train_file=SOURCES/'blended_training_list.txt'
    train={p.strip() for p in train_file.read_text().splitlines() if p.strip()}
    chosen=random.Random(SEED).sample(sorted((available&train)-set(getattr(sample,'exclude_scenes',()))),5)
    for scene_index,scene in enumerate(chosen):
        prefix='BlendedMVS/'+scene+'/'
        lines=archive.read(prefix+'cams/pair.txt').decode().splitlines()
        count=int(lines[0]);candidates=[]
        for i in range(count):
            a=int(lines[1+2*i]);parts=lines[2+2*i].split()
            for j in range(int(parts[0])):
                b=int(parts[1+2*j]);score=float(parts[2+2*j]);candidates.append((a,b,score))
        selected=select_pairs(candidates,20,seed=SEED+scene_index,score_key=lambda p:p[2])
        names=sorted({n for p in selected for n in p[:2]})
        def download(n):
            image_member=prefix+f'blended_images/{n:08d}.jpg'
            camera_member=prefix+f'cams/{n:08d}_cam.txt'
            raw=archive.read(camera_member);text=raw.decode().splitlines()
            e=text.index('extrinsic');k=text.index('intrinsic')
            extrinsic=[[float(x) for x in line.split()] for line in text[e+1:e+5]]
            K=[[float(x) for x in line.split()] for line in text[k+1:k+4]]
            metadata=sample.root/'metadata'/scene/f'{n:08d}_cam.txt';metadata.parent.mkdir(parents=True,exist_ok=True);metadata.write_bytes(raw)
            return n,sample.image(scene+f'/{n:08d}.jpg',payload=archive.read(image_member),K=K,
                world_to_camera=extrinsic,source_member=image_member,source_url='https://github.com/YoYo000/BlendedMVS/releases/tag/v1.0.0')
        with ThreadPoolExecutor(8) as pool:paths=dict(pool.map(download,names))
        for a,b,score in selected:
            sample.pair(paths[a],paths[b],scene,source_split='official training list',mvs_view_score=score,
                        overlap=None,overlap_type='Official MVS candidate pair; score is not a normalized overlap')
        sample.save_status();print('BLENDED',scene,len(sample.pairs),flush=True)
    sample.report.update(source_url='https://github.com/YoYo000/BlendedMVS',selection='5 official training scenes x 20 distinct pairs from cams/pair.txt',
                         ground_truth_pose=True,ground_truth_depth=False,resolution='official low-res 768x576')
    sample.complete()


def aerial_ground(sample):
    archive=archive_from_index('aerial')
    meta_root=SOURCES/'aerial_metadata'
    prefix='0001/sfm_output_localization/sfm_superpoint+superglue/localized_dense_metric/'
    cameras={}
    for line in (meta_root/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        p=line.split();assert p[1]=='SIMPLE_PINHOLE'
        focal,cx,cy=map(float,p[4:7]);cameras[int(p[0])]=dict(K=[[focal,0,cx],[0,focal,cy],[0,0,1]],source_width=int(p[2]),source_height=int(p[3]))
    frames={}
    with (meta_root/'images.txt').open() as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            p=line.split();points=f.readline().split();name=Path(p[9]).name
            frames[name]=dict(q=list(map(float,p[1:5])),t=list(map(float,p[5:8])),camera=cameras[int(p[8])],
                              tracks={int(x) for x in points[2::3] if int(x)>=0})
    air=sorted(n for n in frames if n.startswith('0001_') and n.endswith('.jpeg'))
    ground=sorted(set(frames)-set(air))
    if sample.name=='aerialmegadepth':
        pairs=list(itertools.product(air,ground))
    else:pairs=list(itertools.combinations(ground,2))
    random.Random(SEED).shuffle(pairs)
    candidates=[]
    for a,b in pairs[:10000]:
        x,y=frames[a]['tracks'],frames[b]['tracks'];common=len(x&y);overlap=common/max(min(len(x),len(y)),1)
        if common>=30 and .05<=overlap<=.8:candidates.append((a,b,overlap,common))
    selected=select_pairs(candidates,100,score_key=lambda p:p[2])
    names=sorted({n for p in selected for n in p[:2]})
    def download(name):
        f=frames[name];member=prefix+'images/'+name
        return name,sample.image('0001/'+name,payload=archive.read(member),**f['camera'],
            world_to_camera_quaternion_wxyz=f['q'],world_to_camera_translation=f['t'],
            view_type='aerial_render' if name in air else 'MegaDepth_ground_photo',
            source_member=member,source_url=archive.urls[0])
    with ThreadPoolExecutor(8) as pool:paths=dict(pool.map(download,names))
    for a,b,overlap,common in selected:
        sample.pair(paths[a],paths[b],'0001',source_split='official AerialMegaDepth public sample',
                    overlap=overlap,overlap_type='sparse shared localized COLMAP tracks / min track count',shared_points=common)
    sample.report.update(source_url='https://github.com/kvuong2711/aerial-megadepth/tree/main/data_generation',
        selection='100 unique aerial-ground pairs' if sample.name=='aerialmegadepth' else '100 unique ground-ground pairs',
        ground_truth_pose=True,ground_truth_depth=False,
        limitation='One public sample scene. MegaDepth ground photos are provided through the official AerialMegaDepth sample; the two subsets share scene/content and are not independent evaluation sets.')
    sample.complete()


def hdf5_module():
    # A downloaded wheel is unpacked locally; sfm site-packages is untouched.
    sys.path.insert(0,str(SOURCES/'python_deps'))
    import h5py
    return h5py


def hypersim(sample):
    h5py=hdf5_module()
    split=list(csv.DictReader((SOURCES/'hypersim_splits.csv').open()))
    allowed=[r for r in split if r['included_in_public_release']=='True' and r['split_partition_name']=='train' and r['camera_name']=='cam_00']
    scenes=sorted({r['scene_name'] for r in allowed}-set(getattr(sample,'exclude_scenes',())))
    chosen=random.Random(SEED).sample(scenes,5)
    camera_parameters={r['scene_name']:r for r in csv.DictReader((SOURCES/'hypersim_cameras.csv').open())}
    for scene_index,scene in enumerate(chosen):
        index_path=SOURCES/f'hypersim_{scene}_index.json'
        if index_path.exists():archive=archive_from_index('hypersim_'+scene)
        else:
            archive=RemoteZip(f'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip')
            write_json(index_path,dict(urls=archive.urls,sizes=archive.sizes,entries=archive.entries))
        prefix=scene+'/_detail/cam_00/'
        values={}
        for name in ['camera_keyframe_frame_indices.hdf5','camera_keyframe_positions.hdf5','camera_keyframe_orientations.hdf5']:
            raw=archive.read(prefix+name)
            dest=sample.root/'metadata'/scene/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            with h5py.File(io.BytesIO(raw),'r') as h:values[name]=h['dataset'][()]
        scene_metadata=archive.read(scene+'/_detail/metadata_scene.csv')
        rows=list(csv.DictReader(io.StringIO(scene_metadata.decode())))
        scale=float(next(r['parameter_value'] for r in rows if r['parameter_name']=='meters_per_asset_unit'))
        frame_ids=values['camera_keyframe_frame_indices.hdf5'].reshape(-1).astype(int)
        positions=values['camera_keyframe_positions.hdf5']*scale
        orientations=values['camera_keyframe_orientations.hdf5']
        pose_by_id={int(n):(positions[i],orientations[i]) for i,n in enumerate(frame_ids)
                    if np.isfinite(positions[i]).all() and np.isfinite(orientations[i]).all()}
        image_by_id={int(Path(n).name.split('.')[1]):n for n in archive.by_name if '/scene_cam_00_final_preview/' in n and n.endswith('.tonemap.jpg')}
        valid={int(r['frame_id']) for r in allowed if r['scene_name']==scene}&set(pose_by_id)&set(image_by_id)
        candidates=[]
        for a,b in itertools.combinations(sorted(valid),2):
            if b-a>10:continue
            relative=pose_by_id[a][1].T@pose_by_id[b][1]
            angle=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
            if angle<60:candidates.append((a,b,angle))
        selected=select_pairs(candidates,20,seed=SEED+scene_index,score_key=lambda p:p[2])
        params=camera_parameters[scene];w=int(float(params['settings_output_img_width']));h=int(float(params['settings_output_img_height']))
        M=np.array([[float(params[f'M_proj_{i}{j}']) for j in range(4)] for i in range(4)])
        K=[[M[0,0]*w/2,-M[0,1]*w/2,(1-M[0,2])*w/2-.5],
           [-M[1,0]*h/2,M[1,1]*h/2,(1+M[1,2])*h/2-.5],[0,0,1]]
        names=sorted({n for p in selected for n in p[:2]})
        def download(n):
            position,orientation=pose_by_id[n];c2w=np.eye(4);c2w[:3,:3]=orientation@np.diag([1,-1,-1]);c2w[:3,3]=position
            return n,sample.image(scene+f'/frame.{n:04d}.jpg',payload=archive.read(image_by_id[n]),K=K,
                camera_to_world=c2w.tolist(),camera_convention='OpenCV',source_url=archive.urls[0],source_member=image_by_id[n])
        with ThreadPoolExecutor(8) as pool:paths=dict(pool.map(download,names))
        for a,b,angle in selected:
            sample.pair(paths[a],paths[b],scene,source_split='official train; public non-excluded frames',
                        relative_rotation_degrees=angle,overlap=None,overlap_type='Not computed; nearby trajectory frames, gap <=10')
        sample.save_status();print('HYPERSIM',scene,len(sample.pairs),flush=True)
    sample.report.update(source_url='https://github.com/apple/ml-hypersim',selection='5 training scenes x 20 unique nearby-frame pairs',
        ground_truth_pose=True,ground_truth_depth=False,rgb='Official tonemapped JPEG previews',
        calibration='K derived from official per-scene projection matrix with pixel centers; camera poses converted from OpenGL to OpenCV and meters',
        limitation='Dense overlap has not been measured; suitable for calibration/relative-pose checks, not an official pair benchmark.')
    sample.complete()


def tartanair(sample):
    from scripts.sample_data_io import RemoteFile
    h5py=hdf5_module()
    metadata=json.loads((SOURCES/'tartan_metadata/first506_metadata.json').read_text())
    rows=[dict(zip(metadata['columns'],r)) for r in metadata['data']]
    # Spread across the environments present in the first official training shard.
    groups={}
    for i,row in enumerate(rows):
        if row['img1_env']==row['img2_env']:groups.setdefault(row['img1_env'],[]).append(i)
    rng=random.Random(SEED)
    for v in groups.values():rng.shuffle(v)
    scene_names=sorted(groups);rng.shuffle(scene_names);chosen=[]
    while len(chosen)<100:
        advanced=False
        for scene in scene_names:
            if groups[scene]:chosen.append(groups[scene].pop());advanced=True
            if len(chosen)==100:break
        if not advanced:raise ValueError('Insufficient same-environment source pairs')
    chosen.sort()
    first=json.loads((SOURCES/'tartan_train.txt').read_text())[0]
    url='https://huggingface.co/datasets/infinity1096/TA-WB-MapAnything/resolve/main/'+first['path']
    remote=RemoteFile(url,size=first['size'],block_size=256*1024)
    matrices={}
    for name in ['cam1_extrinsics','cam2_extrinsics','cam1_intrinsics','cam2_intrinsics']:
        with np.load(SOURCES/'tartan_metadata'/f'{name}.npz',allow_pickle=False) as z:matrices[name]=z['arr_0'][chosen]
    with h5py.File(remote,'r') as h:
        blocks=set()
        for name in ['img0','img1']:
            dataset=h[name]
            for i in chosen:
                info=dataset.id.get_chunk_info_by_coord((i,0,0,0))
                for start in range(info.byte_offset//remote.block_size*remote.block_size,
                                   (info.byte_offset+info.size-1)//remote.block_size*remote.block_size+1,remote.block_size):
                    blocks.add(start)
        print('TARTAN prefetch',len(blocks),'blocks',len(blocks)*remote.block_size,'bytes',flush=True)
        from scripts.sample_data_io import prefetch_blocks
        prefetch_blocks(url,blocks,remote.block_size,remote.size)
        for j,i in enumerate(chosen):
            paths=[]
            row=rows[i]
            for direction,name in enumerate(['img0','img1'],1):
                pixels=h[name][i]
                payload=io.BytesIO();Image.fromarray(pixels).save(payload,format='PNG')
                paths.append(sample.image(f'{row["img1_env"]}/{i:06d}_{direction}.png',payload=payload.getvalue(),
                    K=matrices[f'cam{direction}_intrinsics'][j].tolist(),
                    camera_to_world=matrices[f'cam{direction}_extrinsics'][j].tolist(),
                    camera_convention='OpenCV, as supplied by official MapAnything converter',
                    source_url=url,source_member=f'{name}[{i}]'))
            sample.pair(*paths,row['img1_env'],source_split='train; first official HDF5 shard',
                        source_row=i,source_pair_metadata=row,overlap=None,overlap_type='Official wide-baseline source pair; dense overlap not computed')
            if (j+1)%20==0:sample.save_status();print('TARTAN',j+1,flush=True)
    sample.report.update(source_url='https://huggingface.co/datasets/infinity1096/TA-WB-MapAnything',
        selection='100 existing training pairs, environment-balanced within first 506-pair HDF5 shard; same-environment only',
        ground_truth_pose=True,ground_truth_depth=False,
        limitation='One training shard sampled; dense overlap not measured. RGB only was read, depth arrays were not downloaded.')
    sample.complete()


def dl3dv(sample):
    from huggingface_hub import get_token
    if not get_token():raise PermissionError('DL3DV requires existing authorized Hugging Face login')
    repo='DL3DV/DL3DV-ALL-480P'
    # This checks existing permission; it does not accept terms or request access.
    fetch('https://huggingface.co/api/datasets/'+repo+'/auth-check')
    tree=json.loads(fetch('https://huggingface.co/api/datasets/'+repo+'/tree/main/1K?limit=1000'))
    files=sorted([e for e in tree if e['path'].endswith('.zip') and
                  Path(e['path']).stem not in set(getattr(sample,'exclude_scenes',()))],key=lambda e:e['path'])
    chosen=random.Random(SEED).sample(files,5)
    for scene_index,item in enumerate(chosen):
        scene=Path(item['path']).stem
        index_path=SOURCES/f'dl3dv_{scene}_index.json'
        if index_path.exists():archive=archive_from_index('dl3dv_'+scene)
        else:
            url='https://huggingface.co/datasets/'+repo+'/resolve/main/'+item['path']
            archive=RemoteZip(url,[item['size']]);write_json(index_path,dict(urls=archive.urls,sizes=archive.sizes,entries=archive.entries))
        transforms_name=next(n for n in archive.by_name if n.endswith('/transforms.json') or n=='transforms.json')
        raw=archive.read(transforms_name);meta=json.loads(raw)
        dest=sample.root/'metadata'/scene/'transforms.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        image_files={Path(n).name:n for n in archive.by_name if n.lower().endswith(('.png','.jpg','.jpeg'))}
        frames=sorted([f for f in meta['frames'] if Path(f['file_path']).name in image_files],key=lambda f:f['file_path'])
        candidates=[]
        for i in range(len(frames)):
            for j in range(i+5,min(i+61,len(frames))):
                x=np.array(frames[i]['transform_matrix']);y=np.array(frames[j]['transform_matrix'])
                if np.linalg.norm(x[:3,3]-y[:3,3])<1e-6:continue
                angle=float(np.degrees(np.arccos(np.clip((np.trace(x[:3,:3].T@y[:3,:3])-1)/2,-1,1))))
                if angle<60:candidates.append((i,j,angle))
        selected=select_pairs(candidates,20,seed=SEED+scene_index,score_key=lambda p:p[2])
        names=sorted({n for p in selected for n in p[:2]})
        def download(i):
            frame=frames[i];name=Path(frame['file_path']).name;member=image_files[name]
            payload=archive.read(member)
            with Image.open(io.BytesIO(payload)) as im:w,h=im.size
            params={**meta,**frame};sx=w/params['w'];sy=h/params['h']
            K=[[params['fl_x']*sx,0,params['cx']*sx],[0,params['fl_y']*sy,params['cy']*sy],[0,0,1]]
            c2w=np.array(frame['transform_matrix']);c2w[:3,1:3]*=-1
            return i,sample.image(scene+'/'+name,payload=payload,K=K,camera_to_world=c2w.tolist(),
                camera_convention='OpenCV',camera_model=params.get('camera_model','OPENCV'),
                distortion={k:params.get(k,0.) for k in ['k1','k2','k3','k4','p1','p2']},
                calibration_reference_size=[params['w'],params['h']],intrinsic_scale=[sx,sy],
                source_url=archive.urls[0],source_member=member)
        with ThreadPoolExecutor(8) as pool:paths=dict(pool.map(download,names))
        for a,b,angle in selected:
            sample.pair(paths[a],paths[b],scene,source_split='official ALL-480P 1K shard; exact FlexGlue train membership unverified',
                relative_rotation_degrees=angle,overlap=None,overlap_type='Not measured; nonzero baseline, frame gap 5..60, relative rotation <60 degrees')
        sample.save_status();print('DL3DV',scene,len(sample.pairs),flush=True)
    sample.report.update(source_url='https://huggingface.co/datasets/'+repo,
        selection='5 existing authorized 480P scenes x 20 unique temporal/pose candidate pairs',
        ground_truth_pose=True,ground_truth_depth=False,
        calibration='Original OpenCV distortion retained; K scaled to downloaded image dimensions; OpenGL c2w converted to OpenCV',
        limitation='Dense overlap not measured. Camera translations have arbitrary SfM scale; distortion must be handled for geometric evaluation.')
    sample.complete()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['scannetpp','mapfree','blendedmvs','aerialmegadepth','megadepth','hypersim','tartanair','dl3dv'])
    parser.add_argument('--output',type=Path,default=ROOT/'data/flexglue_ptq_100')
    parser.add_argument('--scannetpp-source',type=Path,help='Local ScanNet++ DSLR scene directory')
    args=parser.parse_args();sample=SampleSet(args.output,args.dataset)
    sample.scannetpp_source=args.scannetpp_source
    try:
        if args.dataset in ('aerialmegadepth','megadepth'):aerial_ground(sample)
        else:globals()[args.dataset](sample)
    except Exception as e:
        sample.report.update(state='failed',error=f'{type(e).__name__}: {e}');sample.save_status();raise


if __name__=='__main__':main()
