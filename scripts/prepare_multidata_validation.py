"""Download held-out 100-pair subsets with calibration-image exclusion checks."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import itertools
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from scripts import prepare_flexglue_samples as source
from scripts.sample_data_io import write_json, fetch, RemoteFile, prefetch_blocks

ROOT = source.ROOT
OUTPUT = ROOT/'data/flexglue_eval_multidata_100'
SEED = 20261118


def pixel_hash(path):
    with Image.open(path) as im:
        pixels=np.asarray(im.convert('RGB'))
    h=hashlib.sha256()
    h.update(str(pixels.shape).encode());h.update(pixels.tobytes())
    return h.hexdigest()


class ValidationSet(source.SampleSet):
    def __init__(self, root, name):
        super().__init__(root,name)
        calibration=ROOT/'data/flexglue_ptq_100'
        self.calibration_manifest_sha256=hashlib.sha256((calibration/'manifest.json').read_bytes()).hexdigest()
        manifests=[(calibration,json.loads((calibration/'images.json').read_text())),
                   (ROOT/'data/flexglue_eval_100',json.loads((ROOT/'data/flexglue_eval_100/images.json').read_text()))]
        self.excluded_sha={m['sha256'] for _,images in manifests for m in images.values()}
        self.excluded_pixels=set()
        # Reuse a deterministic hash cache; existing manifests remain untouched.
        from concurrent.futures import ThreadPoolExecutor
        paths=[folder/p for folder,images in manifests for p in images]
        with ThreadPoolExecutor(8) as pool:self.excluded_pixels=set(pool.map(pixel_hash,paths))
        records=json.loads((calibration/'manifest.json').read_text())['datasets']
        self.exclude_scenes={s for r in records if r['dataset']==name for s in r['scenes']}
        self.report.update(seed=SEED,purpose='Held-out validation only; no PTQ calibration or QAT',
            calibration_manifest_sha256=self.calibration_manifest_sha256,
            excluded_calibration_scenes=sorted(self.exclude_scenes))
        self.save_status()

    def image(self, relative, payload=None, local=None, **metadata):
        # Always copy evaluation images: no mutable inode sharing with other sets.
        if local is not None:
            payload=Path(local).read_bytes();local=None
        if payload is not None and hashlib.sha256(payload).hexdigest() in self.excluded_sha:
            raise ValueError(f'Image duplicates calibration/ETH validation content: {relative}')
        path=super().image(relative,payload=payload,local=local,**metadata)
        ph=pixel_hash(self.root/path)
        if ph in self.excluded_pixels:
            raise ValueError(f'Decoded pixels duplicate calibration/ETH validation: {relative}')
        self.images[path]['decoded_rgb_sha256']=ph
        return path

    def complete(self):
        super().complete()
        scenes=set(self.report['scenes'])
        self.report.update(purpose='Held-out validation only; no PTQ calibration or QAT',
            calibration_file_hash_overlap=0,calibration_decoded_pixel_overlap=0,
            calibration_scene_overlap=sorted(scenes&self.exclude_scenes),
            scene_disjoint_from_calibration=not bool(scenes&self.exclude_scenes))
        self.save_status()


def aerialmegadepth(sample):
    """Use unused images of the public example when full-dataset access is denied."""
    archive=source.archive_from_index('aerial')
    folder=source.SOURCES/'aerial_metadata'
    cameras={}
    for line in (folder/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        p=line.split();assert p[1]=='SIMPLE_PINHOLE'
        f,cx,cy=map(float,p[4:7]);cameras[int(p[0])]=dict(K=[[f,0,cx],[0,f,cy],[0,0,1]],
            source_width=int(p[2]),source_height=int(p[3]))
    existing=json.loads((ROOT/'data/flexglue_ptq_100/images.json').read_text())
    excluded={Path(m.get('source_member',n)).name for n,m in existing.items()
              if n.startswith(('aerialmegadepth/','megadepth/'))}
    frames={}
    with (folder/'images.txt').open() as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            p=line.split();points=f.readline().split();name=Path(p[9]).name
            if name in excluded:continue
            frames[name]=dict(q=list(map(float,p[1:5])),t=list(map(float,p[5:8])),camera=cameras[int(p[8])],
                tracks={int(x) for x in points[2::3] if int(x)>=0})
    air=sorted(n for n in frames if n.startswith('0001_') and n.endswith('.jpeg'))
    ground=sorted(set(frames)-set(air));candidates=[]
    for a,b in itertools.product(air,ground):
        x,y=frames[a]['tracks'],frames[b]['tracks'];common=len(x&y);overlap=common/max(min(len(x),len(y)),1)
        if common>=30 and .05<=overlap<=.8:candidates.append((a,b,overlap,common))
    selected=source.select_pairs(candidates,100,seed=SEED,score_key=lambda p:p[2])
    prefix='0001/sfm_output_localization/sfm_superpoint+superglue/localized_dense_metric/images/'
    def download(name):
        frame=frames[name];member=prefix+name
        return name,sample.image('0001/'+name,payload=archive.read(member),**frame['camera'],
            world_to_camera_quaternion_wxyz=frame['q'],world_to_camera_translation=frame['t'],
            view_type='aerial_render' if name in air else 'MegaDepth_ground_photo',
            source_member=member,source_url=archive.urls[0])
    names=sorted({n for p in selected for n in p[:2]})
    with ThreadPoolExecutor(8) as pool:paths=dict(pool.map(download,names))
    for a,b,overlap,common in selected:
        sample.pair(paths[a],paths[b],'0001',kind='positive',source_split='Unused images in official public sample',
            overlap=overlap,overlap_type='shared localized SfM tracks / min track count',shared_points=common)
    sample.report.update(source_url='https://github.com/kvuong2711/aerial-megadepth/tree/main/data_generation',
        selection='100 aerial-ground pairs, images excluded against both original aerial/ground calibration subsets',
        ground_truth_pose=True,ground_truth_depth=False,
        limitation='Full dataset new-scene ZIP access returned HTTP 403 with existing credentials. '
            'Public scene0001 only: images/pairs disjoint from calibration, scene is shared. '
            'Reference poses/points come from localized SfM, not independent laser ground truth.')
    sample.complete()


def scannetpp(sample):
    archive=source.archive_from_index('scannetpp_eval_batch')
    metadata_names=sorted(n for n in archive.by_name if n.endswith('/nerfstudio/transforms_undistorted.json'))
    metadata_names=[n for n in metadata_names if n.split('/')[0] not in sample.exclude_scenes]
    assert len(metadata_names)>=3
    for scene_index,metadata_name in enumerate(metadata_names[:3]):
        scene=metadata_name.split('/')[0];raw=archive.read(metadata_name);meta=json.loads(raw)
        dest=sample.root/'metadata'/scene/'transforms_undistorted.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        prefix=scene+'/dslr/resized_undistorted_images/'
        frames=sorted([f for f in meta['frames'] if not f.get('is_bad',False) and
                       prefix+Path(f['file_path']).name in archive.by_name],key=lambda f:f['file_path'])
        candidates=[]
        for i in range(len(frames)):
            for j in range(i+2,min(i+26,len(frames))):
                x=np.array(frames[i]['transform_matrix']);y=np.array(frames[j]['transform_matrix'])
                if np.linalg.norm(x[:3,3]-y[:3,3])<.02:continue
                angle=float(np.degrees(np.arccos(np.clip((np.trace(x[:3,:3].T@y[:3,:3])-1)/2,-1,1))))
                if angle<45:candidates.append((i,j,angle))
        selected=source.select_pairs(candidates,34 if scene_index==0 else 33,seed=SEED+scene_index,score_key=lambda p:p[2])
        def download(i):
            frame=frames[i];name=Path(frame['file_path']).name;member=prefix+name;payload=archive.read(member)
            with Image.open(io.BytesIO(payload)) as im:w,h=im.size
            params={**meta,**frame};sx=w/params['w'];sy=h/params['h']
            K=[[params['fl_x']*sx,0,params['cx']*sx],[0,params['fl_y']*sy,params['cy']*sy],[0,0,1]]
            c2w=np.array(frame['transform_matrix']);c2w[:3,1:3]*=-1
            return i,sample.image(scene+'/'+name,payload=payload,K=K,camera_to_world=c2w.tolist(),camera_model='PINHOLE',
                camera_convention='OpenCV',source_member=member,source_url=archive.urls[0])
        names=sorted({n for p in selected for n in p[:2]})
        with ThreadPoolExecutor(8) as pool:paths=dict(pool.map(download,names))
        for a,b,angle in selected:
            sample.pair(paths[a],paths[b],scene,kind='positive',source_split='New scene, non-bad undistorted DSLR frames',
                relative_rotation_degrees=angle,overlap=None,
                overlap_type='Nearby ordered good frames; >2cm baseline, rotation<45deg. Dense overlap not measured.')
        sample.save_status();print('SCANNETPP',scene,len(sample.pairs),flush=True)
    sample.report.update(source_url='https://huggingface.co/datasets/yifei-liu/scannetppv2',
        selection='Three new scenes, 34/33/33 nearby-view pairs',ground_truth_pose=True,ground_truth_depth=False,
        calibration='Undistorted pinhole images, official laser-aligned reference poses',
        limitation='Candidate overlap selected by trajectory/rotation; no dense depth/overlap or full sparse track file downloaded.')
    sample.complete()


def megadepth(sample):
    archive=source.archive_from_index('megadepth_eval')
    metadata={n:archive.read('megadepth1500/'+n) for n in ['views.txt','pairs.txt','pairs_calibrated.txt']}
    for name,payload in metadata.items():
        dest=sample.root/'metadata'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
    frames={}
    for line in metadata['views.txt'].decode().splitlines():
        if not line.strip() or line.startswith('#'):continue
        p=line.split();assert p[13]=='PINHOLE'
        w2c=np.eye(4);w2c[:3,:3]=np.asarray(p[1:10],dtype=float).reshape(3,3);w2c[:3,3]=np.asarray(p[10:13],dtype=float)
        w,h=map(int,p[14:16]);fx,fy,cx,cy=map(float,p[16:20])
        frames[p[0]]=dict(K=[[fx,0,cx],[0,fy,cy],[0,0,1]],world_to_camera=w2c.tolist(),
            source_width=w,source_height=h,camera_model='PINHOLE')
    groups={}
    for line in metadata['pairs.txt'].decode().splitlines():
        if not line.strip() or line.startswith('#'):continue
        a,b=line.split()[:2];scene=a.split('/')[0];assert b.split('/')[0]==scene
        if scene not in sample.exclude_scenes:groups.setdefault(scene,[]).append((a,b))
    assert set(groups)=={'0015','0022'}
    selected=[]
    for index,scene in enumerate(sorted(groups)):
        selected.extend(source.select_pairs(groups[scene],50,seed=SEED+index))
    names=sorted({n for pair in selected for n in pair})
    def download(name):
        member='megadepth1500/images/'+name
        return name,sample.image(name,payload=archive.read(member),**frames[name],source_url=archive.urls[0],source_member=member)
    with ThreadPoolExecutor(8) as pool:paths=dict(pool.map(download,names))
    for a,b in selected:
        sample.pair(paths[a],paths[b],a.split('/')[0],kind='positive',source_split='MegaDepth-1500 official evaluation pairs',
            overlap=None,overlap_type='Official benchmark pair; no new model-dependent pair selection')
    sample.report.update(source_url=archive.urls[0],selection='50 official benchmark pairs per scene0015/0022',
        ground_truth_pose=True,ground_truth_depth=False,
        limitation='100-pair subset of MegaDepth-1500, not the complete 1500-pair benchmark. RGB and reference cameras only.')
    sample.complete()


def tartanair(sample):
    h5py=source.hdf5_module()
    folder=source.SOURCES/'tartan_val_metadata'
    meta=json.loads((folder/'metadata.json').read_text())
    rows=[dict(zip(meta['columns'],r)) for r in meta['data']]
    matrices={}
    for name in ['cam1_extrinsics','cam2_extrinsics','cam1_intrinsics','cam2_intrinsics']:
        with np.load(folder/(name+'.npz'),allow_pickle=False) as z:matrices[name]=z['arr_0']
    tree=json.loads((source.SOURCES/'tartan_val_tree.json').read_text())
    files=sorted([e for e in tree if e['path'].endswith('.h5')],key=lambda e:int(Path(e['path']).stem.split('_')[0]))
    selected=[]
    # Official val may share environment labels with train; trajectories and images are checked below.
    for item in files[:2]:
        start,end=map(int,Path(item['path']).stem.split('_'))
        eligible=[i for i in range(start,end+1) if rows[i]['img1_env']==rows[i]['img2_env']]
        picked=source.select_pairs([(i,i+100000) for i in eligible],50,seed=SEED+start)
        selected.append((item,[a for a,_ in picked]))
    for item,indices in selected:
        start=int(Path(item['path']).stem.split('_')[0])
        url='https://huggingface.co/datasets/infinity1096/TA-WB-MapAnything/resolve/main/'+item['path']
        remote=RemoteFile(url,size=item['size'])
        with h5py.File(remote,'r') as h:
            print('TARTAN VAL SHARD',item['path'],{k:h[k].shape for k in h},flush=True)
            blocks=set()
            for name in ['img0','img1']:
                ds=h[name]
                for global_i in indices:
                    chunk=ds.id.get_chunk_info_by_coord((global_i-start,0,0,0))
                    for b in range(chunk.byte_offset//remote.block_size*remote.block_size,
                        (chunk.byte_offset+chunk.size-1)//remote.block_size*remote.block_size+1,remote.block_size):blocks.add(b)
            prefetch_blocks(url,blocks,remote.block_size,remote.size)
            for global_i in indices:
                row=rows[global_i];paths=[]
                for direction,key in enumerate(['img0','img1'],1):
                    pixels=h[key][global_i-start];buf=io.BytesIO();Image.fromarray(pixels).save(buf,format='PNG')
                    paths.append(sample.image(f'{row["img1_env"]}/val_{global_i:06d}_{direction}.png',payload=buf.getvalue(),
                        K=matrices[f'cam{direction}_intrinsics'][global_i].tolist(),
                        camera_to_world=matrices[f'cam{direction}_extrinsics'][global_i].tolist(),
                        camera_convention='OpenCV, as supplied by MapAnything converter',source_url=url,
                        source_member=f'{key}[{global_i-start}]',source_global_row=global_i))
                sample.pair(*paths,row['img1_env'],kind='positive',source_split='official val',source_row=global_i,
                    source_pair_metadata=row,overlap=None,overlap_type='Official wide-baseline pair; dense overlap not measured')
            sample.save_status();print('TARTAN VAL',len(sample.pairs),flush=True)
    sample.report.update(source_url='https://huggingface.co/datasets/infinity1096/TA-WB-MapAnything',
        selection='100 official val pairs from two HDF5 shards, 50 per shard',
        ground_truth_pose=True,ground_truth_depth=False,
        limitation='Official train/val environment names may coincide; image content and trajectories are separately checked. No dense depth downloaded.')
    sample.complete()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['mapfree','blendedmvs','hypersim','dl3dv','aerialmegadepth','megadepth','scannetpp','tartanair'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    complete=args.output/args.dataset/'status.json'
    if complete.exists() and json.loads(complete.read_text())['state']=='complete':
        print('ALREADY COMPLETE',args.dataset);return
    source.SEED=SEED
    sample=ValidationSet(args.output,args.dataset)
    try:
        function=globals().get(args.dataset) or getattr(source,args.dataset)
        function(sample)
    except Exception as e:
        sample.report.update(state='failed',error=f'{type(e).__name__}: {e}');sample.save_status();raise


if __name__=='__main__':main()
