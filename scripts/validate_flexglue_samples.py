"""Audit downloaded image pairs, reference calibration, and provenance."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.sample_data_io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/flexglue_ptq_100')
    args=parser.parse_args();records=[];content_owners={};validation=[];all_images={};all_pairs=[]
    for status in sorted(args.data.glob('*/status.json')):
        report=json.loads(status.read_text());name=report['dataset']
        if report['state']!='complete':
            records.append(report);continue
        images=json.loads((status.parent/'images.json').read_text())
        pairs=[json.loads(line) for line in (status.parent/'pairs.jsonl').read_text().splitlines()]
        assert len(pairs)==100
        assert len({tuple(sorted([p['image0'],p['image1']])) for p in pairs})==100
        assert len({tuple(sorted(images[p[k]]['sha256'] for k in ['image0','image1'])) for p in pairs})==100
        assert {p[k] for p in pairs for k in ('image0','image1')}==set(images)
        max_rotation_error=0.
        for relative,meta in images.items():
            path=status.parent/relative
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
            with Image.open(path) as im:
                im.load();assert im.size==(meta['width'],meta['height'])
            K=np.array(meta['K'],dtype=float)
            assert K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0
            assert -.5<=K[0,2]<=meta['width']+.5 and -.5<=K[1,2]<=meta['height']+.5
            if 'source_width' in meta:assert (meta['source_width'],meta['source_height'])==(meta['width'],meta['height'])
            for key in ['camera_to_world','world_to_camera']:
                if key not in meta:continue
                pose=np.array(meta[key],dtype=float)
                assert pose.shape==(4,4) and np.isfinite(pose).all() and np.allclose(pose[3],[0,0,0,1])
                R=pose[:3,:3];error=float(np.max(np.abs(R.T@R-np.eye(3))))
                max_rotation_error=max(max_rotation_error,error)
                assert error<1e-3 and abs(np.linalg.det(R)-1)<1e-3,(name,relative,error)
            if 'world_to_camera_quaternion_wxyz' in meta:
                q=np.array(meta['world_to_camera_quaternion_wxyz']);assert abs(np.linalg.norm(q)-1)<1e-3
                assert np.isfinite(meta['world_to_camera_translation']).all()
            unified=dict(meta)
            if 'camera_to_world' not in unified:
                if 'world_to_camera' in unified:world_to_camera=np.array(unified['world_to_camera'])
                else:
                    w,x,y,z=np.array(unified['world_to_camera_quaternion_wxyz'])/np.linalg.norm(unified['world_to_camera_quaternion_wxyz'])
                    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                    world_to_camera=np.eye(4);world_to_camera[:3,:3]=R;world_to_camera[:3,3]=unified['world_to_camera_translation']
                unified['camera_to_world']=np.linalg.inv(world_to_camera).tolist()
            unified['camera_convention']='OpenCV'
            unified.setdefault('camera_model','PINHOLE')
            all_images[name+'/'+relative]=unified
            content_owners.setdefault(meta['sha256'],set()).add(name)
        for pair in pairs:
            all_pairs.append({**pair,'dataset':name,'image0':name+'/'+pair['image0'],'image1':name+'/'+pair['image1']})
        record={**report,'scene_count':len(report['scenes']),'image_count':len(images),'pair_count':len(pairs)}
        records.append(record)
        validation.append(dict(dataset=name,decoded_images=len(images),unique_content_pairs=100,
            sha256_verified=True,calibration_finite=True,max_rotation_orthogonality_error=max_rotation_error))
    shared={}
    for owners in content_owners.values():
        if len(owners)>1:
            key=' + '.join(sorted(owners));shared[key]=shared.get(key,0)+1
    write_json(args.data/'manifest.json',dict(datasets=records,
        completed_datasets=sum(r['state']=='complete' for r in records),
        total_pairs=sum(r.get('pair_count',0) for r in records),
        total_images=sum(r.get('image_count',0) for r in records),
        cross_dataset_shared_image_hashes=shared))
    write_json(args.data/'validation.json',dict(checks=validation,cross_dataset_shared_image_hashes=shared))
    write_json(args.data/'images.json',all_images)
    (args.data/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in all_pairs))
    (args.data/'pairs.txt').write_text(''.join(f"{p['image0']} {p['image1']}\n" for p in all_pairs))
    lines=['FlexGlue 每数据集 100 对真实图像小样本','',
           'dataset             state                            pairs images scenes']
    for r in records:
        lines.append(f"{r['dataset']:20s}{r['state']:33s}{r.get('pair_count',0):6d}{r.get('image_count',0):7d}{r.get('scene_count',0):7d}")
    lines += ['', '目录格式：每个数据集包含 images/、images.json、pairs.txt、pairs.jsonl、status.json。',
        'pairs.txt 中的路径相对该数据集目录；pairs.jsonl 记录场景、来源及可用的重叠/配对信息。',
        'images.json 记录逐图 SHA256、尺寸、内参、参考位姿和来源。',
        '根目录 pairs.jsonl/pairs.txt 合并全部图像对；根目录 images.json 统一提供 OpenCV camera_to_world。',
        '下载仅提取选定 RGB 与相机元数据，没有下载完整数据集或全部深度。',
        'ScanNet++ 使用本地已有、非 bad 的训练帧，以硬链接复用原图。',
        'Map-free、BlendedMVS、Hypersim 各取 5 个训练场景、每场景 20 对。',
        'TartanAir 取官方 MapAnything 训练 HDF5 首个分片中的 100 对，按环境分散抽样。',
        'MegaDepth 暂取 AerialMegaDepth 官方公开样例中的地面照片；AerialMegaDepth 取同场景空地配对。',
        '因此这两份子集共享场景和部分图像，不能当作独立评估集。共享图像哈希统计见 validation.json。',
        '可用重叠分数类型不同：Map-free 为官方 SfM 共视分数，ScanNet++/空地样例为共享稀疏轨迹比例。',
        'BlendedMVS 使用官方 cams/pair.txt；Hypersim/TartanAir 没有计算稠密重叠，已在每对元数据中标明。',
        '这是初期 PTQ 校准/量化回归样本集合，尚未划分独立校准集和评估集，也不是官方匹配基准。',
        '本次覆盖 8 个主训练数据集，未包含负样本微调的 Doppelgangers。',
        'DL3DV 通过服务器已有的 Hugging Face 授权访问，按需提取 5 个官方 480P 场景、每场景 20 对。',
        'DL3DV 保留原始 OpenCV 畸变参数，K 已缩放到实际图像尺寸，几何评估需处理畸变。',
        '授权入口：https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P', '',
        '校验：图像全部解码、SHA256 一致、每集 100 对内容不重复、内参有限、旋转矩阵与四元数检查。',
        '重跑校验：python -m scripts.validate_flexglue_samples',
        'HDF5 读取依赖只解包到 dataset_sources/python_deps，未安装/替换 sfm 中的 package。']
    (args.data/'README.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
