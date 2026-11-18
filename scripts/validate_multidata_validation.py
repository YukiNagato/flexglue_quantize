"""Freeze eight held-out RGB/camera subsets and audit image/pair independence."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_multidata_validation import ROOT, OUTPUT, SEED, pixel_hash
from scripts.matching_metrics import quaternion_rotation
from scripts.sample_data_io import write_json

DATASETS=['aerialmegadepth','blendedmvs','dl3dv','hypersim','mapfree','megadepth','scannetpp','tartanair']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cameras(meta):
    out=dict(meta)
    K=np.asarray(out['K'],dtype=float)
    assert K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0
    assert -.5<=K[0,2]<=out['width']+.5 and -.5<=K[1,2]<=out['height']+.5
    if 'camera_to_world' not in out:
        if 'world_to_camera' in out:w2c=np.asarray(out['world_to_camera'],dtype=float)
        else:
            q=np.asarray(out['world_to_camera_quaternion_wxyz'],dtype=float)
            assert abs(np.linalg.norm(q)-1)<1e-3
            w2c=np.eye(4);w2c[:3,:3]=quaternion_rotation(q)
            w2c[:3,3]=out['world_to_camera_translation']
        out['camera_to_world']=np.linalg.inv(w2c).tolist()
    c2w=np.asarray(out['camera_to_world'],dtype=float)
    assert c2w.shape==(4,4) and np.isfinite(c2w).all() and np.allclose(c2w[3],[0,0,0,1])
    R=c2w[:3,:3]
    error=float(np.max(np.abs(R.T@R-np.eye(3))))
    assert error<1e-3 and abs(np.linalg.det(R)-1)<1e-3
    out['camera_convention']='OpenCV'
    out.setdefault('camera_model','PINHOLE')
    return out,error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=OUTPUT)
    args=parser.parse_args()
    calibration=ROOT/'data/flexglue_ptq_100'
    eth=ROOT/'data/flexglue_eval_100'
    old={}
    for root in [calibration,eth]:
        images=json.loads((root/'images.json').read_text())
        for path,meta in images.items():old[root/path]=meta
    old_files={m['sha256'] for m in old.values()}
    with ThreadPoolExecutor(8) as pool:old_pixels=set(pool.map(pixel_hash,old))
    old_pairs=set()
    for root in [calibration,eth]:
        images=json.loads((root/'images.json').read_text())
        for line in (root/'pairs.jsonl').read_text().splitlines():
            pair=json.loads(line);old_pairs.add(tuple(sorted(images[pair[k]]['sha256'] for k in ['image0','image1'])))
    records=[];all_images={};all_pairs=[];seen_pairs=set();owners={}
    for name in DATASETS:
        folder=args.data/name
        status=json.loads((folder/'status.json').read_text())
        assert status['state']=='complete',(name,status)
        assert status['calibration_manifest_sha256']==digest(calibration/'manifest.json')
        images=json.loads((folder/'images.json').read_text())
        pairs=[json.loads(line) for line in (folder/'pairs.jsonl').read_text().splitlines()]
        assert len(pairs)==100 and len({p['pair_id'] for p in pairs})==100
        assert {p[k] for p in pairs for k in ['image0','image1']}==set(images)
        normalized={};maximum_error=0.;total_bytes=0
        for relative,meta in images.items():
            path=folder/relative
            assert digest(path)==meta['sha256']
            assert meta['sha256'] not in old_files,(name,relative,'file overlap')
            with Image.open(path) as im:
                im.load();assert im.size==(meta['width'],meta['height'])
            ph=pixel_hash(path)
            assert ph==meta['decoded_rgb_sha256'] and ph not in old_pixels,(name,relative,'pixel overlap')
            normalized[relative],error=cameras(meta);maximum_error=max(maximum_error,error)
            total_bytes+=path.stat().st_size
            owners.setdefault(ph,set()).add(name)
        content_pairs=set();baselines=[]
        for p in pairs:
            hashes=tuple(sorted(normalized[p[k]]['decoded_rgb_sha256'] for k in ['image0','image1']))
            assert len(set(hashes))==2,(name,p['pair_id'],'identical images')
            assert hashes not in content_pairs and hashes not in seen_pairs,(name,p['pair_id'],'duplicate pair')
            content_pairs.add(hashes);seen_pairs.add(hashes)
            file_hashes=tuple(sorted(normalized[p[k]]['sha256'] for k in ['image0','image1']))
            assert file_hashes not in old_pairs
            p.setdefault('kind','positive')
            p['dataset']=name
            p['reference_label']='Candidate overlapping pair from source metadata; not a guaranteed dense-correspondence label'
            T0=np.asarray(normalized[p['image0']]['camera_to_world']);T1=np.asarray(normalized[p['image1']]['camera_to_world'])
            baselines.append(float(np.linalg.norm(T0[:3,3]-T1[:3,3])))
            all_pairs.append({**p,'image0':name+'/'+p['image0'],'image1':name+'/'+p['image1']})
        write_json(folder/'images.json',normalized)
        (folder/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
        (folder/'pairs.txt').write_text(''.join(f"{p['image0']} {p['image1']}\n" for p in pairs))
        record=dict(dataset=name,state='complete',pairs=100,images=len(images),scenes=status['scenes'],
            scene_count=len(status['scenes']),image_bytes=total_bytes,
            calibration_scene_overlap=status['calibration_scene_overlap'],
            scene_disjoint_from_calibration=status['scene_disjoint_from_calibration'],
            calibration_file_hash_overlap=0,calibration_decoded_pixel_overlap=0,
            eth_validation_file_hash_overlap=0,eth_validation_decoded_pixel_overlap=0,
            duplicate_pairs=0,max_rotation_orthogonality_error=maximum_error,
            minimum_reference_camera_baseline=min(baselines),zero_reference_baseline_pairs=sum(v<1e-8 for v in baselines),
            pair_manifest_sha256=digest(folder/'pairs.jsonl'),image_manifest_sha256=digest(folder/'images.json'),
            source_url=status.get('source_url'),selection=status.get('selection'),limitation=status.get('limitation'),
            ground_truth_pose=status.get('ground_truth_pose',True),ground_truth_depth=False,
            calibration_manifest_sha256=digest(calibration/'manifest.json'),
            purpose='Frozen held-out validation only, never used for PTQ calibration/QAT/scales; independent of calibration samples, not necessarily original model training scenes')
        write_json(folder/'manifest.json',record)
        all_images.update({name+'/'+p:m for p,m in normalized.items()});records.append(record)
        print(name,'100 pairs',len(images),'images','scene overlap',record['calibration_scene_overlap'],flush=True)
    shared={}
    for group in owners.values():
        if len(group)>1:
            key=' + '.join(sorted(group));shared[key]=shared.get(key,0)+1
    assert not shared,shared
    write_json(args.data/'images.json',all_images)
    (args.data/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in all_pairs))
    (args.data/'pairs.txt').write_text(''.join(f"{p['image0']} {p['image1']}\n" for p in all_pairs))
    snapshot=ROOT/'results/quantization_packages_before.json'
    before=json.loads(snapshot.read_text()) if snapshot.exists() else None
    after={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()}
    if before is None:before=dict(after)
    diff=dict(added={k:v for k,v in after.items() if k not in before},removed={k:v for k,v in before.items() if k not in after},
        changed={k:[v,after[k]] for k,v in before.items() if k in after and v!=after[k]})
    result=dict(state='complete',seed=SEED,datasets=records,completed_datasets=8,total_pairs=len(all_pairs),
        total_images=len(all_images),image_bytes=sum(r['image_bytes'] for r in records),
        calibration_manifest_sha256=digest(calibration/'manifest.json'),eth_manifest_sha256=digest(eth/'manifest.json'),
        pair_manifest_sha256=digest(args.data/'pairs.jsonl'),image_manifest_sha256=digest(args.data/'images.json'),
        calibration_file_hash_overlap=0,calibration_decoded_pixel_overlap=0,calibration_pair_content_overlap=0,
        eth_validation_file_hash_overlap=0,eth_validation_decoded_pixel_overlap=0,
        cross_dataset_shared_image_hashes=shared,unique_content_pairs=len(seen_pairs),
        all_images_decoded=True,all_file_hashes_checked=True,all_camera_matrices_finite=True,package_difference=diff,
        scripts_sha256={n:digest(ROOT/'scripts'/n) for n in ['prepare_multidata_validation.py','validate_multidata_validation.py','sample_data_io.py','prepare_flexglue_samples.py']},
        purpose='Held-out validation only; no model outputs used for pair selection or range fitting')
    assert result['total_pairs']==800
    write_json(args.data/'manifest.json',result);write_json(args.data/'validation.json',result)
    lines=['FlexGlue 多数据集独立验证样本：每数据集100对，共800对','',
        'dataset            pairs images scenes calibration_scene_overlap']
    for r in records:lines.append(f"{r['dataset']:20s}{r['pairs']:5d}{r['images']:7d}{r['scene_count']:7d} {r['calibration_scene_overlap']}")
    lines+=['',f"RGB图像共{result['total_images']}个文件，{result['image_bytes']/1024**2:.1f}MiB。",
        '与既有800对校准集、ETH3D的100对验证集：文件哈希、解码RGB内容均无交集。',
        '800个配对按无序图像内容去重；不同数据集之间也未复用相同图像内容。',
        '原校准集和ETH3D验证集原封不动，未用新验证数据训练、校准、调scale或选择模型。',
        '这里的独立是相对现有PTQ校准样本；不保证场景从未被原始FlexGlue预训练使用。','',
        '场景隔离：7个数据集选新场景/环境。AerialMegaDepth仅公开0001，图像隔离但同场景。',
        'AerialMegaDepth完整数据新场景通过现有授权下载返回HTTP403，没有提交/接受新授权条款。',
        'MegaDepth：官方MegaDepth-1500的0015/0022测试场景，各随机固定50对；并非全部1500对。',
        'TartanAir：官方val两个HDF5分片中100对，覆盖3个新环境；只读取RGB而非整片深度。',
        'ScanNet++：3个新场景的非bad去畸变DSLR候选视对，按轨迹邻近/旋转和基线选对。',
        'Map-free/BlendedMVS：官方共视/MVS配对；Hypersim/DL3DV为近邻轨迹候选，未算稠密重叠。',
        '所有集合都是有参考相机的候选正样本；没有额外构造负样本，不能直接评测负样本误接受率。','',
        '各数据集子目录：images/、metadata/、images.json、pairs.jsonl、pairs.txt、status.json、manifest.json。',
        '根目录清单合并全部800对；路径相对根目录。子目录清单路径相对各子目录。',
        'images.json统一包含OpenCV camera_to_world和K，保留原始来源、文件SHA256和解码RGB哈希。',
        'DL3DV保留原图与OPENCV畸变参数，几何评测须去畸变特征点；不能直接套只接受PINHOLE的旧评测器。',
        '参考位姿来源各异：合成真实相机、SfM重建、ScanNet++激光对齐；未下载深度/逐像素对应GT。',
        'P@极线误差、位姿AUC可用相机参考评估；严格逐像素匹配精度仍需另取深度。','',
        '下载：python -m scripts.prepare_multidata_validation <dataset>',
        '校验：python -m scripts.validate_multidata_validation',
        '下载日志：dataset_sources/eval_<dataset>_prepare.log；HTTP范围缓存和数据源索引仍在dataset_sources。',
        '未安装或替换sfm环境package，未运行模型匹配评测。']
    (args.data/'README.txt').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',result['total_pairs'],'pairs',result['total_images'],'images',result['image_bytes']/1024**2,'MiB',flush=True)


if __name__=='__main__':main()
