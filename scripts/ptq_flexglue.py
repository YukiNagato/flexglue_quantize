"""Real-data torchao static W8A8 calibration for the local FlexGlue matcher.

The calibration artifact contains only min/max observers and fixed scales.
Validation re-creates quantized weights from the original checkpoint and never
updates these observers. INT8 and FP8 share the same observed floating-point run.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn

from scripts.inference import ROOT,Config,FeatureExtractor,Matcher,provenance,synchronize
from scripts.sample_data_io import write_json


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible(module,fqn=''):
    return isinstance(module,nn.Linear) and module.in_features%16==0 and module.out_features%16==0


def coverage(model):
    return [dict(name=n,shape=list(m.weight.shape),weight_type=type(m.weight).__name__,dtype=str(m.weight.dtype),
                 quantized=type(m.weight) not in (nn.Parameter,torch.Tensor))
            for n,m in model.named_modules() if isinstance(m,nn.Linear)]


def apply_static_ptq(model,artifact,kind):
    """Apply frozen calibration statistics with native torchao APIs, before compile."""
    from torchao.quantization import (quantize_,Float8StaticActivationFloat8WeightConfig as F8,
                                     Int8StaticActivationInt8WeightConfig as I8)
    from torchao.quantization.granularity import PerTensor,PerRow
    artifact=Path(artifact)
    info=json.loads((artifact/'calibration.json').read_text())
    if info['state']!='complete':raise ValueError('Calibration incomplete')
    state=torch.load(artifact/'observers.pt',weights_only=True)
    if digest(artifact/'observers.pt')!=info['observers_sha256']:raise ValueError('Calibration artifact changed')
    targets=set(info['target_modules'])
    for name in targets:
        module=model.get_submodule(name)
        assert list(module.weight.shape)==info['statistics'][name]['weight_shape']
    if kind=='fp8-static':
        quantize_(model,F8(step='prepare',set_inductor_config=False),filter_fn=lambda m,n:n in targets)
        for name in targets:
            module=model.get_submodule(name);obs=module.input_act_obs
            obs.min_val=state[name]['min_val'].to(module.weight.device)
            obs.max_val=state[name]['max_val'].to(module.weight.device)
        quantize_(model,F8(step='convert',set_inductor_config=False),filter_fn=lambda m,n:n in targets)
        for name in targets:
            module=model.get_submodule(name)
            # torch._scaled_mm requires Half/BF16 fused bias for Half output.
            if module.bias is not None:module.bias=nn.Parameter(module.bias.half(),requires_grad=False)
    elif kind=='int8-static':
        for name in sorted(targets):
            module=model.get_submodule(name)
            cfg=I8(act_quant_scale=state[name]['int8_scale'].to(module.weight.device),
                   granularity=(PerTensor(),PerRow(-1)),version=2,set_inductor_config=False)
            quantize_(module,cfg)
    else:raise ValueError(kind)
    model.eval()
    result=coverage(model)
    assert sum(row['quantized'] for row in result)==len(targets)
    assert not any(hasattr(m,'input_act_obs') for m in model.modules()),'Observers must not run during validation'
    return dict(kind=kind,artifact=str(artifact.resolve()),artifact_sha256=info['observers_sha256'],
        calibration_pair_count=info['calibration_pairs'],calibration_dataset_manifest_sha256=info['calibration_manifest_sha256'],
        linear_coverage=result,activation_granularity='PerTensor (fixed scalar, input-rank broadcast shape)',
        weight_granularity='PerTensor' if kind=='fp8-static' else 'PerRow',
        fp8_bias_adapter='FP16 bias for FP8 linears' if kind=='fp8-static' else None)


class FeatureStore:
    def __init__(self,cfg,data,images,report):
        self.cfg,self.data,self.images=cfg,Path(data),images
        config=dict(max_keypoints=cfg.max_keypoints,detection_resize=cfg.detection_resize,feature_resize=cfg.feature_resize,
            weights_sha256=report['weights_sha256'],images_sha256=digest(self.data/'images.json'),torch=str(torch.__version__),
            sources={str(f.relative_to(ROOT)):digest(f) for f in [ROOT/'scripts/inference.py',ROOT/'flexglue_quantize/descriptor.py',*sorted((ROOT/'scripts/raco').glob('*.py'))]})
        self.fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        self.cache=ROOT/'results'/('ptq_calibration_features_'+self.fingerprint[:16]);self.cache.mkdir(parents=True,exist_ok=True)
        write_json(self.cache/'manifest.json',config)
        self.loaded=OrderedDict()
    def path(self,name):return self.cache/(hashlib.sha256(name.encode()).hexdigest()+'.pt')
    def prepare(self):
        missing=[n for n in sorted(self.images) if not self.path(n).exists()]
        if not missing:return
        extractor=FeatureExtractor(self.cfg)
        for i,name in enumerate(missing,1):
            assert digest(self.data/name)==self.images[name]['sha256']
            feature=extractor.extract(self.data/name)
            path=self.path(name);temp=path.with_suffix('.tmp');torch.save(feature,temp);temp.replace(path)
            if i%50==0 or i==len(missing):print('CALIB FEATURES',i,'/',len(missing),flush=True)
        del extractor;synchronize(self.cfg.device);torch.cuda.empty_cache()
    def get(self,name):
        feature=self.loaded.pop(name,None)
        if feature is None:feature=torch.load(self.path(name),weights_only=True)
        self.loaded[name]=feature
        while len(self.loaded)>64:self.loaded.popitem(last=False)
        return feature


@torch.inference_mode()
def calibrate(args):
    from torchao.quantization import quantize_,Float8StaticActivationFloat8WeightConfig as F8
    from torchao.quantization.observer import AffineQuantizedMinMaxObserver
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quant_primitives import MappingType
    from flexglue_quantize.utils import ops
    import torchao
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0)
    cfg=Config(weights=str(args.weights), detector_weights=str(args.detector_weights), device=args.device)
    args.output.mkdir(parents=True,exist_ok=False)
    pairs=[json.loads(line) for line in (args.data/'pairs.jsonl').read_text().splitlines()]
    images=json.loads((args.data/'images.json').read_text())
    if not pairs:
        raise ValueError('Calibration requires at least one pair')
    validation=json.loads((args.validation_data/'images.json').read_text()) if args.validation_data else {}
    assert not ({m['sha256'] for m in validation.values()}&{m['sha256'] for m in images.values()})
    report=provenance(cfg)
    report.update(state='extracting',torchao=torchao.__version__,calibration_pairs=len(pairs),calibration_images=len(images),
        calibration_manifest_sha256=digest(args.data/'manifest.json'),pairs_sha256=digest(args.data/'pairs.jsonl'),
        images_sha256=digest(args.data/'images.json'),calibration_data=str(args.data.resolve()),
        validation_images_used=0,source_sha256_ptq=digest(__file__),
        scope='Floating matcher, no gradients, FlashAttention FP16, no early exit. Each pair runs all nine layers; all heads 0..9 run in both directions. Covisibility branches are observed in FP32 and the early-exit FP16 autocast context. Native FP8 prepare observers and torchao INT8 min/max observers collect the same inputs. No statistics from validation.',
        config=asdict(cfg))
    def save():write_json(args.output/'calibration.json',report)
    save()
    try:
        store=FeatureStore(cfg,args.data,images,report)
        report.update(feature_cache=str(store.cache),feature_fingerprint=store.fingerprint);save()
        t=time.perf_counter();store.prepare();report['feature_preparation_seconds']=time.perf_counter()-t
        owner=Matcher(cfg);model=owner.model
        targets=[name for name,module in model.named_modules() if eligible(module)]
        report['target_modules']=targets
        report['skipped_modules']=[dict(name=n,shape=list(m.weight.shape),reason='Tensor-core dimension alignment') for n,m in model.named_modules() if isinstance(m,nn.Linear) and not eligible(m)]
        quantize_(model,F8(step='prepare',set_inductor_config=False),filter_fn=eligible)
        observers={};handles=[];counts={};ranks={};shapes={}
        for name,module in model.named_modules():
            if not isinstance(module,nn.Linear):continue
            observer=AffineQuantizedMinMaxObserver(mapping_type=MappingType.SYMMETRIC,target_dtype=torch.int8,
                granularity=PerTensor(),eps=torch.finfo(torch.float32).eps,scale_dtype=torch.float32,
                zero_point_dtype=torch.int64,keepdim=True)
            observers[name]=observer;counts[name]=0;ranks[name]=set();shapes[name]=list(module.weight.shape)
            def hook(m,inputs,name=name,observer=observer):
                x=inputs[0]
                if not x.numel():return
                observer(x);counts[name]+=1;ranks[name].add(x.ndim)
            handles.append(module.register_forward_pre_hook(hook))
        report.update(state='observing',observed_modules=len(observers));save()
        head_counts={str(k):0 for k in range(10)};zero_points=0;t=time.perf_counter()
        for i,pair in enumerate(pairs,1):
            f0,f1=[store.get(pair[k]) for k in ['image0','image1']]
            data0,data1=[owner.input(f) for f in (f0,f1)]
            x0,x1,e0,e1=model.encode_pair(data0['features'],data1['features'])
            hw=(data0['features'].shape[-2:],data1['features'].shape[-2:])
            points=[ops.normalize_keypoints(d['keypoints'],d['image_size']) for d in (data0,data1)]
            for k in range(10):
                if k:
                    with torch.autocast('cuda',dtype=torch.float16):x0,x1=model.transformers[k-1](x0,x1,e0,e1)
                context=model.pair_context(x0,x1,hw,k)
                for direction in (0,1):
                    if points[direction].shape[1]:
                        out=model.heads[k](points[direction],*context.direction(direction));del out
                    else:zero_points+=1
                # The original early-exit score runs inside the transformer autocast.
                if 0<k<9:
                    with torch.autocast('cuda',dtype=torch.float16):score=model.covisibility_score(x0,x1,k)
                    del score
                head_counts[str(k)]+=1
            del context,x0,x1,e0,e1
            if i%50==0 or i==len(pairs):
                synchronize(cfg.device);report.update(processed_pairs=i,head_pair_counts=head_counts,observer_seconds=time.perf_counter()-t);save()
                print('CALIB PAIRS',i,'/',len(pairs),'elapsed',report['observer_seconds'],flush=True)
        for h in handles:h.remove()
        assert all(counts.values()),{n:c for n,c in counts.items() if not c}
        assert all(len(s)==1 for s in ranks.values()),ranks
        state={};statistics={}
        for name,obs in observers.items():
            minimum,maximum=obs.min_val.cpu(),obs.max_val.cpu()
            assert torch.isfinite(minimum).all() and torch.isfinite(maximum).all()
            int8_scale=obs.calculate_qparams()[0].cpu()
            fp8_scale=None
            if name in targets:
                native=model.get_submodule(name).input_act_obs
                assert torch.equal(native.min_val.cpu(),minimum) and torch.equal(native.max_val.cpu(),maximum)
                fp8_scale=native.calculate_qparams()[0].cpu()
            state[name]=dict(min_val=minimum,max_val=maximum,int8_scale=int8_scale,fp8_scale=fp8_scale)
            statistics[name]=dict(min=float(minimum.flatten()[0]),max=float(maximum.flatten()[0]),calls=counts[name],
                input_rank=next(iter(ranks[name])),weight_shape=shapes[name],
                int8_scale=float(int8_scale.flatten()[0]),fp8_scale=float(fp8_scale.flatten()[0]) if fp8_scale is not None else None)
        torch.save(state,args.output/'observers.pt')
        report.update(state='complete',statistics=statistics,head_pair_counts=head_counts,zero_keypoint_directions=zero_points,
            observers_sha256=digest(args.output/'observers.pt'),
            activation_calibration=f'Unclipped min/max over all {len(pairs)} pairs, PerTensor; native torchao symmetric qparams',
            int8_weight_granularity='PerRow(-1)',fp8_weight_granularity='PerTensor',
            source_checkpoint_sha256=digest(cfg.weights),target_count=len(targets))
        save();print('CALIBRATION COMPLETE',len(targets),'eligible linears;',len(observers),'observed',flush=True)
    except Exception:
        report.update(state='failed',error=traceback.format_exc());save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/flexglue_ptq_100')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=ROOT/'weights/flexglue.pth')
    parser.add_argument('--detector-weights',type=Path,default=ROOT/'weights/raco.pth')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--validation-data',type=Path,help='Optional held-out manifest; reject overlapping images')
    calibrate(parser.parse_args())
