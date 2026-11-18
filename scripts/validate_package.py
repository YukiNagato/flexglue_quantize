"""Compare the installed inference package with saved pre-migration predictions."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from flexglue_quantize import FlexGlue


def gpu_input(feature):
    h,w=feature['image_size']
    return dict(features=feature['features'][None].cuda(),
                keypoints=feature['keypoints'][None].cuda(),
                image_size=torch.tensor([[w,h]],device='cuda',dtype=torch.float32))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--pairs',type=Path,required=True,help='Courtyard folder containing pairs.txt and features/')
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--eager',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from torch._inductor import config
    config.compile_threads=4
    # Instantiate on CPU then .cuda(), as users of the original FlexGlue do.
    model=FlexGlue(weights=args.checkpoint, cute_preprocess='qkv', head_backend='cute-precise',
                   early_exit_backend='cute', min_covisibility=.9, early_stop_decay=.5).eval().cuda()
    if not args.eager:model.compile()
    reference=torch.load(args.reference,weights_only=True)
    pairs=np.loadtxt(args.pairs/'pairs.txt',dtype=int).reshape(-1,2)
    if args.limit:pairs=pairs[:args.limit]
    inputs={int(i):gpu_input(torch.load(args.pairs/'features'/f'{i}.pt',weights_only=True)) for i in np.unique(pairs)}
    report=dict(state='running',compiled=not args.eager,model_module=type(model).__module__,cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for mode,decay,ref_key in [('full',None,'noexit'),('early',.5,'cute')]:
        layers=Counter()
        for count,(i,j) in enumerate(pairs,1):
            key=f'{i}_{j}'
            out=model({'image0':inputs[int(i)],'image1':inputs[int(j)]},sparse=True,early_stop_decay=decay)
            expected=reference[ref_key][key]
            exact={name:torch.equal(out[name].cpu(),expected[name]) for name in ('matches0','matching_scores0','covisibility_score')}
            exact['stop_layer']=int(out['stop_layer'])==expected['stop_layer']
            if not all(exact.values()):
                report['failure']=dict(mode=mode,pair=key,exact=exact);save()
                raise AssertionError(report['failure'])
            layers[int(out['stop_layer'])]+=1
            report['cases'].append(dict(mode=mode,pair=key,stop_layer=int(out['stop_layer']),exact=True))
            del out
            if count%100==0:save();print(mode,count,flush=True)
        report[mode]=dict(pairs=len(pairs),layers=dict(layers),all_outputs_exact=True)
        print('PASS',mode,report[mode],flush=True);save()
    report['state']='complete';save()


if __name__=='__main__':main()
