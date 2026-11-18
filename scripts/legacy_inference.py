"""Historical loader for existing root-level backend ablations.

New applications must use flexglue_quantize. This compatibility implementation
retains the original process-global experimental backends and source paths.
"""
import torch
from torch import nn
from flexglue_quantize.linear import FP8Linear

@torch.inference_mode()
def load_best_ptq(checkpoint, device='cuda', compile=True, early_stop_decay=None, cute_preprocess='none',head_backend='none',early_exit_backend='none'):
    """Load packed FP8 matcher weights without the original float checkpoint.

    Input remains the original FlexGlue feature/keypoint dictionaries. Image
    feature extraction is separate. Attention installation is process-global;
    use this entry point once in a fresh process.
    cute_preprocess optionally selects editable CuTe prologues ('qk' or 'qkv').
    head_backend='cute-precise' fuses sorting/temperature/window gathering while
    retaining native entmax arithmetic. Small FP32/tie differences remain;
    'none' preserves the original head. All custom backends are inference-only.
    This changes process-global Sage hooks; compare modes in fresh processes.
    early_exit_backend='cute' fuses residual/FP8 covisibility scoring and the
    threshold comparison at each checked block, retaining native host decisions.
    'compiled' compiles the block and existing scoring ops together; 'none' is
    unchanged. These affect only intermediate early checks, not the final score.
    Confirmed early rejection now returns invalid matches without running heads
    or nearest-neighbor search. For discarded auxiliary-coordinate diagnostics,
    use model.match(..., skip_head_on_early_exit=False) to retain the old path.
    """
    if cute_preprocess not in ('none','qk','qkv'):
        raise ValueError(f'Unknown CuTe preprocess mode: {cute_preprocess}')
    if early_exit_backend not in ('none','compiled','cute'):
        raise ValueError(f'Unknown early-exit backend: {early_exit_backend}')
    from local_models.flexglue import FlexGlue
    from benchmark_quantization_synthetic import install_attention
    packed = torch.load(checkpoint, weights_only=True)
    assert packed['format'] == 'flexglue-best-ptq-v1'
    conf = dict(packed['conf'])
    conf['early_stop_decay'] = early_stop_decay
    model = FlexGlue(**conf).eval().to(device)
    targets = {row['name'] for row in packed['ptq']['linear_coverage'] if row['quantized']}
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for name, module in linears:
        if name not in targets:
            continue
        replacement = FP8Linear(module, torch.ones((), device=device))
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, replacement)
    model.load_state_dict(packed['state_dict'], strict=True)
    model.eval()
    install_attention('sage', layout='HND', granularity='per_thread', smooth_k=True)
    if cute_preprocess != 'none':
        from cute_attention_preprocess import install_qk,install_v
        install_qk()
        if cute_preprocess == 'qkv':
            install_v()
    if head_backend != 'none':
        from fused_match_head import install
        install(model,head_backend)
    if early_exit_backend != 'none':
        from cute_early_exit import install
        install(model,early_exit_backend)
    if compile:
        model.compile(mode='reduce-overhead')
    return model

