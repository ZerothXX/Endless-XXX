"""Align bitsandbytes quantization metadata after block-level weight offloading.

Diffusers can move parameter data without moving its nested quantization state.
These hooks copy metadata without requantizing or changing any numeric values.
"""
from pathlib import Path
import hashlib,json

def align_quant_state(model,out):
    import bitsandbytes as bnb
    source=Path(__file__);digest=hashlib.sha256(source.read_bytes()).hexdigest()
    out=Path(out);frozen=out/'workers'/(digest+'.py');frozen.parent.mkdir(parents=True,exist_ok=True)
    if not frozen.exists():frozen.write_bytes(source.read_bytes())
    manifest=out/'dependencies.json'
    dependencies=json.loads(manifest.read_text('utf8')) if manifest.exists() else {}
    dependencies['local_quant_offload.py']={'snapshot':str(frozen.resolve()),'sha256':digest}
    manifest.write_text(json.dumps(dependencies,indent=2),'utf8')
    handles=[]
    def align(module,inputs):
        device=inputs[0].device
        state=getattr(module.weight,'quant_state',None) or module.quant_state
        state.to(device)
        module.quant_state=state;module.weight.quant_state=state
    for module in model.modules():
        if isinstance(module,bnb.nn.Linear4bit):handles.append(module.register_forward_pre_hook(align))
    return handles
