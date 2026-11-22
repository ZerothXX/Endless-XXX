"""Official PE-I2I prompt compiler, entirely offline after pinned download.

The official system prompt is unmodified. Sampling settings follow its model
card; NF4 language / BF16 vision is a local hardware adaptation, not the
official BF16 benchmark. Reject schema/canvas errors rather than fix by hand.
"""
import argparse,os,sys,time,json,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from local_multiref import read,write,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--model',type=Path,default=Path(os.environ.get('ENDLESS_PE_I2I_MODEL',ROOT/'models/Qwen-Image-2.1-PE-I2I')))
 p.add_argument('--max-tokens',type=int);p.add_argument('--only',nargs='+')
 p.add_argument('--profile',choices=['qwen','flux3'],default='qwen');a=p.parse_args()
 a.max_tokens=a.max_tokens or (8192 if a.profile=='qwen' else 3584)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');a.out.mkdir(parents=True,exist_ok=True)
 system=((a.model/'system_prompt.txt').read_text(encoding='utf8').strip() if a.profile=='qwen'
  else read(a.model/'system_prompts.json')['i2i'])
 manifest=read(a.model/'download_manifest.json')
 if a.profile=='qwen':assert manifest['complete'],'Incomplete local download'
 else:
  for f in manifest['files']:assert (a.model/f['path']).stat().st_size==f['bytes']
 system_path=a.model/('system_prompt.txt' if a.profile=='qwen' else 'system_prompts.json')
 worker=sha(__file__);(a.out/(worker+'.py')).write_bytes(Path(__file__).read_bytes())
 cases=[c for c in read(a.plan)['cases'] if not a.only or c['id'] in a.only]
 assert cases
 from gpu_slot import gpu_slot
 with gpu_slot():
  import torch
  from PIL import Image,ImageOps
  from transformers import AutoProcessor,AutoModelForImageTextToText,BitsAndBytesConfig
  # Even the Rust pread backend opens the single 18GB PuP container with a
  # Windows mapping on this host. Use bounded exact tensor reads instead.
  import transformers.modeling_utils as model_utils
  from safetensors_pread import PreadSafeOpen
  model_utils.safe_open=PreadSafeOpen
  torch.set_num_threads(4)
  processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,min_pixels=128**2,max_pixels=512**2)
  model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},
   low_cpu_mem_usage=True,attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=['model.visual','visual'])).eval()
  visual_types=sorted({type(m).__name__ for n,m in model.named_modules() if '.visual.' in n})
  assert 'Linear4bit' not in visual_types
  # The first fully resident attempt spilled 1.84GB into WDDM shared memory.
  # Exact BF16 embeddings stay on CPU; the BF16 vision tower is on GPU only
  # for its forward pass. No numerical precision reduction or extra model.
  embedding=model.get_input_embeddings().to('cpu')
  embedding.register_forward_pre_hook(lambda module,args:(args[0].to('cpu'),*args[1:]))
  embedding.register_forward_hook(lambda module,args,output:output.to('cuda'))
  visual=model.model.visual.to('cpu')
  def onload_visual(module,args):module.to('cuda')
  visual.register_forward_pre_hook(onload_visual)
  def offload_visual(module,args,output):
   module.to('cpu');torch.cuda.empty_cache()
  visual.register_forward_hook(offload_visual)
  # Generation may derive position IDs using the CPU embedding's device.
  # The language blocks remain GPU-resident; align all their tensor inputs.
  def align_language(module,args,kwargs):
   from accelerate.utils import send_to_device
   return send_to_device(args,'cuda'),send_to_device(kwargs,'cuda',skip_keys=['past_key_values'])
  model.model.language_model.register_forward_pre_hook(align_language,with_kwargs=True)
  torch.cuda.empty_cache()
  write(a.out/'load.json',{'model':str(a.model),'revision':manifest['revision'],'precision':'NF4 language including untied lm_head; BF16 vision and input embedding',
   'visual_types':visual_types,'lm_head_class':type(model.lm_head).__name__,'model_parameters':sum(p.numel() for p in model.parameters()),
   'system_prompt_sha256':sha(system_path),'worker_sha256':worker,'online_inference':False,
   'weight_io':'bounded exact safetensors pread','weight_io_worker_sha256':sha(Path(__file__).parent/'safetensors_pread.py'),
   'placement':'BF16 input embedding on CPU; BF16 vision tower staged to GPU only for its forward; NF4 language including lm_head on GPU'})
  compiled=[]
  for c in cases:
   paths=[c['source'],*c['references']]
   binding={'case':c,'inputs':[{'path':x,'sha256':sha(x)} for x in paths],'model_manifest_sha256':sha(a.model/'download_manifest.json'),
    'system_sha256':sha(system_path),'worker_sha256':worker,'max_new_tokens':a.max_tokens,'sampling_seed':12345,'profile':a.profile,
    'placement':'cpu_embedding_staged_visual'}
   dest=a.out/(c['id']+'.json')
   if dest.exists():
    r=read(dest);assert r['binding']==binding,'Stale rewrite cache'
   else:
    content=[]
    for path in paths:
     with Image.open(path) as raw:
      im=ImageOps.exif_transpose(raw).convert('RGBA');im=Image.alpha_composite(Image.new('RGBA',im.size,'white'),im).convert('RGB')
      content.append({'type':'image','image':im.copy()})
    request=c['instruction'] if a.profile=='qwen' else 'Reference images follow in order (the first image, the second image, ...).\nUser request: '+c['instruction']+'\nOutput the caption.'
    content.append({'type':'text','text':request})
    messages=[{'role':'system','content':[{'type':'text','text':system}]},{'role':'user','content':content}]
    start=time.monotonic();torch.manual_seed(12345);torch.cuda.reset_peak_memory_stats()
    inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,
     return_tensors='pt',enable_thinking=a.profile=='qwen').to('cuda')
    with torch.inference_mode():
     from transformers.generation.streamers import BaseStreamer
     class ProgressStreamer(BaseStreamer):
      def __init__(self):self.count=0;self.prompt=True
      def put(self,value):
       if self.prompt:self.prompt=False;return
       self.count+=value.numel()
       if self.count%64==0:
        write(a.out/'progress.json',{'id':c['id'],'generated_tokens':self.count,'seconds':time.monotonic()-start,
         'cuda_allocated':torch.cuda.memory_allocated(),'cuda_reserved':torch.cuda.memory_reserved()})
        print('PE progress',c['id'],self.count,round(time.monotonic()-start,1),flush=True)
      def end(self):pass
     sampling={'do_sample':True,'temperature':1.0,'top_p':.95,'top_k':20} if a.profile=='qwen' else {'do_sample':False}
     output=model.generate(**inputs,max_new_tokens=a.max_tokens,**sampling,streamer=ProgressStreamer())
    tokens=output[0,inputs['input_ids'].shape[1]:];raw=processor.tokenizer.decode(tokens,skip_special_tokens=True)
    _,sep,answer=raw.partition('</think>');parsed=None;error=None
    try:
     if a.profile=='flux3':
      caption=raw.strip()
      if not caption or len(caption)<100 or '<think>' in caption or '```' in caption:raise ValueError('Invalid or empty official PuP caption')
      parsed={'rewritten_prompt':caption,'ratio_follow':'<image1>','wh_ratio':''}
     else:
      if not sep:raise ValueError('No completed thinking boundary')
      parsed=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',answer.strip()))
     if not isinstance(parsed,dict) or not isinstance(parsed.get('rewritten_prompt'),str):raise ValueError('Invalid output schema')
     if parsed.get('ratio_follow')!='<image1>' or parsed.get('wh_ratio') not in ('',None):raise ValueError('Rewriter changed the source canvas')
     mentioned={int(n) for n in re.findall(r'<image(\d+)>',parsed['rewritten_prompt'])}
     if a.profile=='qwen' and (1 not in mentioned or not mentioned.issubset(set(range(1,len(paths)+1)))):raise ValueError('Invalid image slot binding')
    except (ValueError,TypeError) as exc:error=str(exc)
    r={'binding':binding,'raw':raw,'parsed':parsed,'error':error,'generated_tokens':len(tokens),'truncated':len(tokens)>=a.max_tokens,
     'seconds':time.monotonic()-start,'online_inference':False,'quality_accepted':False,'cuda_peak_allocated':torch.cuda.max_memory_allocated()}
    write(dest,r);print('rewritten',c['id'],'tokens',len(tokens),'error',error,flush=True)
    del inputs,output,tokens;torch.cuda.empty_cache()
   if r['error'] or r['truncated']:continue
   compiled.append({**c,'id':c['id']+'_pe','instruction':r['parsed']['rewritten_prompt'],
    'instruction_origin':{'kind':'unmodified official '+('PE-I2I' if a.profile=='qwen' else 'FLUX3 PuP')+' local output','record':str(dest.resolve()),'record_sha256':sha(dest),
     'precision':'NF4 language/BF16 vision','quality_accepted':False}})
  write(a.out/'plan.json',{'cases':compiled,'quality_accepted':False,'online_inference':False})

if __name__=='__main__':main()
