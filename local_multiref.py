"""Offline multi-reference editor. Staged encoders fit consumer GPUs; no cloud fallback.

Run with the local model environment selected by ENDLESS_PYTHON. A plan contains source,
references, instruction and output size; cases and roles never select code paths.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path(os.environ.get('ENDLESS_QWEN_IMAGE_MODEL', ROOT / 'models/Qwen-Image-2.1'))

def read(p): return json.loads(Path(p).read_text('utf8'))
def write(p, v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
WORKER_SHA=sha(__file__)
def images(case):
    from PIL import Image,ImageOps
    result=[]
    order=case.get('condition_order','source_then_references')
    if order not in ('source_then_references','references_then_source'):raise ValueError('Unknown condition order')
    paths=([case['source'],*case['references']] if order=='source_then_references' else [*case['references'],case['source']])
    for path in paths:
        with Image.open(path) as raw:
            im=ImageOps.exif_transpose(raw).convert('RGBA');result.append(im.copy())
    return result
def binding(case,args):
    result={'case':case,'inputs':[{'path':p,'sha256':sha(p)} for p in [case['source'],*case['references']]],
        'reference_area_side':args.reference_resolution,'model':str(args.model.resolve()),
        'encoder_precision':args.encoder_precision}
    if args.encoder_vision_bf16:result['encoder_vision_bf16']=True
    return result
def metrics(torch):
    return {'cuda_peak_allocated':torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved':torch.cuda.max_memory_reserved()}

def enable_cpu_prefix_cache():
    """Keep exact, unquantized prefix KV on CPU; copy only the active layer back."""
    import diffusers.pipelines.qwenimage21.pipeline_qwenimage21 as pipeline_module
    class CpuLayerCache:
        def __init__(self):self.k=None;self.v=None;self.device=None
        def store(self,k,v):
            self.device=k.device
            self.k=k.to('cpu',copy=True);self.v=v.to('cpu',copy=True)
        def get(self):
            if self.k is None:raise RuntimeError('Prefix cache has not been populated')
            return self.k.to(self.device),self.v.to(self.device)
    class CpuPrefixCache:
        def __init__(self,num_layers):self.layer_caches=[CpuLayerCache() for _ in range(num_layers)]
        def get_layer(self,index):return self.layer_caches[index]
    pipeline_module.QwenImage21KVCache=CpuPrefixCache

def encode(cases,args,out,torch):
    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    from transformers import Qwen3VLProcessor,Qwen3VLForConditionalGeneration,BitsAndBytesConfig
    missing=[]
    for c in cases:
        path=out/'embeddings'/(c['id']+'.pt')
        if path.exists():
            assert torch.load(path,weights_only=True)['binding']==binding(c,args),'Existing cache belongs to different inputs or prompt'
        else:missing.append(c)
    if not missing:return
    print('Loading local vision-language encoder',args.encoder_precision,flush=True)
    processor=Qwen3VLProcessor.from_pretrained(args.model/'processor',local_files_only=True)
    kw={'local_files_only':True,'dtype':torch.bfloat16,'low_cpu_mem_usage':True}
    if args.encoder_precision=='nf4':
        kw.update(device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=['model.visual','visual','lm_head'] if args.encoder_vision_bf16 else None))
    elif args.encoder_disk_offload:
        # Keep loading itself bounded; CPU-only loading first would retain all
        # 8B BF16 weights in system RAM before installing offload hooks.
        kw.update(device_map='auto',max_memory={0:'2GiB','cpu':'2GiB'},
            offload_folder=str(args.encoder_disk_offload.resolve()),offload_state_dict=True)
    else:kw['device_map']={'':'cpu'}
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model/'text_encoder',**kw).eval()
    visual_types=sorted({type(m).__name__ for n,m in model.named_modules() if '.visual.' in n})
    if args.encoder_vision_bf16 and 'Linear4bit' in visual_types:
        raise RuntimeError('Visual precision exclusion failed')
    if args.encoder_precision=='bf16' and not args.encoder_disk_offload:
        from accelerate import cpu_offload
        cpu_offload(model,execution_device=torch.device('cuda'),offload_buffers=True)
    pipe=QwenImage21Pipeline(scheduler=None,vae=None,transformer=None,text_encoder=model,processor=processor)
    write(out/'encoder_load.json',{'precision':args.encoder_precision,'vision_bf16':args.encoder_vision_bf16,
        'visual_module_types':visual_types,'parameters':sum(p.numel() for p in model.parameters()),
        'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
        'linear4bit_modules':sum(type(m).__name__=='Linear4bit' for m in model.modules()),
        'device_map':getattr(model,'hf_device_map',None),
        'disk_offload':str(args.encoder_disk_offload.resolve()) if args.encoder_disk_offload else None,**metrics(torch)})
    for c in missing:
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();prepared=[]
        for im in images(c):
            w,h,_=calculate_dimensions(args.reference_resolution**2,im.width/im.height)
            prepared.append(pipe.image_processor.resize(im,width=w,height=h))
        e,m,im=pipe.encode_prompt(prompt=c['instruction'],image=prepared,device=torch.device('cuda'))
        assert torch.isfinite(e).all()
        cache={'binding':binding(c,args),'embedding':e.cpu(),'mask':None if m is None else m.cpu(),'image_mask':im.cpu(),
            'preprocessed_sizes':[x.size for x in prepared]}
        p=out/'embeddings'/(c['id']+'.pt');p.parent.mkdir(exist_ok=True)
        torch.save(cache,p)
        write(out/'encodings'/(c['id']+'.json'),{'seconds':time.monotonic()-start,'shape':list(e.shape),
            'image_slots':int(im.sum()),'preprocessed_sizes':cache['preprocessed_sizes'],**metrics(torch)})
        print('encoded',c['id'],list(e.shape),flush=True)
        del e,m,im,cache;torch.cuda.empty_cache()

def render(cases,args,out,torch):
    from diffusers import (QwenImage21Pipeline,QwenImage21Transformer2DModel,AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,BitsAndBytesConfig)
    from transformers import Qwen3VLProcessor
    if args.kv_cache_cpu:enable_cpu_prefix_cache()
    class CachedPipeline(QwenImage21Pipeline):
        source_init=None
        def prepare_latents(self,images,batch_size,num_channels_latents,height,width,dtype,device,generator,latents=None):
            noise,conditions=super().prepare_latents(images,batch_size,num_channels_latents,height,width,dtype,device,generator,latents)
            if self.source_init is None:return noise,conditions
            source=self.image_processor.preprocess(self.source_init['image'],width=width,height=height).unsqueeze(2).to(device=device,dtype=self.vae.dtype)
            z=self._encode_vae_image(source,generator)
            z=self._pack_latents(z,batch_size,num_channels_latents,z.shape[3],z.shape[4]).to(dtype=dtype)
            if z.shape!=noise.shape:raise ValueError('Source initialization latent shape mismatch')
            sigma=self.source_init['sigma'];mixed=(1-sigma)*z+sigma*noise
            if not torch.isfinite(mixed).all():raise ValueError('Nonfinite source initialization')
            return mixed,conditions
        def encode_prompt(self,prompt,image=None,device=None,num_images_per_prompt=1,
                          prompt_embeds=None,prompt_embeds_mask=None,image_pad_mask=None):
            assert prompt is None and num_images_per_prompt==1 and prompt_embeds is not None
            mask=self.cached['mask']
            return prompt_embeds,None if mask is None else mask.to(device),self.cached['image_mask'].to(device)
    for c in cases:
        for seed in args.seeds:
            existing=out/'records'/f"{c['id']}_s{seed}_{args.precision}.json"
            if not existing.exists():continue
            r=read(existing)
            assert r['binding']==binding(c,args) and r['steps']==args.steps and r['kv_cache']==args.kv_cache,'Existing result uses different inputs or sampling'
            assert r.get('trim_cache',False)==args.trim_cache,'Existing result uses a different allocator policy'
            assert r.get('kv_cache_cpu',False)==args.kv_cache_cpu,'Existing result uses different KV placement'
            assert r.get('cpu_group_offload',r['precision']=='bf16')==(args.cpu_offload or args.precision=='bf16'),'Existing result uses different weight placement'
            assert r.get('source_init_requested_sigma')==args.source_init_sigma,'Existing result uses different source initialization'
            assert r.get('vae_reference_mode','all')==args.vae_reference_mode,'Existing result uses different reconstructive conditions'
            assert r.get('vae_condition_indices')==args.vae_condition_indices,'Existing result uses different selected image slots'
            assert r.get('vae_decode_tiling',False)==bool(args.vae_decode_tile_size),'Existing result uses a different decode policy'
            assert r.get('vae_decode_tile_size',0)==args.vae_decode_tile_size,'Existing result uses a different decode tile size'
            assert sha(r['output'])==r['sha256'],'Existing image changed'
    remaining=[c for c in cases if any(not (out/'records'/f"{c['id']}_s{s}_{args.precision}.json").exists() for s in args.seeds)]
    if not remaining:return
    start=time.monotonic()
    kw={'local_files_only':True,'torch_dtype':torch.bfloat16,'low_cpu_mem_usage':True}
    if args.precision=='nf4':
        kw.update(device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    else:kw['device_map']={'':'cpu'}
    transformer=QwenImage21Transformer2DModel.from_pretrained(args.model/'transformer',**kw).eval()
    if args.precision=='bf16' or args.cpu_offload:
        transformer.enable_group_offload(onload_device=torch.device('cuda'),offload_device=torch.device('cpu'),
            offload_type='block_level',num_blocks_per_group=1,use_stream=not args.synchronous_offload,
            record_stream=not args.synchronous_offload,low_cpu_mem_usage=True,
            offload_to_disk_path=str(args.offload_disk.resolve()) if args.offload_disk else None)
        if args.precision=='nf4':
            from local_quant_offload import align_quant_state
            quant_handles=align_quant_state(transformer,out)
    vae=AutoencoderKLQwenImage21.from_pretrained(args.model/'vae',local_files_only=True,torch_dtype=torch.bfloat16).eval().to('cuda')
    vae.disable_tiling()
    if args.vae_decode_tile_size:
        native_decode=vae.decode
        def tiled_decode(*decode_args,**decode_kwargs):
            size=args.vae_decode_tile_size;stride=size*3//4
            vae.enable_tiling(tile_sample_min_height=size,tile_sample_min_width=size,
                tile_sample_stride_height=stride,tile_sample_stride_width=stride)
            try:return native_decode(*decode_args,**decode_kwargs)
            finally:vae.disable_tiling()
        vae.decode=tiled_decode
    pipe=CachedPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.model/'scheduler',local_files_only=True),
        vae=vae,transformer=transformer,text_encoder=None,
        processor=Qwen3VLProcessor.from_pretrained(args.model/'processor',local_files_only=True))
    write(out/f'generator_load_{args.precision}.json',{'seconds':time.monotonic()-start,'precision':args.precision,
        'model':str(args.model),'vae_tiling':False,'offline_only':True,**metrics(torch)})
    for c in remaining:
        cached=torch.load(out/'embeddings'/(c['id']+'.pt'),weights_only=True)
        assert cached['binding']==binding(c,args),'Stale prompt/image cache'
        vae_images=images(c);routing_record=None
        if args.vae_reference_mode in ('source-only','selected'):
            # Keep every contextual text embedding from the all-image VLM pass,
            # but remove reference reconstruction slots and their VAE latents.
            # This is an experimental distribution change, not a native option.
            positions=cached['image_mask'][0].nonzero().flatten()
            boundaries=(positions[1:]-positions[:-1]>1).nonzero().flatten()+1
            runs=torch.tensor_split(positions,boundaries.tolist())
            if len(runs)!=len(vae_images):raise ValueError('Cannot identify condition image slot blocks')
            source_index=0 if c.get('condition_order','source_then_references')=='source_then_references' else len(vae_images)-1
            selected=([source_index] if args.vae_reference_mode=='source-only' else args.vae_condition_indices)
            if not selected or source_index not in selected or min(selected)<0 or max(selected)>=len(runs):
                raise ValueError('Selected reconstructive conditions must include the source and valid unique image indices')
            selected=sorted(set(selected));keep=~cached['image_mask'][0].bool()
            for index in selected:keep[runs[index]]=True
            cached={**cached,'embedding':cached['embedding'][:,keep],
                'image_mask':cached['image_mask'][:,keep],
                'mask':None if cached['mask'] is None else cached['mask'][:,keep]}
            vae_images=[vae_images[index] for index in selected]
            routing_record={'vlm_context':'all original conditions retained in contextual text embeddings',
                'vae_conditions':'selected original condition blocks','selected_condition_indices':selected,
                'removed_reference_image_blocks':len(runs)-len(selected),
                'source_slot_index':source_index,'kept_sequence_length':int(keep.sum()),
                'scope':'experimental condition ablation; not a trained disentanglement mechanism'}
        pipe.cached=cached
        schedule_kwargs={};source_init_record=None;pipe.source_init=None
        if args.source_init_sigma is not None:
            import numpy as np
            from PIL import Image,ImageOps
            from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
            raw=np.linspace(1.,1/args.steps,args.steps)
            reference_scheduler=FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            mu=calculate_shift((c['height']//pipe.vae_scale_factor)*(c['width']//pipe.vae_scale_factor),
                pipe.scheduler.config.base_image_seq_len,pipe.scheduler.config.max_image_seq_len,
                pipe.scheduler.config.base_shift,pipe.scheduler.config.max_shift)
            reference_scheduler.set_timesteps(sigmas=raw,mu=mu,device='cpu')
            effective=reference_scheduler.sigmas[:-1].cpu().numpy();index=int(np.argmin(abs(effective-args.source_init_sigma)))
            selected=raw[index:];check_scheduler=FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            check_scheduler.set_timesteps(sigmas=selected,mu=mu,device='cpu')
            sigma=float(check_scheduler.sigmas[0]);assert abs(sigma-float(effective[index]))<1e-5
            with Image.open(c['source']) as source:source=ImageOps.exif_transpose(source).convert('RGBA').copy()
            pipe.source_init={'image':source,'sigma':sigma};schedule_kwargs={'sigmas':selected.tolist()}
            source_init_record={'requested_sigma':args.source_init_sigma,'effective_start_sigma':sigma,
                'original_schedule_steps':args.steps,'skipped_steps':index,'actual_steps':len(selected),
                'method':'experimental flow-matching source latent/noise initialization; not native API strength'}
        for seed in args.seeds:
            recordpath=out/'records'/f"{c['id']}_s{seed}_{args.precision}.json"
            if recordpath.exists():continue
            torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            trimmed=[]
            def trim_once(module,inputs):
                if not trimmed:
                    before=torch.cuda.memory_reserved();allocated=torch.cuda.memory_allocated()
                    torch.cuda.empty_cache()
                    trimmed.append({'reserved_before':before,'reserved_after':torch.cuda.memory_reserved(),
                        'allocated_before':allocated,'allocated_after':torch.cuda.memory_allocated()})
            hook=transformer.register_forward_pre_hook(trim_once) if args.trim_cache else None
            def progress(p,i,t,kw):
                if i%5==0:print(c['id'],args.precision,'step',i+1,'/',args.steps,flush=True)
                if args.save_final_latents and i==len(p.scheduler.timesteps)-1:
                    path=out/'latents'/f"{c['id']}_s{seed}_{args.precision}.pt";path.parent.mkdir(exist_ok=True)
                    torch.save({'latents':kw['latents'].detach().cpu(),'binding':cached['binding'],
                        'seed':seed,'worker_sha256':WORKER_SHA,'shape_before_unpack':list(kw['latents'].shape),
                        'width':c['width'],'height':c['height'],'online_inference':False},path)
                    print('saved exact final denoised latents',str(path),flush=True)
                return kw
            result=pipe(image=vae_images,prompt_embeds=cached['embedding'].to('cuda'),
                width=c['width'],height=c['height'],output_resolution=args.reference_resolution,
                num_inference_steps=args.steps,true_cfg_scale=1.0,use_kv_cache=args.kv_cache,
                generator=torch.Generator('cuda').manual_seed(seed),callback_on_step_end=progress,**schedule_kwargs).images[0]
            if hook:hook.remove()
            # Nested experiment folders can share a leaf name (e.g. "patch").
            # Bind the filename to the full run directory to avoid cross-run overwrites.
            run_digest=hashlib.sha256(str(out.resolve()).encode('utf8')).hexdigest()[:8]
            imagepath=ROOT/'output/result'/f"{c['role']}_result"/f"local1108_{out.name}_{run_digest}_{c['id']}_s{seed}_{args.precision}.png"
            imagepath.parent.mkdir(parents=True,exist_ok=True);result.save(imagepath)
            record={'id':c['id'],'role':c['role'],'target':c.get('target'),'source':c['source'],
                'references':c['references'],'instruction':c['instruction'],'binding':cached['binding'],
                'condition_order':c.get('condition_order','source_then_references'),
                'source_init_requested_sigma':args.source_init_sigma,'source_initialization':source_init_record,
                'vae_reference_mode':args.vae_reference_mode,'condition_routing':routing_record,
                'vae_condition_indices':args.vae_condition_indices,
                'output':str(imagepath),'sha256':sha(imagepath),'actual_size':list(result.size),
                'reference_sizes':cached['preprocessed_sizes'],'model':str(args.model),'precision':args.precision,
                'encoder_precision':args.encoder_precision,'steps':args.steps,'seed':seed,'true_cfg_scale':1.0,
                'encoder_vision_bf16':args.encoder_vision_bf16,
                'vae_tiling':False,'kv_cache':args.kv_cache,'seconds':time.monotonic()-start,
                'vae_decode_tiling':bool(args.vae_decode_tile_size),'vae_decode_tile_size':args.vae_decode_tile_size,
                'final_latents_saved':args.save_final_latents,
                'trim_cache':args.trim_cache,'allocator_trim':trimmed,
                'cpu_group_offload':args.cpu_offload or args.precision=='bf16',
                'kv_cache_cpu':args.kv_cache_cpu,
                'synchronous_offload':args.synchronous_offload,
                'offload_disk':str(args.offload_disk.resolve()) if args.offload_disk else None,
                'worker_sha256':WORKER_SHA,'online_inference':False,'training_used':False,**metrics(torch)}
            write(recordpath,record);print('rendered',c['id'],round(record['seconds'],1),str(imagepath),flush=True)
            del result;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['encode','render'],required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL);p.add_argument('--only',nargs='+')
    p.add_argument('--precision',choices=['nf4','bf16'],default='nf4')
    p.add_argument('--encoder-precision',choices=['nf4','bf16'],default='nf4')
    p.add_argument('--encoder-vision-bf16',action='store_true',help='Preserve the visual tower and LM head in BF16; language weights may remain NF4')
    p.add_argument('--encoder-disk-offload',type=Path,help='BF16 encoder: automatic 2GiB GPU / 2GiB CPU / disk placement; no language quantization')
    p.add_argument('--source-init-sigma',type=float,help='Experimental source-latent initialization at the nearest effective schedule sigma, 0<sigma<1')
    p.add_argument('--vae-reference-mode',choices=['all','source-only','selected'],default='all',help='Experimental: keep all contextual text but select reconstructive image blocks')
    p.add_argument('--vae-condition-indices',type=int,nargs='+',help='Zero-based condition slots for the selected mode, including the source slot')
    p.add_argument('--reference-resolution',type=int,default=512);p.add_argument('--steps',type=int,default=40)
    p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--kv-cache',action='store_true')
    p.add_argument('--trim-cache',action='store_true')
    p.add_argument('--vae-decode-tile-size',type=int,help='Experimental decode-only tiling; reference encoding remains untiled')
    p.add_argument('--save-final-latents',action='store_true',help='Save exact final denoised latents before decoding, including if decoding later fails')
    p.add_argument('--cpu-offload',action='store_true',help='Stream quantized generator blocks from CPU as well')
    p.add_argument('--synchronous-offload',action='store_true',help='Avoid asynchronous pinned-weight staging on RAM-limited hosts; preserves weight values')
    p.add_argument('--offload-disk',type=Path,help='Experimental BF16 group storage on disk instead of keeping full weights in RAM; use a fresh directory per model')
    p.add_argument('--kv-cache-cpu',action='store_true',help='Keep unquantized image-prefix KV on CPU; implies --kv-cache')
    args=p.parse_args();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.vae_decode_tile_size and (args.vae_decode_tile_size<64 or args.vae_decode_tile_size%32):p.error('VAE decode tile size must be a multiple of 32 and at least 64')
    if args.encoder_disk_offload and args.encoder_precision!='bf16':p.error('Encoder disk placement requires encoder-precision bf16')
    if args.offload_disk:
        if args.precision!='bf16':p.error('Disk group storage is validated only for BF16 tensors')
        args.cpu_offload=True;args.synchronous_offload=True
    if args.source_init_sigma is not None and not 0<args.source_init_sigma<1:p.error('source-init-sigma must be in (0,1)')
    if args.kv_cache_cpu:args.kv_cache=True
    cases=read(args.plan)['cases'];cases=[c for c in cases if not args.only or c['id'] in args.only]
    assert cases and (not args.only or {c['id'] for c in cases}==set(args.only))
    args.out.mkdir(parents=True,exist_ok=True)
    frozen=args.out/'workers'/(WORKER_SHA+'.py');frozen.parent.mkdir(exist_ok=True)
    if not frozen.exists():frozen.write_bytes(Path(__file__).read_bytes())
    write(args.out/f'request_{args.stage}_{args.precision}.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    from gpu_slot import gpu_slot
    print('Waiting for local study GPU slot',flush=True)
    with gpu_slot():
        import torch
        torch.set_num_threads(4)
        with torch.inference_mode():
            print('GPU slot acquired',flush=True)
            (encode if args.stage=='encode' else render)(cases,args,args.out,torch)
if __name__=='__main__':main()
