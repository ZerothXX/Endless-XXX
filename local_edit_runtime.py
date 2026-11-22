"""Standalone local Qwen edit + local visual feedback runner, without a Codex dependency."""
import argparse,os,subprocess,sys,time,uuid,hashlib
from pathlib import Path
from local_multiref import ROOT,read,write,sha

def load_role_card(folder,explicit=None):
    path=explicit or folder/'local_design.json'
    pointer=folder/'card/current.json'
    if explicit is None and pointer.exists():path=Path(read(pointer)['card'])
    if explicit and not path.exists():raise FileNotFoundError(path)
    card=read(path) if path.exists() else {}
    for asset in [*card.get('source_assets',[]),*card.get('derived_assets',[])]:
        if not Path(asset['path']).is_file() or sha(asset['path'])!=asset['sha256']:
            raise ValueError('Character references changed; rebuild the versioned role card before generation')
    return path,card

def run_stage(script,args,log):
    python=Path(os.environ.get('ENDLESS_PYTHON',sys.executable))
    with Path(log).open('w',encoding='utf8') as stream:
        result=subprocess.run([str(python),'-u',str(ROOT/script),*map(str,args)],cwd=ROOT,
            env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),
            stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'本地阶段失败，见 {log}')

def build_plan(character,sources,kind,out,edge,instruction=None,detail_references=None,design_constraints=None,references=None,design_card=None,reference_profile=None):
    from PIL import Image,ImageOps
    folder=ROOT/'dataset'/character
    card_path,card=load_role_card(folder,design_card)
    refs=[]
    if references:
        refs=[str(p.resolve()) for p in references]
    elif reference_profile:
        if reference_profile not in card.get('profiles',{}):raise ValueError('Unknown character card profile: '+reference_profile)
        refs=[str((card_path.parent/Path(x)).resolve()) for x in card['profiles'][reference_profile]]
    elif card.get('material_references') and kind!='human':
        refs=[str((card_path.parent/Path(x)).resolve()) for x in card['material_references']]
    elif card.get('identity_references'):
        refs=[str((card_path.parent/Path(x)).resolve()) for x in card['identity_references']]
    else:
        pack=read(folder/'reference_pack.json')
        for row in pack['fixed']:
            p=Path(row['path']);refs.append(str((p if p.is_absolute() else folder/p).resolve()))
    assert refs,'At least one fixed character/design reference is required'
    base_count=len(refs)
    refs += [str(p.resolve()) for p in (detail_references or [])]
    for path in refs:
        if not Path(path).is_file():raise FileNotFoundError(path)
    evidence=('show ONE target character and provide only identity and design' if kind=='human' else
        'supply only the fixed material, palette and motif design vocabulary; never copy their layout')
    common=(f'Image 1 is the sole edit target. Images 2 through {base_count+1} {evidence}. '
        'Preserve the source framing, subject location, proportions, pose, expression, eye state, limb positions, '
        'background, viewpoint and illumination. Never import reference poses, cameras, weapons or extra people. ')
    if detail_references:
        common += f'Images {base_count+2} through {len(refs)+1} provide additional fixed motif/detail evidence only; never copy their layout. '
    constraints=design_constraints if design_constraints is not None else card.get('design_constraints','')
    if constraints:common += constraints+' '
    if reference_profile=='material_isolated' and kind!='human':
        from local_palette_prompt import palette_instruction
        common += palette_instruction(card['palette'])
    if kind=='human' and card.get('identity_constraints'):common += card['identity_constraints']+' '
    actions={
        'human':('Replace the person in place with the exact reference character: actual hairstyle, face design, eyes, '
            'costume construction, fasteners, bows, precise flower/crest shapes and color relationships. '
            'Keep the body pose underneath, while fully replacing the original garment design, collar, sleeve construction '
            'and closures. Do not retain the source person\'s clothing, hairstyle or facial identity. '
            'Use a refined dimensional anime interpretation that matches the source rendering medium and lighting. '
            'Preserve source torso/head angles and expression before displaying costume details.'),
        'object':('Redesign the existing object using the character design language, with a visibly complete tasteful color-panel '
            'design, faithful small motifs and restrained physical accessories. Keep its original material and function, '
            'silhouette, openings, handle, contents and scene. Translate textile colors into surface finishes naturally. '
            'Keep flat patterns flat and physical ornaments dimensional. No character print, face or human clothing pasted on it.'),
        'animal':('Keep the exact original natural animal, species, face, fur, eye color, anatomy and posture. '
            'Add a modest well-fitted textile collar or small accessory based on exact character colors, '
            'motif geometry, bows and materials. Preserve fur outside the accessory. No human face, wig, '
            'makeup, humanoid body or full-body costume. Match real light, folds and contact shadows.')}
    cases=[]
    for i,source in enumerate(sources):
        with Image.open(source) as im:
            im=ImageOps.exif_transpose(im)
            scale=edge/max(im.size);w=max(32,round(im.width*scale/32)*32);h=max(32,round(im.height*scale/32)*32)
        cases.append({'id':f'input_{i+1}','role':character,'target':Path(source).stem,'kind':kind,
            'source':str(Path(source).resolve()),'references':refs,'instruction':instruction or common+actions[kind],
            'width':w,'height':h})
    path=out/'plan.json';write(path,{'cases':cases,'online_inference':False,'prompt_source':'generic local runner or explicit user instruction'})
    return path

def main(argv=None):
    p=argparse.ArgumentParser(description='本地多参考编辑与本地视觉反馈；不调用线上成图服务')
    p.add_argument('--character');p.add_argument('--input',type=Path,nargs='+');p.add_argument('--kind',choices=['human','object','animal'],default='human')
    p.add_argument('--plan',type=Path,help='使用已审核的完整实验清单，优先于通用请求构建')
    p.add_argument('--out',type=Path);p.add_argument('--edge',type=int,default=768)
    p.add_argument('--precision',choices=['nf4','bf16'],default='nf4');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--encoder-vision-bf16',action='store_true',help='保留图文编码器视觉层BF16；其余语言层仍NF4')
    p.add_argument('--kv-cache',action='store_true',help='缓存图像条件，可能增加共享内存；不表示质量提高')
    p.add_argument('--cpu-offload',action='store_true',help='量化生成器也逐块卸载到CPU')
    p.add_argument('--synchronous-offload',action='store_true',help='使用同步逐块卸载，避免内存不足时的异步固定内存换页')
    p.add_argument('--trim-cache',action='store_true',help='去噪前释放未使用的CUDA内存池')
    p.add_argument('--kv-cache-cpu',action='store_true',help='图像前缀缓存放在CPU，逐层搬运')
    p.add_argument('--reference-resolution',type=int,default=512,help='Qwen条件图目标面积边长')
    p.add_argument('--feedback-rounds',type=int,default=0);p.add_argument('--instruction')
    p.add_argument('--detail-reference',type=Path,nargs='+',help='在当前参考组织后追加固定设计细节图')
    p.add_argument('--reference',type=Path,nargs='+',help='显式指定固定外观/材质参考，替代默认参考组织')
    p.add_argument('--design-card',type=Path,help='固定角色设计数据；默认优先读取 dataset/<角色>/card/current.json 版本指针')
    p.add_argument('--reference-profile',help='显式选择版本角色卡中的参考组织，如 human_sheet / material_palette_dominant')
    p.add_argument('--auto-compose',action='store_true',help='实验：animal任务自动定位项圈，仅将预测项圈合成回原图；未检出时拒绝')
    p.add_argument('--design-constraints',help='与固定角色素材对应的设计说明，不按输入图片选择代码分支')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if a.feedback_rounds not in range(4):p.error('feedback-rounds must be 0..3')
    if a.edge<256 or a.edge%32:p.error('edge must be a multiple of 32 and at least 256')
    if a.reference_resolution<256 or a.reference_resolution%32:p.error('reference-resolution must be a multiple of 32 and at least 256')
    out=a.out or ROOT/'output/local_edit_runs'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=True)
    if a.plan:plan=a.plan.resolve()
    else:
        if not a.character or not a.input:p.error('Provide --plan or --character and --input')
        if Path(a.character).name!=a.character or a.character in ('.','..'):p.error('Invalid character name')
        plan=build_plan(a.character,a.input,a.kind,out,a.edge,a.instruction,a.detail_reference,a.design_constraints,a.reference,a.design_card,a.reference_profile)
    if a.auto_compose and any(c.get('kind')!='animal' for c in read(plan)['cases']):
        p.error('auto-compose is currently validated only for animal collar support')
    if a.prepare_only:print(plan);return
    current_plan=plan
    run_id=hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:10]
    latest_candidates={}
    for round_index in range(a.feedback_rounds+1):
        work=out/f'{run_id}_round{round_index}'
        for stage in ['encode','render']:
            args=['--stage',stage,'--plan',current_plan,'--out',work,'--precision',a.precision,'--seeds',a.seed]
            args+=['--reference-resolution',a.reference_resolution]
            if a.kv_cache:args.append('--kv-cache')
            if a.cpu_offload:args.append('--cpu-offload')
            if a.synchronous_offload:args.append('--synchronous-offload')
            if a.trim_cache:args.append('--trim-cache')
            if a.kv_cache_cpu:args.append('--kv-cache-cpu')
            if a.encoder_vision_bf16:args.append('--encoder-vision-bf16')
            print('本地阶段',round_index,stage,flush=True)
            run_stage('local_multiref.py',args,out/f'round{round_index}_{stage}.log')
        for record_path in (work/'records').glob('*.json'):
            candidate=read(record_path);case=candidate.get('binding',{}).get('case',{})
            latest_candidates[case.get('root_case_id',candidate['id'])]=candidate
        if round_index<a.feedback_rounds:
            feedback=out/f'feedback{round_index}'
            run_stage('local_edit_critic.py',['--plan',current_plan,'--records',work/'records','--out',feedback,
                '--precision',a.precision,'--seed',a.seed],out/f'feedback{round_index}.log')
            next_plan=feedback/'plan_critic1.json'
            if not read(next_plan)['cases']:
                print('Local critic abstained; retaining the previous candidates for review.',flush=True)
                break
            current_plan=next_plan
    result=list(latest_candidates.values())
    if a.auto_compose:
        from local_mask_compose import compose
        regions=out/'grounded_regions';jobs=[]
        for i,r in enumerate(result):jobs.append({'id':f'candidate_{i}','image':r['output'],'query':'collar.','threshold':.25})
        jobs_path=out/'grounding_jobs.json';write(jobs_path,{'jobs':jobs})
        run_stage('local_grounded_regions.py',['--jobs',jobs_path,'--out',regions],out/'grounding.log')
        composites=[]
        for i,r in enumerate(result):
            dest=ROOT/'output/result'/f"{r['role']}_result"/f'local_{run_id}_{i}_auto_collar.png'
            original_source=r.get('binding',{}).get('case',{}).get('original_source',r['source'])
            record=compose(original_source,r['output'],regions/f'candidate_{i}.json','collar',dest)
            record.update(id=r['id'],role=r['role'],parent_record=r)
            write(out/'composite_records'/f'candidate_{i}.json',record);composites.append(record)
        result=composites
    write(out/'results.json',{'records':result,'offline_inference':True,'quality_status':'requires visual review; local VLM is not automatic acceptance'})
    for r in result:print(r['output'])
if __name__=='__main__':main()
