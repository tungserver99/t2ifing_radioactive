"""Sequential, resumable locked v3 campaign. No oracle is imported or executed."""
import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
import torch
from common import HERE, read_json, freeze, sha256, image_name, write_json
from semantic_native_rollout import OBJECTIVE
from semantic_sign_native_margin_refine import release


def call(run,file,*args):
    """Isolate GPU stages so the previous model/allocator is always released."""
    command=[sys.executable,str(HERE/file),'--run',str(run),*map(str,args)]
    folder=run/'logs';folder.mkdir(parents=True,exist_ok=True)
    label=file.removesuffix('.py')+'_'+('_'.join(map(str,args))).replace('/','_')
    with (folder/(label[:180]+'.log')).open('a',encoding='utf8') as log:
        log.write('\n'+json.dumps(command)+'\n');log.flush()
        try:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Stage failed: {file} {args}. See {log.name}') from exc


def margin_metrics(run,label,n):
    from semantic_sign_report import analytic_metrics
    from semantic_report import score_rows
    from semantic_carrier import SemanticCarrier,check_finite
    carrier=SemanticCarrier.load(run/'carrier')
    rows=score_rows(run,label,n,carrier)
    r=torch.tensor([row['coordinates'] for row in rows],dtype=torch.float32)
    signed=r*carrier.signs
    ordered=torch.sort(signed,descending=True,stable=True).values
    m26=ordered[:,25];check_finite(m26)
    k=torch.tensor([row['K'] for row in rows])
    return {**analytic_metrics(run,label,n,carrier),
            'D_26':(.02-m26).clamp_min(0).square().mean().item(),
            'margin_26':m26.tolist(),'margin_26_quantiles':dict(zip(['p10','p50','p90'],torch.quantile(m26,torch.tensor([.1,.5,.9])).tolist())),
            'fraction_margin26_ge_mu':(m26>=.02).float().mean().item(),
            'mean_per_image_bit_accuracy':k.float().mean().item()/32,
            'fraction_all32_correct':(k==32).float().mean().item(),
            'per_image_K':k.tolist()}


def choose_teacher(candidates):
    eligible=[]
    for r in candidates:
        m,q=r['metrics'],r['quality']
        if not all(math.isfinite(v) for v in (m['detection_rate'],m['D_26'],q['clip_drop'],q['paired_pixel_mse'])):
            raise ValueError('Nonfinite v3 teacher screening metric')
        if m['detection_rate']>=.9 and q['clip_drop']<=.01:eligible.append(r)
    return min(eligible,key=lambda r:(r['metrics']['D_26'],r['quality']['paired_pixel_mse'],r['step'])) if eligible else None


def screen(run):
    from semantic_checkpoint import verified_state
    from semantic_teacher import generate_set,load_native,score
    from semantic_report import teacher_quality
    from student import load_adapter
    a=read_json(run/'assets.json');folder=run/'teacher_fp/native_margin_refine'
    protocol=read_json(folder/'protocol.json');results=[]
    for step in (50,100,200,300):
        checkpoint=folder/f'checkpoint-{step}';verified_state(checkpoint,protocol)
        label=f'native_margin_checkpoint_{step}_validation';out=checkpoint/'screen.json'
        identity={'checkpoint_sha256':sha256(checkpoint/'checkpoint.json'),
                  'validation_images_sha256':sha256(run/'images/teacher_clean_validation/image_hashes.json'),
                  'prompts_sha256':sha256(run/'prompts.json'),'config_sha256':sha256(run/'config.json'),
                  'n':256,'subset':'first 256 frozen validation rows'}
        if out.exists():
            row=read_json(out)
            if row['identity']!=identity:raise RuntimeError('Teacher screening identity changed')
            for key,path in [('score_sha256',run/'scores'/f'{label}.json'),
                             ('image_manifest_sha256',run/'images'/label/'image_hashes.json')]:
                if row[key]!=sha256(path):raise RuntimeError('Teacher screening artifact changed')
            if margin_metrics(run,label,256)!=row['metrics']:raise RuntimeError('Teacher screening evidence changed')
        else:
            pipe=load_native(a['teacher_clean'],dtype=torch.float32)
            load_adapter(pipe.unet,checkpoint,pipe)
            pipe.fuse_lora(safe_fusing=True);pipe.unload_lora_weights();pipe.to(dtype=torch.float16)
            from peft.tuners.tuners_utils import BaseTunerLayer
            if any(isinstance(m,BaseTunerLayer) for m in pipe.unet.modules()):
                raise RuntimeError('Runtime adapter survived screening merge')
            model_identity={**a['teacher_clean_hashes'],'fused_adapter_sha256':sha256(checkpoint/'adapter_model.safetensors'),'fuse_precision':'float32'}
            generate_set(run,a['teacher_clean'],'validation',label,pipe,limit=256,model_identity=model_identity)
            del pipe;release()
            score(run,[label]);release()
            metrics=margin_metrics(run,label,256)
            quality=teacher_quality(run,label,limit=256);release()
            row={'step':step,'metrics':metrics,'quality':quality,'identity':identity,
                 'score_sha256':sha256(run/'scores'/f'{label}.json'),
                 'image_manifest_sha256':sha256(run/'images'/label/'image_hashes.json')}
            freeze(out,row)
        results.append(row)
    freeze(run/'result/teacher_screening.json',{'checkpoints':results,'selection':'eligible TPR>=.90/CLIP drop<=.01; min D26, pixel MSE, earlier step'})
    return results


def select_and_validate(run):
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import generate_set,score
    from semantic_assets import snapshot_identity
    from semantic_report import teacher_quality
    from semantic_sign_report import paired_pixel_visuals
    c=read_json(run/'config.json');candidates=read_json(run/'result/teacher_screening.json')['checkpoints']
    chosen=choose_teacher(candidates)
    if chosen is None:
        freeze(run/'result/teacher_status.json',{'status':'failed','reason':'No eligible checkpoint','downstream_ready':False})
        return False
    checkpoint=run/'teacher_fp/native_margin_refine'/f"checkpoint-{chosen['step']}"
    model=merge_teacher(run,checkpoint);release()
    selected={**chosen,'teacher_objective':OBJECTIVE,'model':model.relative_to(run).as_posix(),
              'model_hashes':snapshot_identity(model),'test_used_for_selection':False,
              'rule':'TPR>=.90 and CLIP drop<=.01; lowest D26, pixel MSE, earlier step'}
    freeze(run/'teacher_fp/selected.json',selected)
    generate_set(run,model,'validation','teacher_wm_validation');release()
    score(run,['teacher_wm_validation']);release()
    metrics=margin_metrics(run,'teacher_wm_validation',c['validation_n'])
    quality=teacher_quality(run,'teacher_wm_validation');release()
    visuals=paired_pixel_visuals(run,'teacher_clean_validation','teacher_wm_validation',c['validation_n'],max_visuals=16)
    full={'metrics':metrics,'quality':quality,'visuals':visuals,
          'selected_sha256':sha256(run/'teacher_fp/selected.json'),
          'scores_sha256':sha256(run/'scores/teacher_wm_validation.json')}
    freeze(run/'result/teacher_full_validation.json',full)
    passed=metrics['detection_rate']>=.9 and quality['clip_drop']<=.01
    freeze(run/'result/teacher_status.json',{'status':'passed' if passed else 'failed',
            'reason':'full source validation TPR/CLIP gate','downstream_ready':passed,'lpips_is_gate':False})
    return passed


def downstream_data(run):
    from semantic_teacher import generate_set
    from semantic_assets import snapshot_identity
    from semantic_native_lineage import verify_images
    a=read_json(run/'assets.json');c=read_json(run/'config.json');selection=read_json(run/'teacher_fp/selected.json')
    if not read_json(run/'result/teacher_status.json')['downstream_ready']:
        raise RuntimeError('Teacher gate failed; downstream is prohibited')
    model=run/selection['model']
    if snapshot_identity(model)!=selection['model_hashes']:raise RuntimeError('Merged teacher changed')
    generate_set(run,model,'train','teacher_wm_train');release()
    prompts=read_json(run/'prompts.json')
    for condition in ('clean','wm'):
        for split in ('train','validation'):
            verify_images(run/'images'/f'teacher_{condition}_{split}',prompts[split],c)
    freeze(run/'student_data.json',{'format':'ordinary RGB-caption pairs only',
            'prompts_sha256':sha256(run/'prompts.json'),
            'images':{f'{condition}_{split}':sha256(run/'images'/f'teacher_{condition}_{split}'/'image_hashes.json')
                      for condition in ('clean','wm') for split in ('train','validation')},
            'metadata':{f'{condition}_{split}':sha256(run/'images'/f'teacher_{condition}_{split}'/'metadata.jsonl')
                        for condition in ('clean','wm') for split in ('train','validation')},
            'student_inputs_contain_key':False,'teacher_sha256':sha256(run/'teacher_fp/selected.json')})


def final_generation(run):
    from semantic_teacher import generate_set
    from semantic_assets import snapshot_identity
    from student import branches
    c=read_json(run/'config.json');a=read_json(run/'assets.json');selected=read_json(run/'teacher_fp/selected.json')
    for b in branches(c):
        selection=read_json(run/'result/selection'/f"{b['name']}.json")
        if selection['test_used_for_selection'] or selection['detector_used_for_selection'] or selection['utility_used_for_selection']:
            raise RuntimeError('Student selection leaked non-MSE metrics')
    if snapshot_identity(run/selected['model'])!=selected['model_hashes']:raise RuntimeError('Selected merged model changed')
    generate_set(run,a['teacher_clean'],'test','teacher_clean_test');release()
    generate_set(run,run/selected['model'],'test','teacher_wm_test');release()
    # Isolated calls release every generation pipeline.
    for name in ['unadapted']+[b['name'] for b in branches(c)]:
        call(run,'student.py','--stage','generate','--branch',name)


def final_metrics(run):
    from student import branches
    from semantic_sign_report import random_key_validation,paired_pixel_visuals
    from semantic_report import score_rows
    c=read_json(run/'config.json')
    labels=['teacher_clean_test','teacher_wm_test','unadapted']+[b['name'] for b in branches(c)]
    call(run,'semantic_teacher.py','--stage','score','--directories',*labels)
    metrics={label:margin_metrics(run,label,c['test_n']) for label in labels}
    random_keys={label:random_key_validation(torch.tensor([r['coordinates'] for r in score_rows(run,label,c['test_n'])]),32,.01) for label in labels}
    panels={'teacher':paired_pixel_visuals(run,'teacher_clean_test','teacher_wm_test',c['test_n'],16)}
    for b in branches(c):panels[b['name']]=paired_pixel_visuals(run,'unadapted',b['name'],c['test_n'],16)
    # A fixed three-column student comparison uses the same first 16 test rows.
    from PIL import Image,ImageDraw
    target=run/'result/final_visuals';target.mkdir(parents=True,exist_ok=True)
    for i in range(min(16,c['test_n'])):
        canvas=Image.new('RGB',(1536,536),'white');draw=ImageDraw.Draw(canvas)
        for j,label in enumerate(('unadapted','lora_r32_conv_wm','lora_r32_conv_clean')):
            with Image.open(run/'images'/label/image_name(i)) as im:canvas.paste(im.convert('RGB'),(512*j,24))
            draw.text((512*j,4),label,fill='black')
        canvas.save(target/image_name(i))
    freeze(run/'result/final_rssf.json',{'models':metrics,'random_keys':random_keys,'paired_diagnostics':panels,
         'primary':'per-image K>=24, alpha=.01','owner_key_unchanged':True,'test_used_for_selection':False})
    # LPIPS is a paired realization diagnostic, never optimized or gated.
    import lpips
    from semantic_teacher import pixels
    perceptual=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    distances={}
    for clean,marked in [('teacher_clean_test','teacher_wm_test'),('unadapted','lora_r32_conv_wm'),('unadapted','lora_r32_conv_clean')]:
        values=[]
        for i in range(c['test_n']):
            with torch.no_grad():value=perceptual(pixels([run/'images'/clean/image_name(i)]),pixels([run/'images'/marked/image_name(i)])).item()
            if not math.isfinite(value):raise RuntimeError('Nonfinite paired LPIPS')
            values.append(value)
        distances[marked]={'mean':statistics.mean(values),'per_image':values,'diagnostic_only':True}
    freeze(run/'result/final_lpips.json',distances)


def report(run):
    from student import branches
    c=read_json(run/'config.json');status=read_json(run/'result/teacher_status.json')
    screen=read_json(run/'result/teacher_screening.json')
    payload={'teacher_status':status,'source':read_json(run/'source_lineage.json'),'teacher_screening':screen,
             'code_manifest_sha256':sha256(run/'code_manifest.json')}
    payload['teacher_training']=read_json(run/'teacher_fp/native_margin_refine/trained.json')
    lines=['# RSSF v3 native margin refinement','',f"Teacher: {status['status']}",
           f"Source: {c['source_run']}, trajectory checkpoint-100. Current-weight PNDM50; final ten calls differentiable; dynamic Top-26, mu=.02; fidelity cap=.5.",
           '', '| Step | TPR | D26 | CLIP drop | Pixel MSE | LPIPS (diagnostic) |',
           '|---:|---:|---:|---:|---:|---:|']
    for row in screen['checkpoints']:
        m,q=row['metrics'],row['quality']
        lines.append(f"| {row['step']} | {m['detection_rate']:.5f} | {m['D_26']:.7g} | {q['clip_drop']:.7g} | {q['paired_pixel_mse']:.7g} | {q['paired_lpips']:.7g} |")
    if (run/'result/teacher_full_validation.json').exists():
        payload['teacher_full_validation']=read_json(run/'result/teacher_full_validation.json')
        chosen=read_json(run/'teacher_fp/selected.json')
        lines+=['',f"Selected teacher: step {chosen['step']}; lowest D26 among TPR/CLIP-eligible checkpoints (pixel MSE, earlier step break ties). Full metrics and margin distribution: teacher_full_validation.json; deterministic pairs listed there."]
    if (run/'result/final_rssf.json').exists():
        payload['final_rssf']=read_json(run/'result/final_rssf.json');payload['utility']=read_json(run/'result/utility.json')
        lines+=['','Student: ordinary RGB-caption SD1.4 rank-32 attention+Conv LoRA, 7500 applied denoising-MSE updates.']
        for b in branches(c):
            name=b['name'];selection=read_json(run/'result/student_selection'/f'{name}.json')
            payload[name]={'selection':selection,'training':read_json(run/'students'/name/'protocol.json')}
            lines+=['',f"{name}: selected {selection['selection']['selected_step']} by validation denoising MSE only.",
                    'Validation losses: '+json.dumps(selection['milestones']),
                    'Exact training command: `'+__import__('shlex').join(payload[name]['training']['command'])+'`']
        lines+=['','| Model | TPR/FPR | Wilson 95% CI | Mean K | K_M/32 | p_M |', '|---|---:|---|---:|---:|---:|']
        for name,m in payload['final_rssf']['models'].items():
            lines.append(f"| {name} | {m['detection_rate']:.5f} | [{m['ci95_low']:.5f}, {m['ci95_high']:.5f}] | {m['mean_K']:.3f} | {m['K_M']}/32 | {m['p_M']:.6g} |")
        lines+=['','Full per-image K, bit accuracy, model signs, wrong-key evidence: final_rssf.json. FID/KID/CLIP: utility.json. Paired LPIPS: final_lpips.json. Fixed student panels: final_visuals/.']
    elif not status['downstream_ready']:
        lines+=['','Teacher failed the frozen validation gate. Student training and test generation stopped; no protocol or key was changed.']
    lines+=['',f"Teacher training: {payload['teacher_training']['events']} fixed batch events; {payload['teacher_training']['applied_updates']} applied updates; {payload['teacher_training']['skipped_zero_margin_gradient']} zero-margin-gradient skips."]
    caveats=['One training seed does not measure between-run uncertainty.','Wilson intervals describe images, not training seeds.',
             'Finite-sample FID is biased.','KID subset standard deviation is not a training-seed confidence interval.',
             'LPIPS measures paired realization distance, not absolute aesthetic quality.',
             'Model-level K_M is not per-image detection.','Test data were not used for teacher or student checkpoint selection.']
    lines+=['']+caveats
    freeze(run/'result/report.json',payload)
    (run/'result/README.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
    freeze(run/'complete.json',{'status':'completed' if status['downstream_ready'] else 'teacher_failed',
           'report_sha256':sha256(run/'result/report.json'),'student_target_is_research_target':True})


def execute(run,stage):
    from semantic_native_lineage import LOCKED,code_identity
    from semantic_sign_native_margin_refine import training_protocol
    c=read_json(run/'config.json')
    if any(c.get(k)!=v for k,v in LOCKED.items()) or c.get('native_micro_batch') not in (1,2):
        raise RuntimeError('Staged run configuration violates locked v3 protocol')
    if read_json(run/'code_manifest.json')!=code_identity():
        raise RuntimeError('Staged run code identity changed')
    protocol_path=run/'teacher_fp/native_margin_refine/protocol.json'
    if protocol_path.exists() and read_json(protocol_path)!=training_protocol(run):
        raise RuntimeError('Staged run inputs differ from committed training protocol')
    if stage=='refine':
        from semantic_sign_native_margin_refine import train
        return train(run)
    if stage=='screen':return screen(run)
    if stage=='teacher-validation':return select_and_validate(run)
    if stage=='data':return downstream_data(run)
    if stage=='test-generation':return final_generation(run)
    if stage=='test-metrics':return final_metrics(run)
    if stage=='report':return report(run)
    raise ValueError(stage)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,help='Explicit completed v2 run; never guessed by modification time')
    p.add_argument('--output',type=Path,help='Separate v3 run; rerun identical command to resume')
    p.add_argument('--run',type=Path,help='Internal stage / explicit resume run')
    p.add_argument('--warm-start-step',type=int,choices=[100],default=100)
    p.add_argument('--student-snapshot',type=Path)
    p.add_argument('--native-micro-batch',type=int,choices=[1,2],default=None)
    p.add_argument('--stage',choices=['all','refine','screen','teacher-validation','data','test-generation','test-metrics','report'],default='all')
    a=p.parse_args(argv)
    if a.run:
        run=a.run.resolve();existing=read_json(run/'config.json')
        source=Path(existing['source_run']);micro=a.native_micro_batch or existing['native_micro_batch']
    else:
        if not a.source_run:p.error('--source-run is required for a new campaign')
        source=a.source_run.resolve();run=(a.output or source.with_name(source.name+'_native_margin_v3')).resolve()
        micro=a.native_micro_batch or 2
    if run==source or source in run.parents or run in source.parents:
        p.error('New run and source must be separate, non-nested directories')
    if a.stage!='all':return execute(run,a.stage)
    from semantic_native_lineage import prepare
    from student import branches
    run.mkdir(parents=True,exist_ok=True)
    # Match the repository lock on both Linux server and local Windows tooling.
    import run as runner
    with (run/'.campaign.lock').open('a+') as lock:
        runner.fcntl.flock(lock,runner.fcntl.LOCK_EX|runner.fcntl.LOCK_NB)
        c=prepare(run,source,micro,a.student_snapshot)
        print(f'V3 run: {run}. Stage logs: {run/"logs"}',flush=True)
        call(run,'semantic_teacher.py','--stage','score','--directories','teacher_clean_calibration','teacher_clean_validation')
        from semantic_sign_report import freeze_analytic_verifier,validate_clean_null
        freeze_analytic_verifier(run);validate_clean_null(run,('teacher_clean_calibration',))
        call(run,'semantic_native_campaign.py','--stage','refine')
        call(run,'semantic_native_campaign.py','--stage','screen')
        call(run,'semantic_native_campaign.py','--stage','teacher-validation')
        if not read_json(run/'result/teacher_status.json')['downstream_ready']:
            report(run);print(f'Teacher gate failed; stopped. {run/"result/README.md"}',flush=True);return
        call(run,'semantic_native_campaign.py','--stage','data')
        call(run,'validation.py','--stage','cache')
        for b in branches(c):
            call(run,'student.py','--stage','train','--branch',b['name'])
            call(run,'validation.py','--stage','evaluate','--branch',b['name'])
            # No detector or utility scoring precedes the frozen MSE decision.
            call(run,'validation.py','--stage','select','--branch',b['name'])
        call(run,'semantic_native_campaign.py','--stage','test-generation')
        call(run,'semantic_native_campaign.py','--stage','test-metrics')
        labels=['teacher_clean_test','teacher_wm_test','unadapted']+[b['name'] for b in branches(c)]
        call(run,'utility.py','--labels',*labels)
        report(run)
        print(f'Finished: {run/"result/README.md"}',flush=True)

if __name__=='__main__':main()
