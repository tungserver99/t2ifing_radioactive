"""Isolated GPU stages. A fresh process releases memory between training/evaluation."""
import argparse
import csv
import shutil
from pathlib import Path
import paths
from paths import freeze, read_json, write_json, sha256, SHARED
from new_watermark.data import prepare, load_base, generate, FeatureReader, paired_psnr, verify_images
from new_watermark.method import make_detector, score_features, choose_teacher
from new_watermark.teacher import encode_training, train_teacher, export_teacher, verify_export
from student import branches, checkpoint_hashes, load_pipeline
from validation import choose_checkpoint, validation_label, SELECTION_RULE
from report import calibrate, wilson, paired_ci


def clean_design(run):
    pipe=load_base(run)
    for split in ('design_fit','teacher_train','meta_probe','calibration','pilot_train','pilot_eval'):
        generate(run,pipe,'clean_'+split,split,read_json(run/'assets.json')['base_revision'])


def detector_stage(run):
    c=read_json(run/'config.json'); prompts=read_json(run/'prompts.json'); reader=FeatureReader(run)
    fit=reader.read('clean_design_fit',prompts['design_fit'])
    cal=reader.read('clean_calibration',prompts['calibration'])
    detector=make_detector(fit['features'],c['key_seed'])
    values=score_features(cal['features'],detector)
    detector.update(thresholds=[calibrate(values[:,i].tolist(),c['fpr']) for i in range(values.shape[1])],
        target_fpr=c['fpr'],calibration_n=len(values),
        calibration_features_sha256=sha256(run/'features/clean_calibration.json'),
        fit_features_sha256=sha256(run/'features/clean_design_fit.json'),
        decision='score >= frozen threshold',test_used_for_calibration=False)
    freeze(run/'result/detector.json',detector)


def teacher_design(run,step):
    model=run/'models'/f'teacher_{step}'; identity=verify_export(model)
    pipe=load_base(run,model)
    for split in ('pilot_train','pilot_eval'):
        generate(run,pipe,f'teacher_{step}_{split}',split,identity)


def copy_dataset(source,destination,n):
    verify_images(source,n)
    if (destination/'image_hashes.json').exists():
        if verify_images(destination,n)!=read_json(source/'image_hashes.json'): raise RuntimeError('Changed copied dataset')
        return
    destination.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file(): shutil.copy2(p,destination/p.name)
    verify_images(destination,n)


def setup_pilots(run):
    c=read_json(run/'config.json'); prompts=read_json(run/'prompts.json')
    for ident in ['clean']+[str(s) for s in c['teacher_checkpoints']]:
        folder=run/'pilots'/ident; folder.mkdir(parents=True,exist_ok=True)
        condition='clean' if ident=='clean' else 'wm'
        branch={'name':'pilot_'+condition,'kind':'lora','rank':c['pilot_rank'],'scope':c['student_scope'],'condition':condition}
        pc={**c,'student_branches':[branch],'train_n':c['pilot_train_n'],'test_n':c['pilot_eval_n'],
            'max_train_steps':c['pilot_steps'],'checkpoint_steps':[c['pilot_steps']],
            'seed':c['seed']+10000,'role':'design-only pilot; never the final student'}
        freeze(folder/'config.json',pc); freeze(folder/'assets.json',read_json(run/'assets.json'))
        freeze(folder/'prompts.json',{'train':prompts['pilot_train'],'test':prompts['pilot_eval']})
        label='clean_pilot_train' if ident=='clean' else f'teacher_{ident}_pilot_train'
        copy_dataset(run/'images'/label,folder/'images'/f'teacher_{condition}_train',c['pilot_train_n'])


def pilot_generate(run,ident):
    c=read_json(run/'config.json'); folder=run/'pilots'/ident
    name='pilot_clean' if ident=='clean' else 'pilot_wm'
    snap=folder/'students'/name/f"snapshot-{c['pilot_steps']}"
    pipe,weights=load_pipeline(folder,name,checkpoint_dir=snap)
    # Student only receives its base + freshly trained adapter. Root's common eval seeds.
    generate(run,pipe,'pilot_'+ident,'pilot_eval',{'pilot':ident,'checkpoint':weights})


def select_teacher(run):
    c=read_json(run/'config.json'); prompts=read_json(run/'prompts.json')['pilot_eval']
    detector=read_json(run/'result/detector.json'); threshold=detector['thresholds'][0]
    reader=FeatureReader(run)
    clean=reader.read('clean_pilot_eval',prompts); control=reader.read('pilot_clean',prompts)
    clean_rate=float((score_features(control['features'],detector)[:,0]>=threshold).mean())
    rows=[]
    for step in c['teacher_checkpoints']:
        label=f'teacher_{step}_pilot_eval'
        teacher=reader.read(label,prompts); student=reader.read('pilot_'+str(step),prompts)
        row={'step':step,'teacher_tpr':float((score_features(teacher['features'],detector)[:,0]>=threshold).mean()),
             'teacher_ci_low':wilson(int((score_features(teacher['features'],detector)[:,0]>=threshold).sum()),len(prompts))[0],
             'pilot_tpr':float((score_features(student['features'],detector)[:,0]>=threshold).mean()),
             'clean_pilot_fpr':clean_rate,'clip_drop':clean['clip_mean']-teacher['clip_mean'],
             'pilot_clip_drop':control['clip_mean']-student['clip_mean'],
             'psnr':paired_psnr(run/'images/clean_pilot_eval',run/'images'/label,len(prompts))}
        rows.append(row)
    selection=choose_teacher(rows,c['gates'])
    selection.update(measurements=rows,detector_sha256=sha256(run/'result/detector.json'),
        pilot_budget={'per_student_steps':c['pilot_steps'],'students':len(rows)+1,'train_n':c['pilot_train_n'],'eval_n':len(prompts)},
        selected_model_export_sha256=verify_export(run/'models'/f"teacher_{selection['selected_step']}"))
    freeze(run/'result/teacher_selection.json',selection)
    print('Teacher selection:',selection,flush=True)


def final_teacher_images(run,condition):
    c=read_json(run/'config.json'); selection=read_json(run/'result/teacher_selection.json')
    if selection['detector_sha256']!=sha256(run/'result/detector.json'): raise RuntimeError('Frozen detector changed')
    step=selection['selected_step'] if condition=='wm' else c['teacher_checkpoints'][0]
    model=None if condition=='clean' else run/'models'/f'teacher_{step}'
    identity=read_json(run/'assets.json')['base_revision'] if model is None else verify_export(model)
    if condition=='wm' and identity!=selection['selected_model_export_sha256']: raise RuntimeError('Selected teacher changed')
    pipe=load_base(run,model)
    for split in (('test',) if condition=='clean' else ('train','validation','test')):
        generate(run,pipe,f'teacher_{condition}_{split}',split,identity)


def select_student(run,name):
    c=read_json(run/'config.json')
    rows=[read_json(run/'validation'/f'{validation_label(name,s)}.json') for s in c['checkpoint_steps']]
    winner=choose_checkpoint(rows)
    freeze(run/'result/selection'/f'{name}.json',{'rule':SELECTION_RULE,'selected_step':winner['step'],
        'denoising_mse':winner['denoising_mse'],'checkpoint_hashes':winner['checkpoint_hashes'],
        'validation_cache_sha256':winner['validation_cache_sha256'],
        'candidate_loss_sha256':{str(r['step']):sha256(run/'validation'/f"{validation_label(name,r['step'])}.json") for r in rows},
        'test_used_for_selection':False,'detector_used_for_selection':False,'utility_used_for_selection':False})
    print(f"Selected {name}: {winner['step']} by held-out MSE only",flush=True)


def final_student_images(run,name):
    pipe,weights=load_pipeline(run,name)
    generate(run,pipe,name,'test',{'student':name,'checkpoint':weights})


def csv_file(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize(run):
    import numpy as np
    c=read_json(run/'config.json'); prompts=read_json(run/'prompts.json')
    detector=read_json(run/'result/detector.json'); selection=read_json(run/'result/teacher_selection.json')
    if sha256(run/'result/detector.json')!=selection['detector_sha256']: raise RuntimeError('Changed detector after selection')
    reader=FeatureReader(run)
    labels=['teacher_clean_test','teacher_wm_test','unadapted']+[b['name'] for b in branches(c)]
    utility_path=run/'result/utility.json'
    utility=read_json(utility_path)['branches'] if utility_path.exists() else {}
    table=[]; indicators={}; wrong=[]
    for label in labels:
        data=reader.read(label,prompts['test']); scores=score_features(data['features'],detector)
        detected=(scores[:,0]>=detector['thresholds'][0]).astype(int).tolist(); indicators[label]=detected
        n=len(detected); k=sum(detected); ci=wilson(k,n); quality=utility.get(label,{})
        table.append({'model':label,'detected':k,'n':n,'detection_rate':k/n,'ci95_low':ci[0],'ci95_high':ci[1],
            'semantic_clip':data['clip_mean'],'fid':quality.get('frechet_inception_distance'),
            'kid':quality.get('kernel_inception_distance_mean'),
            'meaning':'false-positive rate' if label in ['teacher_clean_test','unadapted'] or label.endswith('_clean') else 'watermark detection rate'})
        for j in range(1,scores.shape[1]):
            wrong.append({'model':label,'wrong_key_index':j,'detected':int((scores[:,j]>=detector['thresholds'][j]).sum()),'n':n})
        write_json(run/'scores'/f'{label}.json',{'scores':scores.tolist(),'image_hashes':data['identity']['images']})
    effects={}
    for rank in c['ranks']:
        diff=[x-y for x,y in zip(indicators[f'lora_r{rank}_wm'],indicators['unadapted'])]
        effects[f'r{rank}_wm_minus_unadapted']={'estimate_pp':100*float(np.mean(diff)),
            'paired_ci95_pp':[100*x for x in paired_ci(diff,c['seed'])],
            'scope':'image bootstrap conditional on one training seed; no training-seed uncertainty',
            'clean_trained_adjustment':False}
    milestones=[]
    for b in branches(c):
        chosen=read_json(run/'result/selection'/f"{b['name']}.json")
        for step in c['checkpoint_steps']:
            label=validation_label(b['name'],step)
            data=reader.read(label,prompts['validation'][:c['validation_generation_n']])
            values=score_features(data['features'],detector)[:,0]
            loss=read_json(run/'validation'/f'{label}.json')['denoising_mse']
            milestones.append({'branch':b['name'],'step':step,'selected':chosen['selected_step']==step,'denoising_mse':loss,
                'detection_rate':float((values>=detector['thresholds'][0]).mean()),'semantic_clip':data['clip_mean'],
                'detector_used_for_selection':False})
    warnings=['Experimental zero-bit detector, not multi-bit decoding or cryptographic ownership proof.',
              'One seed; pilot search cost is additional compute. Repeat seeds and independent keys before paper claims.',
              'Teacher is trained with a one-step denoising proxy; actual generation and transfer can fail.',
              'Semantic shifts can alter content; inspect images and quality metrics, not just TPR.',
              'Wrong keys are a finite negative-control audit, not a security guarantee.',
              'All final students start from the CLEAN base, not the marked teacher; no key or detector loss in student training.']
    if selection['warning']: warnings.append(selection['warning'])
    warnings.append('No final clean-trained or fixed-teacher students: differences versus unadapted cannot isolate watermark inheritance from generic fine-tuning effects or establish a selection benefit.')
    if c['skip_utility']: warnings.append('FID/KID explicitly skipped; no distributional quality conclusion.')
    by={r['model']:r for r in table}
    if by['teacher_wm_test']['detection_rate']<c['gates']['min_teacher_tpr']:
        warnings.append('Teacher watermark failed held-out positive control; low student detection cannot isolate a LoRA transmission failure.')
    teacher_row=by['teacher_wm_test']
    teacher_psnr=paired_psnr(run/'images/teacher_clean_test',run/'images/teacher_wm_test',c['test_n'])
    teacher_valid=(teacher_row['detection_rate']>=c['gates']['min_teacher_tpr']
        and teacher_row['ci95_low']>=c['gates']['min_teacher_ci_low']
        and by['teacher_clean_test']['detection_rate']<=c['gates']['max_clean_fpr']
        and teacher_psnr>=c['gates']['min_psnr']
        and by['teacher_clean_test']['semantic_clip']-teacher_row['semantic_clip']<=c['gates']['max_clip_drop'])
    status={}
    for rank in c['ranks']:
        effect=effects[f'r{rank}_wm_minus_unadapted']
        status[str(rank)]={'teacher_valid':teacher_valid,'design_selection_valid':selection['valid_design_winner'],
            'student_detection_floor_passed':by[f'lora_r{rank}_wm']['detection_rate']>=c['gates']['min_student_tpr'],
            'positive_difference_vs_unadapted_ci':effect['paired_ci95_pp'][0]>0,
            'unadapted_fpr_valid':by['unadapted']['detection_rate']<=c['gates']['max_clean_fpr'],
            'student_clip_gate_vs_unadapted':by['unadapted']['semantic_clip']-by[f'lora_r{rank}_wm']['semantic_clip']<=c['gates']['max_clip_drop']}
        status[str(rank)]['declared_gates_passed']=all(status[str(rank)].values())
    write_json(run/'result/teacher_strength.json', {'valid':teacher_valid,'requirements':c['gates'],'measurement':teacher_row,
        'clean_fpr':by['teacher_clean_test']['detection_rate'],'paired_psnr':teacher_psnr,
        'threshold':detector['thresholds'][0],'calibration_target_fpr':c['fpr'],'test_used_to_change_threshold':False})
    if not teacher_valid: warnings.append('TEACHER_INVALID: teacher detection/CI/quality/FPR guard failed; no successful inheritance claim.')
    updates={b['name']:read_json(run/'students'/b['name']/'complete.json').get('optimizer_update_counts') for b in branches(c)}
    write_json(run/'result/report.json',{'method':'transfer-selected semantic UNet watermark pilot','teacher_selection':selection,
        'results':table,'effects':effects,'warnings':warnings,'optimizer_updates':updates,'teacher_test_psnr':teacher_psnr,'teacher_valid':teacher_valid,'claim_checks':status,
        'clean_trained_controls':False,'causal_inheritance_established':False,
        'test_used_for_selection':False,'checkpoint_selection':'validation denoising MSE only'})
    csv_file(run/'result/summary.csv',table); csv_file(run/'result/checkpoint_metrics.csv',milestones)
    csv_file(run/'result/wrong_key_controls.csv',wrong)
    lines=['# Experimental UNet watermark results','',
        'One fixed detector threshold. Clean rows measure false positives; marked rows measure detection.', '',
        '| Model | Detected | Rate | 95% Wilson CI | FID | KID |', '|---|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['model']} | {r['detected']}/{r['n']} | {r['detection_rate']:.1%} | {r['ci95_low']:.1%}–{r['ci95_high']:.1%} | {r['fid']} | {r['kid']} |")
    lines+=['','Differences versus unadapted (not clean-trained adjusted): see report.json. Passing gates is descriptive, not proof of causal inheritance.','',*['- '+w for w in warnings]]
    (run/'result/README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    from serum_teacher import numerical_policy
    numerical_policy()
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('stage',choices=['assets','clean_design','detector','encode','teacher_train','export','teacher_design',
        'setup_pilots','pilot_generate','select_teacher','final_teacher','select_student','final_student','report'])
    p.add_argument('--step',type=int); p.add_argument('--name')
    a=p.parse_args(); r=a.run.resolve()
    simple={'assets':prepare,'clean_design':clean_design,'detector':detector_stage,'encode':encode_training,
            'teacher_train':train_teacher,'setup_pilots':setup_pilots,'select_teacher':select_teacher,'report':summarize}
    if a.stage in simple: simple[a.stage](r)
    elif a.stage in ('export','teacher_design'):
        if a.step is None: p.error('--step required')
        (export_teacher if a.stage=='export' else teacher_design)(r,a.step)
    else:
        if not a.name: p.error('--name required')
        {'pilot_generate':pilot_generate,'final_teacher':final_teacher_images,
         'select_student':select_student,'final_student':final_student_images}[a.stage](r,a.name)
