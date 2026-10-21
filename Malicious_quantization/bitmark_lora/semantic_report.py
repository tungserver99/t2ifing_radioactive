"""Separate frozen nulls, held-out teacher quality, and semantic lineage tables."""
import csv
import math
from pathlib import Path
import torch
from common import finite, freeze, image_name, read_json, sha256, write_json
from semantic_carrier import SemanticCarrier, check_finite
from semantic_threshold import quality_passes,is_threshold,clip_drop_limit
from semantic_sign import is_sign


def score_rows(run,label,n,carrier=None):
    rows=read_json(run/'scores'/f'{label}.json')
    if len(rows)!=n or [r['file'] for r in rows]!=[image_name(i) for i in range(n)]:
        raise ValueError(f'Incomplete/unaligned semantic scores: {label}')
    finite([r['score'] for r in rows])
    carrier = carrier if carrier is not None else SemanticCarrier.load(run/'carrier')
    coordinates = torch.tensor([r['coordinates'] for r in rows], dtype=torch.float32)
    if coordinates.shape != (n, len(carrier.signs)):
        raise ValueError('Invalid semantic score coordinate shape')
    check_finite(coordinates)
    reconstructed = (coordinates * carrier.signs).mean(-1).tolist()
    if any(not math.isclose(expected, row['score'], rel_tol=1e-5, abs_tol=1e-6)
           for expected, row in zip(reconstructed, rows)):
        raise ValueError('Score does not match saved semantic coordinates and frozen key')
    for r in rows:
        if sha256(run/'images'/label/r['file'])!=r['sha256']: raise RuntimeError('Scored image changed')
    if is_sign(read_json(run/'config.json')):
        from semantic_sign_report import validate_image_evidence
        c=read_json(run/'config.json')
        provenance=read_json(run/'scores'/f'{label}.provenance.json')
        expected={'carrier_sha256':sha256(run/'carrier/complete.json'),
                  'prompt_sha256':sha256(run/'prompts.json'),'reference_assets_sha256':sha256(run/'assets.json'),
                  'batch_size':c['score_batch'],'dtype':'float32','probe_timestep':0,'posterior':'mean'}
        if provenance!=expected: raise RuntimeError('Semantic sign score provenance changed')
        validate_image_evidence(rows,carrier,c.get('analytic_alpha',c.get('alpha_test',.01)))
    return rows


def freeze_null(run, family):
    from report import calibrate
    c=read_json(run/'config.json')
    label='teacher_clean_calibration' if family=='teacher' else 'unadapted_calibration'
    rows=score_rows(run,label,c['calibration_n']); values=[r['score'] for r in rows]
    t=calibrate(values,c['fpr'])
    result={'family':family,'source':label,'threshold':t,'n':len(rows),'target_fpr':c['fpr'],
            'empirical_fpr':sum(v>=t for v in values)/len(values),'comparison':'>= threshold',
            'score_sha256':sha256(run/'scores'/f'{label}.json'),
            'carrier_sha256':sha256(run/'carrier/complete.json'),'test_used':False}
    freeze(run/'result'/f'threshold_{family}.json',result)
    return result


def freeze_thresholds(run):
    teacher=freeze_null(run,'teacher'); student=freeze_null(run,'student')
    result={'primary':'semantic_score','semantic_score':student['threshold'],
            'semantic_score_teacher':teacher['threshold'],'semantic_score_student':student['threshold'],
            'teacher':teacher,'student':student,'note':'Identical fixed semantic score; separate clean family nulls'}
    if is_sign(read_json(run/'config.json')):
        from semantic_sign import detection_count
        c=read_json(run/'config.json')
        result.update(primary='analytic_binomial',analytic_binomial=detection_count(c['num_bits'],c['analytic_alpha']),role='secondary_continuous_comparison',
                      note='Disjoint clean calibration for secondary continuous scores only; analytic verifier is primary')
    freeze(run/'result/threshold.json',result)
    return result


def detection_metrics(run,label,threshold,n):
    from report import wilson
    carrier=SemanticCarrier.load(run/'carrier')
    rows=score_rows(run,label,n,carrier=carrier); values=[r['score'] for r in rows]
    q=torch.tensor([r['coordinates'] for r in rows])
    k=sum(v>=threshold for v in values); ci=wilson(k,n)
    return {'branch':label,'threshold':threshold,'detected':k,'n':n,'detection_rate':k/n,
            'ci95_low':ci[0],'ci95_high':ci[1],'mean_score':sum(values)/n,**carrier.bit_metrics(q)}


def teacher_quality(run,label):
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    import lpips
    from utility import CLIP_REV
    from semantic_teacher import pixels
    c=read_json(run/'config.json'); rows=read_json(run/'prompts.json')['validation']
    sign_objective=is_sign(c)
    clean=run/'images/teacher_clean_validation'; fp=run/'images'/label
    identity={'clean':[sha256(clean/image_name(i)) for i in range(len(rows))],
              'fingerprinted':[sha256(fp/image_name(i)) for i in range(len(rows))],
              'clip_revision':CLIP_REV,'clip_drop_max':clip_drop_limit(c),
              'lpips_max':c['teacher_lpips_max'],'pairing':'identical prompt and Gaussian seed',
              'lpips':'alex, version=0.1, lpips package 0.1.4',
              'quality_policy':'clip_only_lpips_diagnostic' if is_threshold(c) else 'clip_and_lpips'}
    if sign_objective:
        identity.update(clip_drop_max=.01,lpips_max=.15,quality_policy='rssf_clip_lpips_hard_gates',
                        pixel_mse_range='[-1,1]',visual_sample_count=min(8,len(rows)))
    cg=read_json(clean/'generation.json'); fg=read_json(fp/'generation.json')
    if cg['seeds']!=fg['seeds'] or cg['prompts_sha256']!=fg['prompts_sha256']:
        raise ValueError('Teacher quality requires paired prompts and seeds')
    target=run/'result/teacher_candidates'/f'{label}.json'
    if target.exists():
        old=read_json(target)
        if old['identity']!=identity: raise RuntimeError('Teacher quality data/config changed')
        return old
    clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',revision=CLIP_REV).cuda().eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32',revision=CLIP_REV)
    perceptual=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    clips=[[],[]]; distances=[]
    for start in range(0,len(rows),c['score_batch']):
        batch=rows[start:start+c['score_batch']]
        paths=[[d/image_name(start+j) for j in range(len(batch))] for d in (clean,fp)]
        for j,files in enumerate(paths):
            ims=[]
            for p in files:
                with Image.open(p) as im: ims.append(im.convert('RGB'))
            inputs=processor(text=[r['text'] for r in batch],images=ims,return_tensors='pt',padding=True,truncation=True).to('cuda')
            with torch.no_grad():
                out=clip(**inputs)
                similarity=(out.image_embeds*out.text_embeds).sum(-1).clamp_min(0)*2.5
                check_finite(similarity); clips[j]+=similarity.cpu().tolist()
        with torch.no_grad():
            distance=perceptual(pixels(paths[0]),pixels(paths[1])).flatten()
            check_finite(distance); distances+=distance.cpu().tolist()
    clean_clip=sum(clips[0])/len(rows); fp_clip=sum(clips[1])/len(rows); lp=sum(distances)/len(rows)
    result={'identity':identity,'n':len(rows),'clean_clip':clean_clip,'fp_clip':fp_clip,
            'clip_drop':clean_clip-fp_clip,'paired_lpips':lp,
            'passes':quality_passes(c,clean_clip-fp_clip,lp),
            'lpips_is_hard_gate':not is_threshold(c),
            'clip_definition':'same Serum utility: 2.5 * max(cos(image,text),0)',
            'clip_per_image_clean':clips[0],'clip_per_image_fp':clips[1],'lpips_per_image':distances}
    if sign_objective:
        from semantic_sign_report import paired_pixel_visuals
        result.update(paired_pixel_visuals(run,'teacher_clean_validation',label,len(rows)))
        result.update(passes=math.isfinite(clean_clip-fp_clip) and math.isfinite(lp) and
                      clean_clip-fp_clip<=.01 and lp<=.15,lpips_is_hard_gate=True)
    freeze(target,result)
    return result


def summarize(run):
    if is_sign(read_json(run/'config.json')):
        from semantic_sign_report import summarize as sign_summary
        return sign_summary(run)
    from student import branches
    c=read_json(run/'config.json'); threshold=freeze_thresholds(run)
    labels=['teacher_clean_test','teacher_wm_test','unadapted']+[b['name'] for b in branches(c)]
    utility=read_json(run/'result/utility.json') if (run/'result/utility.json').exists() else None
    table=[]
    for label in labels:
        family='teacher' if label.startswith('teacher_') else 'student'
        row=detection_metrics(run,label,threshold[family]['threshold'],c['test_n'])
        quality=(utility or {}).get('branches',{}).get(label,{})
        row.update(fid=quality.get('frechet_inception_distance'),kid=quality.get('kernel_inception_distance_mean'),
                   clip=quality.get('clip_score'),threshold_family=family)
        table.append(row)
    selection={b['name']:read_json(run/'result/selection'/f"{b['name']}.json") for b in branches(c)}
    warnings=['One training seed; image confidence intervals do not measure training-seed uncertainty.',
              'Unadapted baseline contrasts do not isolate generic fine-tuning effects without clean-trained controls.',
              'Teacher strength uses held-out teacher validation only; students selected solely by denoising MSE.']
    if not utility: warnings.append('Student utility unavailable (explicitly skipped); quality preservation is unverified.')
    baseline=next(r for r in table if r['branch']=='unadapted')
    if baseline['ci95_low']>c['fpr']: warnings.append('Independent unadapted test FPR exceeds nominal level; inspect null generalization.')
    freeze(run/'result/selection_manifest.json',selection)
    write_json(run/'result/radioactivity.json',{'method':'semantic_fp','threshold':threshold,'table':table,
        'selected_checkpoints':selection,'teacher_quality':read_json(run/'result/teacher_quality.json'),
        'warnings':warnings,'utility':utility})
    fields=['branch','threshold_family','threshold','detected','n','detection_rate','ci95_low','ci95_high','mean_score','bit_accuracy','fid','kid','clip']
    with (run/'result/detection_semantic.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(table)
    teacher_quality_report = read_json(run/'result/teacher_quality.json')
    quality_table = []
    parameter='kappa' if 'kappa' in teacher_quality_report['selected'] else 'alpha'
    for candidate in teacher_quality_report['candidates']:
        q = candidate['quality']; metrics = candidate['metrics']
        quality_table.append({parameter: candidate[parameter],
            'selected': candidate[parameter] == teacher_quality_report['selected'][parameter],
            'clean_clip': q['clean_clip'], 'fingerprinted_clip': q['fp_clip'], 'clip_drop': q['clip_drop'],
            'paired_lpips': q['paired_lpips'], 'quality_passes': q['passes'],
            'validation_tpr': metrics['detection_rate'], 'mean_score': metrics['mean_score'],
            'bit_accuracy': metrics['bit_accuracy']})
    with (run/'result/teacher_quality.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(quality_table[0]))
        writer.writeheader(); writer.writerows(quality_table)
    lines=['# Native semantic fingerprint: SD2.1 to SD1.4','',
           f"Teacher/student use their respective frozen clean {100*c['fpr']:g}% FPR null thresholds.",'',
           '| Model | Detection | Mean score | Bit accuracy | FID | KID | CLIP |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        fmt=lambda x:'unavailable' if x is None else f'{x:.5g}'
        lines.append(f"| {r['branch']} | {r['detection_rate']:.2%} | {r['mean_score']:.5g} | {r['bit_accuracy']:.1%} | {fmt(r['fid'])} | {fmt(r['kid'])} | {fmt(r['clip'])} |")
    lines += ['', '## Paired teacher validation quality', '',
              f'| {parameter.title()} | Selected | Clean CLIP | Fingerprinted CLIP | CLIP drop | Paired LPIPS | Quality pass |',
              '|---|---|---:|---:|---:|---:|---|']
    for q in quality_table:
        lines.append(f"| {q[parameter]:g} | {q['selected']} | {q['clean_clip']:.5g} | {q['fingerprinted_clip']:.5g} | {q['clip_drop']:.5g} | {q['paired_lpips']:.5g} | {q['quality_passes']} |")
    lines+=['','Teacher paired quality details: teacher_quality.json / teacher_quality.csv. Student checkpoint curves: checkpoint_metrics.csv.','']
    lines += ['- '+w for w in warnings]
    (run/'result/README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--teacher-null-only',action='store_true'); a=p.parse_args()
    if a.teacher_null_only: freeze_null(a.run.resolve(),'teacher')
    else: summarize(a.run.resolve())
