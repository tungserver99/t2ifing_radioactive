"""Versioned threshold-calibrated coherent shift; fixed carrier/verifier unchanged."""
import math
import statistics
import torch
from common import freeze,read_json,sha256
from semantic_carrier import check_finite

OBJECTIVE='threshold_cfg_v1'
CFG_SCALE=7.5
FP_WEIGHT=.8
CYCLE_WEIGHT=.5
SHIFT_CAP=2.5


def is_threshold(config):return config.get('teacher_objective')==OBJECTIVE


def candidate_parameter(config):return 'kappa' if is_threshold(config) else 'alpha'


def cfg_prediction(conditional,unconditional):
    result=unconditional.float()+CFG_SCALE*(conditional.float()-unconditional.float())
    check_finite(result)
    return result


def calibrated_shift_loss(clean_q,fp_q,signs,tau,sigma,kappa):
    if not all(math.isfinite(x) for x in (tau,sigma,kappa)) or sigma<=0 or kappa<=0:
        raise ValueError('Invalid frozen calibration/target')
    clean_q=clean_q.detach().float();fp_q=fp_q.float()
    check_finite(clean_q,fp_q,signs)
    r0=(clean_q*signs).mean(-1);rfp=(fp_q*signs).mean(-1)
    raw_shift=tau+kappa*sigma-r0
    required=raw_shift.clamp(0.,SHIFT_CAP).detach()
    loss=(fp_q-clean_q-required[:,None]*signs[None,:]).square().mean()
    check_finite(loss)
    return loss,{'mean_r0':r0.mean().item(),'mean_rfp':rfp.detach().mean().item(),
                 'mean_required_shift':required.mean().item(),
                 'proxy_tpr':(rfp.detach()>=tau).float().mean().item(),
                 'shift_cap_fraction':(raw_shift>SHIFT_CAP).float().mean().item()}


def clip_drop_limit(config):
    return min(config['teacher_clip_drop_max'],.01) if is_threshold(config) else config['teacher_clip_drop_max']


def quality_passes(config,clip_drop,lpips):
    if not math.isfinite(clip_drop) or not math.isfinite(lpips):return False
    return (clip_drop<=clip_drop_limit(config) and
            (is_threshold(config) or lpips<=config['teacher_lpips_max']))


def frozen_calibration(run):
    """Reverify the original null; never fit statistics from teacher validation/test."""
    from semantic_report import freeze_null,score_rows
    path=run/'result/threshold_teacher.json'
    if not path.exists():raise RuntimeError('Teacher clean threshold must be frozen before training')
    c=read_json(run/'config.json');proof=freeze_null(run,'teacher')
    rows=score_rows(run,'teacher_clean_calibration',c['calibration_n'])
    sigma=statistics.pstdev(row['score'] for row in rows)  # declared ddof=0
    if not math.isfinite(sigma) or sigma<=0:raise ValueError('Degenerate teacher calibration scores')
    result={'objective':OBJECTIVE,'threshold':proof['threshold'],'sigma_clean':sigma,'std_ddof':0,
            'n':proof['n'],'target_fpr':proof['target_fpr'],'empirical_fpr':proof['empirical_fpr'],
            'false_positives':sum(row['score']>=proof['threshold'] for row in rows),
            'threshold_sha256':sha256(path),'score_sha256':proof['score_sha256'],
            'carrier_sha256':proof['carrier_sha256'],'data_source':'teacher_clean_calibration',
            'cfg_scale':CFG_SCALE,'shift_cap':SHIFT_CAP,'fp_weight':FP_WEIGHT,'cycle_weight':CYCLE_WEIGHT,
            'targets':[{'kappa':k,'target_score':proof['threshold']+k*sigma} for k in c['teacher_kappas']],
            'test_used':False,'validation_used':False}
    freeze(run/'teacher_fp/calibration_target.json',result)
    return result
