#!/bin/bash

eval "$(conda shell.bash hook)"
source .bashrc

conda activate inf_water

# debug print-outs
echo USER: $USER
which conda
which python


echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3


LOCAL_OUT=/path/to/local_out
mkdir -p $LOCAL_OUT

export WANDB_MODE=offline
export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

lr=1e-4
epochs=1
pn=1M
bs=1

scales=2
delta=2

weights_path="/path/to/weights"
vae_ckpt="${weights_path}/infinity_vae_d32reg.pth"
infinity_path="${weights_path}/infinity_2b_reg.pth"
exp_name=stripe_mask_lpf1k_lr_${lr}_epoch_${epochs}_pn_${pn}_bs_${bs}_scales_${scales}_delta_${delta}
data_path=/path/to/metadata/jsonl/scales_${scales} # In here should be only the jsonl file, e.g., 1.000_0001000.jsonl
video_data_path=''
local_out_path=$LOCAL_OUT/${exp_name}


mkdir ${local_out_path}


port=$(($RANDOM%(36000-26000+1)+26000))

torchrun \
--master_port=${port} \
--nproc_per_node=1 \
train.py \
--ep=${epochs} \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--data_path=${data_path} \
--video_data_path=${video_data_path} \
--exp_name=${exp_name} \
--tblr=${lr} \
--pn ${pn} \
--model=2bc8 \
--bs=${bs} \
--workers=8 \
--short_cap_prob 0.5 \
--online_t5=1 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--Ct5=2048 \
--t5_path="google/flan-t5-xl" \
--vae_type 32 \
--vae_ckpt="${vae_ckpt}"  \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 1000000 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=False \
--pad=128 \
--rush_resume="${infinity_path}" \