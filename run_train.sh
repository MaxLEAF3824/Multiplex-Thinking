#!/bin/bash
set -x

# export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY="${WANDB_API_KEY:-}"
echo "[WARN] WANDB_API_KEY is empty. Set it via env var if you want wandb logging."


# export NCCL_TIMEOUT=36000
# export NCCL_SOCKET_IFNAME=ibp24s0
# export NCCL_IB_HCA=mlx5_4
export NCCL_CUMEM_HOST_ENABLE=0

############## ray_node_setup.sh ##############
echo ${MASTER_ADDR}
echo $OMPI_COMM_WORLD_RANK

# Generate a random integer between 1 and 100000
WANDB_PROJECT="multiplex-thinking"
EXP_NAME="qwen3-1.7b-base-mt"
RANDOM_INT=$((RANDOM % 100000 + 1))
echo "Random integer: $RANDOM_INT"
EXP_NAME="${EXP_NAME}-${RANDOM_INT}"
echo EXP_NAME: $EXP_NAME
# data.val_les=deepscaler/hdfs_data/$VAL_DATASET.parquet \

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=deepscaler/hdfs_data/train.parquet \
    data.val_files=[/home/aiscuser/SofT-GRPO-master/Soft-Thinking+noise+loss-main/datasets/aime.parquet,/home/aiscuser/SofT-GRPO-master/Soft-Thinking+noise+loss-main/datasets/amc.parquet,/home/aiscuser/SofT-GRPO-master/Soft-Thinking+noise+loss-main/datasets/math.parquet] \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    data.truncation=right \
    +data.use_online_transform=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768  \
    actor_rollout_ref.actor.policy_loss.loss_mode=multiplex_thinking \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.enable_soft_thinking=True \
    actor_rollout_ref.rollout.enable_mixed_rollout=False \
    actor_rollout_ref.rollout.after_thinking_top_p=1.0 \
    actor_rollout_ref.rollout.after_thinking_top_k=-1 \
    actor_rollout_ref.rollout.after_thinking_min_p=0.0 \
    actor_rollout_ref.rollout.max_topk=3 \
    actor_rollout_ref.rollout.used_topk=3 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.early_stopping_entropy_threshold=-1.0 \
    actor_rollout_ref.rollout.early_stopping_length_threshold=256 \
    actor_rollout_ref.rollout.enable_entropy_mask=False \
    actor_rollout_ref.rollout.entropy_mask_threshold=False \
    actor_rollout_ref.rollout.enable_gumbel=False \
    actor_rollout_ref.rollout.gumbel_tau=1.0 \
    actor_rollout_ref.rollout.after_thinking_temperature=1.0 \
    actor_rollout_ref.rollout.enable_replacement=True \
    actor_rollout_ref.rollout.enable_gumbel_after_thinking=False \
    actor_rollout_ref.rollout.enable_unweighting=True \
    actor_rollout_ref.rollout.val_kwargs.enable_replacement=True \
    actor_rollout_ref.rollout.val_kwargs.enable_gumbel_after_thinking=False \
    actor_rollout_ref.rollout.val_kwargs.enable_unweighting=True \
    actor_rollout_ref.rollout.val_kwargs.enable_gumbel=False \
    actor_rollout_ref.rollout.val_kwargs.gumbel_tau=1.0 \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.early_stopping_entropy_threshold=-1.0 \
    actor_rollout_ref.rollout.val_kwargs.early_stopping_length_threshold=256 \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_min_p=0.0 \
    actor_rollout_ref.rollout.val_kwargs.max_topk=3 \
    actor_rollout_ref.rollout.val_kwargs.used_topk=3 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    +actor_rollout_ref.rollout.shuffle_before_dispatch=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.resume_mode="auto" \
    trainer.resume_from_path=null \
    reward_model.reward_manager=hf_math_verify \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_sleep_hack=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    reward_model.enable=False \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=./${WANDB_PROJECT}/${EXP_NAME}