# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/ppo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
from typing import TYPE_CHECKING, Callable, Optional

import torch
from index_advisor.cache import build_cache
from index_advisor.db import DBOption
from index_advisor.utils import compute_workloads_summary
from index_advisor.workload import load_workloads
from lmf_hooks.model import config, logger
from peft.peft_model import PeftModel

from scripts.sft_projector import CustomDataset, new_collator_rl

from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_ref_model, create_reward_model
from .trainer import CustomPPOTrainer


logger = logger.getChild("ppo.workflow")

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_ppo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
    *,
    workload_file: str = "",
    eval_data_file: Optional[str] = None,
    eval_set_percent: float = 0.0,
    eval_batch_size: int = 16,
    eval_size_limit: float,
    eval_count_limit: int,
    gen_batch_size: Optional[int] = None,
    token_level_reward_fn: Optional[
        Callable[[list["torch.Tensor"], list["torch.Tensor"], "PreTrainedTokenizer"], list["torch.Tensor"]]
    ] = None,
    db_option: DBOption,
    # Optional PPOConfig overrides
    ppo_adap_kl_ctrl: Optional[bool] = None,
    ppo_init_kl_coef: Optional[float] = None,
    ppo_kl_penalty: Optional[str] = None,
    ppo_target: Optional[float] = None,
    ppo_horizon: Optional[float] = None,
    ppo_early_stopping: Optional[bool] = None,
    ppo_target_kl: Optional[float] = None,
    ppo_ratio_threshold: Optional[float] = None,
    ppo_lam: Optional[float] = None,
    ppo_gamma: Optional[float] = None,
    ppo_cliprange: Optional[float] = None,
    ppo_cliprange_value: Optional[float] = None,
    ppo_vf_coef: Optional[float] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    from lmf_hooks.model import hook_tokenizer

    tokenizer = hook_tokenizer(tokenizer)
    assert isinstance(tokenizer.pad_token_id, int)

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    if isinstance(model.pretrained_model, PeftModel) and model_args.adapter_name_or_path:
        # load 一个 adapter 的副本以供 reference model 使用
        model.pretrained_model.load_adapter(model_args.adapter_name_or_path[-1], "raw")
    from lmf_hooks.model import hook_rl_model

    model = hook_rl_model(model)

    random.seed(0)
    workloads = load_workloads(workload_file)
    if eval_set_percent > 0:
        eval_size = int(len(workloads) * eval_set_percent)
        train_size = len(workloads) - eval_size
        logger.info(f"拆分数据集，训练集大小: {train_size}, 评估集大小: {eval_size}")
        random.shuffle(workloads)
        train_set = workloads[:train_size]
        eval_set = workloads[train_size:]
        # 为 eval_set 创建摘要以跨训练验证其一致性. 确保当 dataset 和 eval_set_percent 相同时，eval_dataset 在不同运行中保持不变
        eval_summary = compute_workloads_summary(eval_set)
        logger.info(f"评估数据集摘要: {eval_summary}")
    elif eval_data_file and os.path.exists(eval_data_file):
        train_set = workloads
        eval_set = load_workloads(eval_data_file)
    else:
        train_set = workloads
        eval_set = None

    eval_dataset = None
    if eval_set is not None:
        logger.info("为评估数据集构建成本和大小缓存...")
        t = time.time()
        # cost_cache, size_cache = build_cache(eval_set, db_option, 16)
        logger.info(f"评估数据集缓存构建完成，耗时 {time.time() - t:.2f} 秒")
        eval_dataset = CustomDataset(
            eval_set,
            tokenizer,
            config(),
            db_option.to_dict(),
            generate=True,
            extend=True,
            # cost_cache=cost_cache,
            # size_cache=size_cache,
        )
    logger.info("为训练数据集构建成本和大小缓存...")
    t = time.time()
    # cost_cache, size_cache = build_cache(train_set, db_option, 16)
    logger.info(f"训练数据集缓存构建完成，耗时 {time.time() - t:.2f} 秒")
    dataset = CustomDataset(
        train_set,
        tokenizer,
        config(),
        db_option.to_dict(),
        generate=True,
        # cost_cache=cost_cache,
        # size_cache=size_cache,
    )

    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    reward_model = create_reward_model(model, model_args, finetuning_args)

    # Initialize our Trainer
    ppo_trainer: CustomPPOTrainer = CustomPPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        model=model,
        reward_model=reward_model,
        ref_model=ref_model,
        ref_adapter_name="raw" if model_args.adapter_name_or_path else None,
        data_collator=new_collator_rl(tokenizer.pad_token_id),
        training_data_collator=new_collator_rl(tokenizer.pad_token_id, padding_side="right"),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        eval_batch_size=eval_batch_size,
        eval_size_limit=eval_size_limit,
        eval_count_limit=eval_count_limit,
        gen_batch_size=gen_batch_size,
        reward_fn=token_level_reward_fn,
        ppo_adap_kl_ctrl=ppo_adap_kl_ctrl,
        ppo_init_kl_coef=ppo_init_kl_coef,
        ppo_kl_penalty=ppo_kl_penalty,
        ppo_target=ppo_target,
        ppo_horizon=ppo_horizon,
        ppo_early_stopping=ppo_early_stopping,
        ppo_target_kl=ppo_target_kl,
        ppo_ratio_threshold=ppo_ratio_threshold,
        ppo_lam=ppo_lam,
        ppo_gamma=ppo_gamma,
        ppo_cliprange=ppo_cliprange,
        ppo_cliprange_value=ppo_cliprange_value,
        ppo_vf_coef=ppo_vf_coef,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        ppo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        ppo_trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

        ppo_trainer.save_state()  # must be called after save_model to have a folder
        if ppo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
