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
from typing import TYPE_CHECKING, Callable, Optional

import torch
from lmf_hooks.model import config, logger
from torch.utils.data import random_split

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
    gen_batch_size: Optional[int] = None,
    token_level_reward_fn: Optional[
        Callable[[list["torch.Tensor"], list["torch.Tensor"], "PreTrainedTokenizer"], list["torch.Tensor"]]
    ] = None,
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

    from lmf_hooks.model import hook_rl_model

    model = hook_rl_model(model)

    dataset = CustomDataset(workload_file, tokenizer, config(), generate=True)
    eval_dataset = None
    if eval_set_percent > 0:
        eval_size = int(len(dataset) * eval_set_percent)
        train_size = len(dataset) - eval_size
        g = torch.Generator().manual_seed(0)
        dataset, eval_dataset = random_split(dataset, [train_size, eval_size], g)
    elif eval_data_file and os.path.exists(eval_data_file):
        eval_dataset = CustomDataset(eval_data_file, tokenizer, config())

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
        data_collator=new_collator_rl(tokenizer.pad_token_id),
        training_data_collator=new_collator_rl(tokenizer.pad_token_id, padding_side="right"),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        eval_batch_size=eval_batch_size,
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
