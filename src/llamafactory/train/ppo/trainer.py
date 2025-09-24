# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
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

import math
import os
import sys
import time
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from accelerate.utils import DistributedDataParallelKwargs
from index_advisor.ia_logging import logger as ia_logger
from index_advisor.mm.model import DATASET_COLUMNS, DATASET_SQLS
from lmf_hooks.reward import get_reward
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, replace_model, restore_layernorm


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)

mylogger = ia_logger.getChild("ppo.trainer")


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""Inherit PPOTrainer."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        training_data_collator: Callable,
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        eval_batch_size: int = 16,
        gen_batch_size: Optional[int] = None,
        # Optional PPOConfig overrides
        ppo_adap_kl_ctrl: Optional[bool] = None,
        ppo_init_kl_coef: Optional[float] = None,
        ppo_kl_penalty: Optional[str] = None,
        ppo_target: Optional[float] = None,
        ppo_horizon: Optional[float] = None,
        ppo_early_stopping: Optional[bool] = None,
        ppo_target_kl: Optional[float] = None,
    ) -> None:
        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        if gen_batch_size is None:
            gen_batch_size = training_args.per_device_train_batch_size
        assert (backward_batch_size * finetuning_args.ppo_buffer_size) % gen_batch_size == 0, (
            "gen_batch_size 必须能够整除 ppo batch_size"
        )
        self.gen_batch_size = gen_batch_size
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
            # 避免移除 columns 列. PPOTrainer 会根据 model.forward 的参数移除列, 但 PPO 看到的 model 是 AutoModelForCausalLMWithValueHead
            remove_unused_columns=False,
        )

        # Apply optional overrides to PPOConfig if provided via CLI
        if ppo_adap_kl_ctrl is not None:
            ppo_config.adap_kl_ctrl = ppo_adap_kl_ctrl
        if ppo_init_kl_coef is not None:
            ppo_config.init_kl_coef = ppo_init_kl_coef
        if ppo_kl_penalty is not None:
            ppo_config.kl_penalty = ppo_kl_penalty  # type: ignore[assignment]
        if ppo_target is not None:
            ppo_config.target = ppo_target
        if ppo_horizon is not None:
            ppo_config.horizon = ppo_horizon
        if ppo_early_stopping is not None:
            ppo_config.early_stopping = ppo_early_stopping
        if ppo_target_kl is not None:
            ppo_config.target_kl = ppo_target_kl

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            training_data_collator=training_data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.tokenizer = self.processing_class  # 避免打印 Trainer.tokenizer is now deprecated. 警告

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # eval dataset
        self.eval_collator = data_collator
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer."""
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")
        mylogger.info(f"  生成 batch size = {self.gen_batch_size:,}")
        mylogger.info(f"  PPO epochs = {self.config.ppo_epochs:,}")
        mylogger.info(f"  PPO batch size = {self.config.batch_size:,}")
        mylogger.info(f"  PPO backward batch size = {self.config.backward_batch_size:,}")
        mylogger.info(f"  PPO mini batch size = {self.config.mini_batch_size:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        gen_time_meter = AverageMeter()
        ppo_time_meter = AverageMeter()
        eval_time_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # 1. 首先在 train dataset 上进行一次 eval, 耗时较长
        # eval_reward = self.evaluate_(self.dataset)
        # logs = dict(eval_reward=eval_reward, epoch=0)
        # self.state.log_history.append(logs)
        # self.callback_handler.on_log(self.args, self.state, self.control, logs)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Get inputs
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            columns: list[torch.Tensor] = []
            sqls: list[torch.Tensor] = []
            t = time.time()
            for idx in range(0, self.config.batch_size, self.gen_batch_size):
                mini_batch = {
                    "input_ids": batch["input_ids"][idx : idx + self.gen_batch_size],
                    "attention_mask": batch["attention_mask"][idx : idx + self.gen_batch_size],
                }
                if DATASET_COLUMNS in batch:
                    mini_batch[DATASET_COLUMNS] = batch[DATASET_COLUMNS][idx : idx + self.gen_batch_size]
                if DATASET_SQLS in batch:
                    mini_batch[DATASET_SQLS] = batch[DATASET_SQLS][idx : idx + self.gen_batch_size]
                mini_batch_queries, mini_batch_responses, mini_batch_columns, mini_batch_sqls = self.get_inputs(
                    mini_batch
                )
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)
                if mini_batch_columns:
                    columns.extend(mini_batch_columns)
                if mini_batch_sqls:
                    sqls.extend(mini_batch_sqls)

            gen_time_meter.update((time.time() - t) / len(queries), n=len(queries))
            mylogger.info(
                f"batch 生成和 reward 耗时: {time.time() - t:.2f}s, 平均每样本 {(time.time() - t) / len(queries):.2f}s, 历史平均 {gen_time_meter.avg:.2f}s"
            )

            if columns:
                self._cached_columns = columns
            else:
                self._cached_columns = None
            if sqls:
                self._cached_sqls = sqls
            else:
                self._cached_sqls = None

            # Run PPO step
            self.model.train()
            # print(self.is_distributed)
            t = time.time()
            stats = self.step(queries, responses, rewards)
            ppo_time_meter.update((time.time() - t) / len(queries), n=len(queries))
            mylogger.info(
                f"batch PPO step 耗时: {time.time() - t:.2f}s, 平均每样本 {(time.time() - t) / len(queries):.2f}s, 历史平均 {ppo_time_meter.avg:.2f}s"
            )
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                    kl=stats["objective/kl"],
                    kl_coef=stats["objective/kl_coef"],
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                # eval checkpoint
                t = time.time()
                eval_reward = self.evaluate_(self.eval_dataset)
                eval_time_meter.update(time.time() - t)
                mylogger.debug(f"eval 耗时: {time.time() - t:.2f}s, 历史平均 {eval_time_meter.avg:.2f}s")
                logs = dict(eval_reward=eval_reward, epoch=round(step / steps_in_epoch, 2))
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    def evaluate_(self, eval_dataset: Dataset | None) -> float:
        if not eval_dataset:
            return 0
        if not self.eval_batch_size:
            self.eval_batch_size = self.config.mini_batch_size
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.eval_collator,
        )
        reward_meter = AverageMeter()
        self.model.eval()
        for batch in tqdm(dataloader, desc="Evaluating", disable=not self.is_local_process_zero()):
            for k, v in batch.items():
                batch[k] = v.to(self.current_device)
            queries, responses, _, _ = self.get_inputs(batch)
            rewards = self.get_rewards(queries, responses)
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))
        self.model.train()
        return round(reward_meter.avg, 4)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            # 为多模态模型分别设置参数组
            projector_params, lm_params = [], []

            # mylogger.debug(f"create_optimizer {type(model)=}")
            # mylogger.debug(f"create_optimizer {list(model.state_dict().keys())=}")

            # mylogger.debug(f"create_optimizer {type(model.pretrained_model)=}")
            # mylogger.debug(f"create_optimizer {list(model.pretrained_model.state_dict().keys())=}")
            decay_param_names = self.get_decay_parameter_names(model)
            named_parameters = list(model.named_parameters())
            for name, param in named_parameters:
                if param.requires_grad:
                    # 区分投影层和语言模型参数
                    if "multi_modal_projector" in name:
                        projector_params.append(param)
                    else:
                        lm_params.append(param)

                    # 原有的 weight decay 逻辑
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

            # 根据是否有多模态投影层来设置参数组
            if len(projector_params) > 0:
                # 为投影层和语言模型设置不同的学习率
                projector_lr = finetuning_args.projector_lr or training_args.learning_rate
                lm_lr = training_args.learning_rate
                mylogger.debug(f"为 projector 和 lm 使用不同的学习率: {projector_lr=} {lm_lr=}")

                param_groups = []

                # 投影层参数组
                for name, param in named_parameters:
                    if not param.requires_grad:
                        continue
                    if "multi_modal_projector" in name:
                        if name in decay_param_names:
                            param_groups.append(
                                {"params": [param], "lr": projector_lr, "weight_decay": training_args.weight_decay}
                            )
                            # mylogger.debug(f"添加 projector decay 参数组: {name} {param.size()}")
                        else:
                            param_groups.append({"params": [param], "lr": projector_lr, "weight_decay": 0.0})
                            # mylogger.debug(f"添加 projector 参数组: {name} {param.size()}")
                    else:
                        if name in decay_param_names:
                            param_groups.append(
                                {"params": [param], "lr": lm_lr, "weight_decay": training_args.weight_decay}
                            )
                            # mylogger.debug(f"添加 lm decay 参数组: {name} {param.size()}")
                        else:
                            param_groups.append({"params": [param], "lr": lm_lr, "weight_decay": 0.0})
                            # mylogger.debug(f"添加 lm 参数组: {name} {param.size()}")
            else:
                # 原有逻辑，适用于标准模型
                param_groups = [
                    dict(params=nodecay_params),
                    dict(params=decay_params, weight_decay=training_args.weight_decay),
                ]

            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(
        self, batch: dict[str, "torch.Tensor"]
    ) -> tuple[list["torch.Tensor"], list["torch.Tensor"], list["torch.Tensor"], list["torch.Tensor"]]:
        r"""Generate model's responses given queries."""
        # if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
        #     start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
        #     for k, v in batch.items():
        #         batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            generate_output: torch.Tensor = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        columns = []
        column = batch.get("columns", None)
        sqls = []
        sql = batch.get("sqls", None)
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right
            if column is not None:
                # 移除 columns 的 padding，columns 形状为 (batch_size, length, hidden_size)
                # padding 方式是 length 中后面几个全为 0*hidden_size 的零向量
                column_i = column[i]  # shape: (length, hidden_size)
                # 找到非零向量的位置，即去除后面的零向量 padding
                non_zero_mask = column_i.count_nonzero(dim=-1) > 0  # shape: (length,), type: bool
                if non_zero_mask.any():
                    # 找到最后一个非零向量的位置
                    last_non_zero_idx = non_zero_mask.nonzero()[-1].item() + 1
                    columns.append(column_i[:last_non_zero_idx])
                else:
                    # 如果全是零向量，保留一个零向量
                    columns.append(column_i[:1])
            if sql is not None:
                sql_i = sql[i]
                non_zero_mask = sql_i.count_nonzero(dim=-1) > 0
                if non_zero_mask.any():
                    last_non_zero_idx = non_zero_mask.nonzero()[-1].item() + 1
                    sqls.append(sql_i[:last_non_zero_idx])
                else:
                    sqls.append(sql_i[:1])

        return queries, responses, columns, sqls

    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        if self.finetuning_args.reward_model_type == "api":
            prompt = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
            resp = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            rewards = [get_reward(m) for m in zip(prompt, resp)]
            return [torch.tensor(r) for r in rewards]

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # use fp32 type

    @override
    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [{"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries]
            ).to(self.current_device)

            decoder_inputs = self.data_collator(
                [{"input_ids": r, "attention_mask": torch.ones_like(r)} for r in responses]
            ).to(self.current_device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            input_ids = [torch.cat([q, r], dim=-1) for q, r in zip(queries, responses)]
            inputs = []
            for i, ids in enumerate(input_ids):
                inputs.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
                if self._cached_columns is not None:
                    inputs[-1]["columns"] = self._cached_columns[i]
                if self._cached_sqls is not None:
                    inputs[-1]["sqls"] = self._cached_sqls[i]
            input_data = self.data_collator(inputs)

        input_data.pop("labels", None)  # we don't want to compute LM losses
        return input_data

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # from index_advisor.logging import logger

        # logger = logger.getChild("ppo.trainer")

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            # 实际执行
            # logger.debug(f"{type(self.model)=}")
            # logger.debug(f"{list(self.model.state_dict().keys())=}")
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            # logger.debug(f"{type(unwrapped_model)=}")
            # logger.debug(f"{list(unwrapped_model.state_dict().keys())=}")
            # list(unwrapped_model.state_dict().keys())=['v_head.summary.weight', 'v_head.summary.bias']
            # logger.debug(f"{type(unwrapped_model.pretrained_model)=}")
            # logger.debug(f"{list(unwrapped_model.pretrained_model.state_dict().keys())=}")
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
            column_projector_state_dict = {
                name.removeprefix("base_model.model."): param
                for name, param in unwrapped_model.pretrained_model.state_dict().items()
                if "multi_modal_projector" in name
            }
            save_file(
                column_projector_state_dict,
                os.path.join(output_dir, "projector.safetensors"),
                metadata={"format": "pt"},
            )
