import copy
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from nano_emox.common.logger import MetricLogger, SmoothedValue
from nano_emox.common.registry import registry
from nano_emox.datasets.data_utils import prepare_sample
from nano_emox.runners.rewards.grpo_rewarder import GRPORewarder
from nano_emox.tasks.base_task import BaseTask


@registry.register_task("grpo_task")
class GRPOTask(BaseTask):
    """
    GRPO 训练任务
    """

    def __init__(self):
        super().__init__()
        self._debug = False

    def _model_no_ddp(self, model):
        return model.module if hasattr(model, "module") else model

    def _build_inputs_embeds_and_mask(
        self,
        model,
        samples: Dict[str, Any],
        prompt_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self._model_no_ddp(model)

        m.face_or_frame = samples["face_or_frame"]
        frame_llms, face_llms, audio_llms, image_llms, multi_llms = None, None, None, None, None
        frame_hiddens, face_hiddens, audio_hiddens = None, None, None

        if "frames" in samples:
            frame_hiddens, frame_llms = m.encode_video_merge(samples["frames"], samples["raw_frames"])
        if "faces" in samples:
            face_hiddens, face_llms = m.encode_video_merge(samples["faces"], samples["raw_faces"])
        if "audios" in samples:
            audio_hiddens, audio_llms = m.encode_audio_merge(samples["audios"], samples["raw_audios"])
        if "images" in samples:
            _, image_llms = m.encode_image_merge(samples["images"], samples["raw_images"])

        if (samples["input_ids"][0] == m.MULTI_PATCH_TOKEN_ID).sum() != 0:
            if m.face_or_frame.startswith("multiface") and (face_hiddens is not None) and (audio_hiddens is not None):
                _, multi_llms = m.encode_multi_merge(face_hiddens, audio_hiddens)
            if m.face_or_frame.startswith("multiframe") and (frame_hiddens is not None) and (audio_hiddens is not None):
                _, multi_llms = m.encode_multi_merge(frame_hiddens, audio_hiddens)

        input_ids = samples["input_ids"]
        attention_masks = samples["attention_masks"]

        if prompt_only:
            labels = samples["labels"]
            prompt_rows = []
            max_prompt_len = 1
            bsz = input_ids.shape[0]
            for bi in range(bsz):
                row_ids = input_ids[bi]
                row_labels = labels[bi]
                row_attn = attention_masks[bi].bool()
                prompt_mask = (row_labels == -100) & row_attn
                idx = torch.nonzero(prompt_mask, as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    idx = torch.nonzero(row_attn, as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    cur_prompt = row_ids[:1]
                else:
                    cur_prompt = row_ids[: int(idx[-1].item()) + 1]
                prompt_rows.append(cur_prompt)
                if cur_prompt.numel() > max_prompt_len:
                    max_prompt_len = cur_prompt.numel()

            pad_id = m.llama_tokenizer.pad_token_id if hasattr(m, "llama_tokenizer") else 0
            new_input_ids = torch.full(
                (input_ids.shape[0], max_prompt_len),
                fill_value=pad_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            new_attention_masks = torch.zeros(
                (input_ids.shape[0], max_prompt_len),
                dtype=attention_masks.dtype,
                device=attention_masks.device,
            )
            for bi, row in enumerate(prompt_rows):
                L = row.numel()
                new_input_ids[bi, :L] = row
                new_attention_masks[bi, :L] = 1

            input_ids = new_input_ids
            attention_masks = new_attention_masks

        temp_input_ids = copy.deepcopy(input_ids)
        temp_input_ids[temp_input_ids == m.FRAME_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == m.FACE_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == m.AUDIO_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == m.MULTI_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == m.IMAGE_PATCH_TOKEN_ID] = 0

        temp_input_embedding = m.llama_model.model.model.embed_tokens(temp_input_ids)

        cur_idx = 0
        new_input_embeds = []
        for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
            for (patch_token_id, query_token_number, embeds) in [
                (m.FRAME_PATCH_TOKEN_ID, m.num_video_query_token, frame_llms),
                (m.FACE_PATCH_TOKEN_ID, m.num_video_query_token, face_llms),
                (m.AUDIO_PATCH_TOKEN_ID, m.num_audio_query_token, audio_llms),
                (m.MULTI_PATCH_TOKEN_ID, m.num_multi_query_token, multi_llms),
                (m.IMAGE_PATCH_TOKEN_ID, m.num_image_query_token, image_llms),
            ]:
                if (cur_input_ids == patch_token_id).sum() != 0:
                    assert embeds is not None, "Some input info is missing."
                    cur_features = embeds[cur_idx]
                    masked_indices = torch.where(cur_input_ids == patch_token_id)[0]
                    mask_index_start = masked_indices[0]
                    cur_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:mask_index_start],
                            cur_features,
                            cur_input_embeds[mask_index_start + query_token_number :],
                        ),
                        dim=0,
                    )

            new_input_embeds.append(cur_input_embeds)
            cur_idx += 1

        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        attention_mask = attention_masks
        return inputs_embeds, attention_mask

    def _rollout_group(
        self,
        model,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        m = self._model_no_ddp(model)

        out = m.llama_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=group_size,
            return_dict_in_generate=True,
        )

        gen_ids = out.sequences
        if self._debug:
            print(f"\n[DEBUG _rollout_group] prompt_embeds shape: {prompt_embeds.shape}, prompt_mask sum: {prompt_mask.sum().item()}")
            print(f"[DEBUG _rollout_group] gen_ids shape: {gen_ids.shape}")
            print(f"[DEBUG _rollout_group] gen_ids[0] (first 30 tokens): {gen_ids[0][:30].tolist()}")
            print(f"[DEBUG _rollout_group] gen_ids[0] (last 20 tokens): {gen_ids[0][-20:].tolist() if gen_ids.shape[1] > 20 else gen_ids[0].tolist()}")

        tokenizer = m.llama_tokenizer if hasattr(m, "llama_tokenizer") else None
        responses = []
        if tokenizer is not None:
            for i in range(gen_ids.shape[0]):
                responses.append(tokenizer.decode(gen_ids[i], skip_special_tokens=True))
        else:
            responses = [""] * group_size

        return gen_ids, responses

    def _compute_logprob_for_rollout(
        self,
        model,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        gen_ids: torch.Tensor,
    ) -> torch.Tensor:
        m = self._model_no_ddp(model)

        if gen_ids.numel() == 0:
            return torch.zeros(prompt_embeds.shape[0], device=prompt_embeds.device)

        gen_embeds = m.llama_model.model.model.embed_tokens(gen_ids)
        full_embeds = torch.cat([prompt_embeds, gen_embeds], dim=1)

        gen_mask = torch.ones_like(gen_ids)
        full_mask = torch.cat([prompt_mask, gen_mask], dim=1)

        outputs = m.llama_model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            return_dict=True,
            use_cache=False,
        )

        logits = outputs.logits
        L = prompt_embeds.shape[1]
        T = gen_ids.shape[1]

        if self._debug:
            print(f"\n[DEBUG _compute_logprob] full_embeds shape: {full_embeds.shape}, full_mask sum: {full_mask.sum().item()}")
            print(f"[DEBUG _compute_logprob] logits shape: {logits.shape}, L={L}, T={T}")
            print(f"[DEBUG _compute_logprob] logits unique count per sample: {[torch.unique(logits[0, L-1+j]).numel() for j in range(min(3, T))]}")

        pred_logits = logits[:, L - 1 : L - 1 + T, :]
        token_logps = F.log_softmax(pred_logits, dim=-1).gather(
            dim=-1, index=gen_ids.unsqueeze(-1)
        ).squeeze(-1)
        return token_logps.sum(dim=1)

    def _compute_group_advantages(self, rewards: torch.Tensor, group_size: int, eps: float = 1e-6):
        if rewards.numel() == 0:
            return rewards
        grouped = rewards.view(-1, group_size)
        mu = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, unbiased=False, keepdim=True)
        return ((grouped - mu) / (std + eps)).reshape(-1)

    def _expand_samples_for_group(self, samples: Dict[str, Any], group_size: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        bsz = samples["input_ids"].shape[0]
        for k, v in samples.items():
            if isinstance(v, list) and len(v) == bsz:
                vv = []
                for item in v:
                    vv.extend([item] * group_size)
                out[k] = vv
            else:
                out[k] = v
        return out

    def train_epoch_grpo(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        run_cfg=None,
    ):
        inner_epoch = epoch
        iters_per_epoch = lr_scheduler.iters_per_epoch
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        clip_eps = float(getattr(run_cfg, "grpo_clip_eps", 0.2)) if run_cfg is not None else 0.2
        group_size = int(getattr(run_cfg, "grpo_group_size", 4)) if run_cfg is not None else 4
        max_new_tokens = int(getattr(run_cfg, "grpo_max_new_tokens", 512))
        temperature = float(getattr(run_cfg, "grpo_temperature", 0.8))
        top_p = float(getattr(run_cfg, "grpo_top_p", 0.9))

        rewarder = GRPORewarder(type("CfgHolder", (), {"run_cfg": run_cfg}) if run_cfg is not None else None)
        rewarder._debug = self._debug

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("reward", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("ratio", SmoothedValue(window_size=1, fmt="{value:.6f}"))

        logging.info("Start GRPO training epoch {}, {} iters.".format(epoch, iters_per_epoch))
        header = "GRPO Train: [{}]".format(epoch)

        optimizer.zero_grad()

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            if isinstance(samples, dict):
                samples.update({"epoch": inner_epoch, "num_iters_per_epoch": iters_per_epoch, "iters": i})

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            prompt_embeds, prompt_mask = self._build_inputs_embeds_and_mask(model, samples, prompt_only=True)
            bsz = prompt_embeds.shape[0]

            all_gen_ids, all_responses = [], []

            with torch.no_grad():
                for b in range(bsz):
                    gen_ids, responses = self._rollout_group(
                        model=model,
                        prompt_embeds=prompt_embeds[b : b + 1],
                        prompt_mask=prompt_mask[b : b + 1],
                        group_size=group_size,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    all_gen_ids.append(gen_ids)
                    all_responses.extend(responses)

            gen_ids = torch.cat(all_gen_ids, dim=0)

            prompt_embeds_rep = prompt_embeds.repeat_interleave(group_size, dim=0)
            prompt_mask_rep = prompt_mask.repeat_interleave(group_size, dim=0)

            if torch.__version__.startswith("2.1.0"):
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        old_logprob = self._compute_logprob_for_rollout(
                            model=model,
                            prompt_embeds=prompt_embeds_rep,
                            prompt_mask=prompt_mask_rep,
                            gen_ids=gen_ids,
                        ).detach()
            else:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        old_logprob = self._compute_logprob_for_rollout(
                            model=model,
                            prompt_embeds=prompt_embeds_rep,
                            prompt_mask=prompt_mask_rep,
                            gen_ids=gen_ids,
                        ).detach()

            reward_samples = self._expand_samples_for_group(samples, group_size=group_size)
            reward_dict = rewarder.compute_rewards(
                samples=reward_samples,
                responses=all_responses,
                device=gen_ids.device,
            )
            reward_total = reward_dict["reward_total"]
            advantages = self._compute_group_advantages(reward_total, group_size=group_size)

            if torch.__version__.startswith("2.1.0"):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    new_logprob = self._compute_logprob_for_rollout(
                        model=model,
                        prompt_embeds=prompt_embeds_rep,
                        prompt_mask=prompt_mask_rep,
                        gen_ids=gen_ids,
                    )
            else:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    new_logprob = self._compute_logprob_for_rollout(
                        model=model,
                        prompt_embeds=prompt_embeds_rep,
                        prompt_mask=prompt_mask_rep,
                        gen_ids=gen_ids,
                    )

            log_ratio = new_logprob - old_logprob
            log_ratio_clamped = torch.clamp(log_ratio, -3, 3)
            ratio = torch.exp(log_ratio_clamped)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            if self._debug and i == 0:
                print(f"\n[DEBUG iter {i}]")
                print(f"  new_logprob: {new_logprob[:4].tolist() if new_logprob.numel() > 4 else new_logprob.tolist()}")
                print(f"  old_logprob: {old_logprob[:4].tolist() if old_logprob.numel() > 4 else old_logprob.tolist()}")
                print(f"  log_ratio (new-old): {log_ratio[:4].tolist() if log_ratio.numel() > 4 else log_ratio.tolist()}")
                print(f"  log_ratio_clamped: {log_ratio_clamped[:4].tolist() if log_ratio_clamped.numel() > 4 else log_ratio_clamped.tolist()}")
                print(f"  ratio: {ratio[:4].tolist() if ratio.numel() > 4 else ratio.tolist()}")
                print(f"  advantages: {advantages[:4].tolist() if advantages.numel() > 4 else advantages.tolist()}")
                print(f"  surr1: {surr1[:4].tolist() if surr1.numel() > 4 else surr1.tolist()}")
                print(f"  surr2: {surr2[:4].tolist() if surr2.numel() > 4 else surr2.tolist()}")
            loss = -torch.min(surr1, surr2).mean() / accum_grad_iters

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item() * accum_grad_iters)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(reward=reward_total.mean().item() if reward_total.numel() > 0 else 0.0)
            metric_logger.update(ratio=ratio.mean().item())

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
