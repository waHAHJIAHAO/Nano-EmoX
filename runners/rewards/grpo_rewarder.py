import csv
import math
from typing import Any, Dict, List, Set

import torch

import requests

from nano_emox.evaluation.ew_metric import (
    extract_openset_batchcalling,
    func_read_batch_calling_model,
)
from nano_emox.evaluation.wheel import (
    func_get_wheel_cluster,
    func_map_label_to_synonym,
    read_format2raws,
    read_candidate_synonym_merge,
)


class GRPORewarder:
    """
    GRPO 奖励器：
    - format reward
    - accuracy reward (with length penalty)
    - alignment reward
    - dual reward (with length penalty)

      hitrate 计算复用 OV-MER 评估链路中的 5-wheel 平均逻辑。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        run_cfg = cfg.run_cfg if cfg is not None else {}
        self._debug = run_cfg.get("grpo_reward_debug", False)

        self.weights = run_cfg.get(
            "grpo_reward_weights",
            {
                "format": 0.25,
                "accuracy": 0.25,
                "alignment": 0.25,
                "dual": 0.25,
            },
        )

        self.gt_csv_path = run_cfg.get(
            "grpo_gt_csv_path", "data/MER2025/mer2025-ov.csv"
        )
        self.name2gt = self._load_name2gt(self.gt_csv_path)

        self.format_mapping = read_format2raws()
        self.raw_mapping = read_candidate_synonym_merge()
        self.wheel_maps = {
            f"wheel{i}": func_get_wheel_cluster(f"wheel{i}", "level1")
            for i in range(1, 6)
        }

        self.modelname = run_cfg.get("grpo_reward_extract_model", "Qwen25_7B")
        self.use_remote_extractor = run_cfg.get("grpo_use_remote_extractor", True)
        self.extractor_host = run_cfg.get("grpo_extractor_host", "localhost")
        self.extractor_port = run_cfg.get("grpo_extractor_port", 18081)
        self.extractor_url = f"http://{self.extractor_host}:{self.extractor_port}"

        self.llm = None
        self.tokenizer = None
        self.sampling_params = None

    def _lazy_init_extractor(self):
        if self.use_remote_extractor:
            return
        if self.llm is None or self.tokenizer is None or self.sampling_params is None:
            self.llm, self.tokenizer, self.sampling_params = func_read_batch_calling_model(
                modelname=self.modelname
            )

    def _load_name2gt(self, csv_path: str) -> Dict[str, str]:
        name2gt: Dict[str, str] = {}
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("name") or "").strip()
                    ov = (row.get("openset") or "").strip()
                    if name:
                        name2gt[name] = ov
        except Exception:
            return {}
        return name2gt

    def _extract_between(self, text: str, left: str, right: str) -> str:
        if not isinstance(text, str):
            return ""
        l = text.find(left)
        r = text.find(right)
        if l == -1 or r == -1 or r <= l:
            return ""
        return text[l + len(left) : r].strip()

    def parse_output(self, output_text: str):
        ot = self._extract_between(output_text, "<think>", "</think>")
        oa = self._extract_between(output_text, "<answer>", "</answer>")
        return ot, oa

    def format_reward(self, output_text: str) -> float:
        if not isinstance(output_text, str):
            return 0.0

        required_tags = ["<think>", "</think>", "<answer>", "</answer>"]
        for tag in required_tags:
            if tag not in output_text:
                return 0.0

        think_start = output_text.find("<think>")
        think_end = output_text.find("</think>")
        answer_start = output_text.find("<answer>")
        answer_end = output_text.find("</answer>")

        if not (0 <= think_start < think_end < answer_start < answer_end):
            return 0.0

        answer_content = output_text[answer_start + len("<answer>"):answer_end]
        items = [item.strip() for item in answer_content.split(",")]
        if len(items) < 1:
            return 0.0

        for item in items:
            if not item:
                return 0.0

        return 1.0

    def _to_list(self, x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            result = []
            for i in x:
                s = str(i).strip().lower()
                s = s.strip("[]").strip("()").strip("{}")
                if s:
                    result.append(s)
            return result
        s = str(x).strip()
        if not s:
            return []
        s = s.replace(";", ",")
        s = s.strip("[]").strip("()").strip("{}")
        items = [t.strip().lower() for t in s.split(",") if t.strip()]
        return items

    def _list_to_ov_string(self, labels: List[str]) -> str:
        return ",".join([x for x in labels if x])

    def _map_labels_to_wheel(self, labels: List[str], wheel_map: dict, format_mapping: dict, raw_mapping: dict) -> Set[str]:
        mapped = func_map_label_to_synonym(labels, format_mapping, raw_mapping, wheel_map, metric='case3_wheel1_level1')
        return set([item.lower().strip() for item in mapped if item])

    def _f1_score(self, pred_set: Set[str], gt_set: Set[str]) -> float:
        if len(pred_set) == 0 and len(gt_set) == 0:
            return 1.0
        if len(pred_set) == 0 or len(gt_set) == 0:
            return 0.0
        intersection = len(pred_set & gt_set)
        precision = intersection / len(pred_set)
        recall = intersection / len(gt_set)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _jaccard_similarity(self, pred_set: Set[str], gt_set: Set[str]) -> float:
        if len(pred_set) == 0 and len(gt_set) == 0:
            return 1.0
        union = pred_set | gt_set
        if len(union) == 0:
            return 0.0
        intersection = len(pred_set & gt_set)
        return intersection / len(union)

    def _f1_5wheel_avg(self, pred_list: List[str], gt_list: List[str]) -> float:
        if self._debug:
            print(f"\n[DEBUG _f1_5wheel_avg] pred_list ({len(pred_list)}): {pred_list}")
            print(f"[DEBUG _f1_5wheel_avg] gt_list ({len(gt_list)}): {gt_list}")
        if len(pred_list) == 0 or len(gt_list) == 0:
            if self._debug:
                print(f"[DEBUG _f1_5wheel_avg] empty list detected, returning 0.0")
            return 0.0

        pred_lower = [item.lower().strip() for item in pred_list]
        gt_lower = [item.lower().strip() for item in gt_list]

        f1_scores = []
        for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4', 'wheel5']:
            wheel_map = self.wheel_maps[wheel_name]
            pred_mapped = self._map_labels_to_wheel(pred_lower, wheel_map, self.format_mapping, self.raw_mapping)
            gt_mapped = self._map_labels_to_wheel(gt_lower, wheel_map, self.format_mapping, self.raw_mapping)
            f1 = self._f1_score(pred_mapped, gt_mapped)
            f1_scores.append(f1)
            if self._debug:
                print(f"[DEBUG _f1_5wheel_avg] {wheel_name}: pred_mapped={pred_mapped}, gt_mapped={gt_mapped}, F1={f1:.4f}")

        avg_f1 = sum(f1_scores) / len(f1_scores)
        if self._debug:
            print(f"[DEBUG _f1_5wheel_avg] avg_f1={avg_f1:.4f}")
        return avg_f1

    def _jaccard_5wheel_avg(self, pred_list: List[str], gt_list: List[str]) -> float:
        if self._debug:
            print(f"\n[DEBUG _jaccard_5wheel_avg] pred_list ({len(pred_list)}): {pred_list}")
            print(f"[DEBUG _jaccard_5wheel_avg] gt_list ({len(gt_list)}): {gt_list}")
        if len(pred_list) == 0 or len(gt_list) == 0:
            if self._debug:
                print(f"[DEBUG _jaccard_5wheel_avg] empty list detected, returning 0.0")
            return 0.0

        pred_lower = [item.lower().strip() for item in pred_list]
        gt_lower = [item.lower().strip() for item in gt_list]

        jaccard_scores = []
        for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4', 'wheel5']:
            wheel_map = self.wheel_maps[wheel_name]
            pred_mapped = self._map_labels_to_wheel(pred_lower, wheel_map, self.format_mapping, self.raw_mapping)
            gt_mapped = self._map_labels_to_wheel(gt_lower, wheel_map, self.format_mapping, self.raw_mapping)
            jaccard = self._jaccard_similarity(pred_mapped, gt_mapped)
            jaccard_scores.append(jaccard)
            if self._debug:
                print(f"[DEBUG _jaccard_5wheel_avg] {wheel_name}: pred_mapped={pred_mapped}, gt_mapped={gt_mapped}, Jaccard={jaccard:.4f}")

        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
        if self._debug:
            print(f"[DEBUG _jaccard_5wheel_avg] avg_jaccard={avg_jaccard:.4f}")
        return avg_jaccard

    def length_penalty_p3(self, pred_list: List[str], gt_list: List[str]) -> float:
        lp = len(pred_list)
        lg = len(gt_list)
        if lp <= lg:
            return 1.0
        return math.log(lg + 1) / max(math.log(lp + 1), 1e-12)

    def _extract_emotions_with_llm(self, texts: List[str]) -> List[List[str]]:
        clean_texts = [t if isinstance(t, str) and t.strip() else "neutral" for t in texts]
        if len(clean_texts) == 0:
            return []

        if self.use_remote_extractor:
            try:
                out = []
                for i, text in enumerate(clean_texts):
                    # if self._debug:
                    #     print(f"\n[DEBUG _extract_emotions_with_llm] [{i+1}/{len(clean_texts)}] ot_text: {text[:100]}...")
                    resp = requests.post(
                        f"{self.extractor_url}/extract",
                        json={"texts": [text]},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    emotions = data.get("emotions", [])
                    extracted = emotions[0] if isinstance(emotions, list) and len(emotions) > 0 else []
                    out.append(extracted)
                    # if self._debug:
                    #     print(f"[DEBUG _extract_emotions_with_llm] [{i+1}/{len(clean_texts)}] extracted: {extracted}")
                return out
            except Exception as e:
                print(f"[WARNING] Remote extractor failed: {e}, falling back to local")
                self.use_remote_extractor = False

        self._lazy_init_extractor()

        out = []
        for i, text in enumerate(clean_texts):
            if self._debug:
                print(f"\n[DEBUG _extract_emotions_with_llm] [{i+1}/{len(clean_texts)}] ot_text: {text[:100]}...")
            name2reason = {f"tmp_{i}": text}
            names, responses = extract_openset_batchcalling(
                name2reason=name2reason,
                llm=self.llm,
                tokenizer=self.tokenizer,
                sampling_params=self.sampling_params,
            )
            extracted = self._to_list(responses[0]) if responses else []
            out.append(extracted)
            if self._debug:
                print(f"[DEBUG _extract_emotions_with_llm] [{i+1}/{len(clean_texts)}] extracted: {extracted}")
        return out

    def _resolve_ground_truths(self, samples: Dict[str, Any], batch_size: int) -> List[List[str]]:
        for key in ["ovlabel", "gt_ov", "openset", "y", "y_text", "target_ov"]:
            if key in samples:
                val = samples[key]
                if isinstance(val, list):
                    return [self._to_list(v) for v in val]

        for key in ["names", "name", "sample_names"]:
            if key in samples and isinstance(samples[key], list):
                gts = []
                for n in samples[key]:
                    gts.append(self._to_list(self.name2gt.get(str(n), "")))
                return gts

        return [[] for _ in range(batch_size)]

    def accuracy_reward_with_penalty(self, oa_list: List[str], y_list: List[str]) -> float:
        base = self._f1_5wheel_avg(oa_list, y_list)
        penalty = self.length_penalty_p3(oa_list, y_list)
        return penalty * base

    def alignment_reward(self, ot_text: str, oa_list: List[str]) -> float:
        et_list = self._extract_emotions_with_llm([ot_text])[0]
        return self._jaccard_5wheel_avg(et_list, oa_list)

    def dual_reward_with_penalty(self, ot_text: str, y_list: List[str]) -> float:
        et_list = self._extract_emotions_with_llm([ot_text])[0]
        base = self._f1_5wheel_avg(et_list, y_list)
        penalty = self.length_penalty_p3(et_list, y_list)
        return penalty * base

    def compute_rewards(
        self,
        samples: Dict[str, Any],
        responses: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        n = len(responses)
        y_lists = self._resolve_ground_truths(samples, n)

        reward_format = torch.zeros(n, device=device, dtype=torch.float32)
        reward_accuracy = torch.zeros(n, device=device, dtype=torch.float32)
        reward_alignment = torch.zeros(n, device=device, dtype=torch.float32)
        reward_dual = torch.zeros(n, device=device, dtype=torch.float32)

        for i, o in enumerate(responses):
            ot, oa = self.parse_output(o)
            oa_list = self._to_list(oa)
            y_list = y_lists[i] if i < len(y_lists) else []

            r_format = self.format_reward(o)
            r_acc = self.accuracy_reward_with_penalty(oa_list, y_list)
            r_align = self.alignment_reward(ot, oa_list)
            r_dual = self.dual_reward_with_penalty(ot, y_list)

            reward_format[i] = float(r_format)
            reward_accuracy[i] = float(r_acc)
            reward_alignment[i] = float(r_align)
            reward_dual[i] = float(r_dual)

        reward_total = (
            float(self.weights.get("format", 0.0)) * reward_format
            + float(self.weights.get("accuracy", 0.0)) * reward_accuracy
            + float(self.weights.get("alignment", 0.0)) * reward_alignment
            + float(self.weights.get("dual", 0.0)) * reward_dual
        )

        return {
            "reward_total": reward_total,
            "reward_format": reward_format,
            "reward_accuracy": reward_accuracy,
            "reward_alignment": reward_alignment,
            "reward_dual": reward_dual,
        }
