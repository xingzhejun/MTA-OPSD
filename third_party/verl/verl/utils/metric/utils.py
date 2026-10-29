# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Metrics utils.
"""

import re
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch


_IGNORE_NAN_KEY = re.compile(
    r"^rlcsd_(?:w_tilde|lambda_t|snr|e_ctr|selected_residual)_(?:p\d+|mean|min|max)$"
    r"|^rlcsd_selected_tokens_per_sample_mean$"
    r"|^distill_routing_teacher_\d+_(?:winner_distance|winner_loss)_mean$"
    r"|^distill_routing_winner_teacher_(?:avg_logprob|entropy)$"
)
_SUM_KEY = re.compile(
    r"^rlcsd_(?:selected_token_count|response_token_count|sample_count)$"
    r"|^distill_routing_(?:eligible_token_count|final_token_count|near_tie_token_count|margin_token_count)$"
    r"|^distill_routing_(?:final_)?adjacent_(?:pair|switch)_count$"
    r"|^distill_routing_teacher_\d+_(?:assigned|final)_token_count$"
    r"|^distill_routing_(?:high_entropy|rest_entropy)_token_count$"
    r"|^distill_routing_(?:high_prefix|rest_suffix)_[0-9_]+_count$"
    r"|^distill_routing_(?:high|rest)_teacher_\d+_count$"
    r"|^distill_routing_(?:high_prefix|rest_suffix)_distance_gap_(?:sum|token_count)$"
    r"|^distill_routing_(?:high_nearest|rest_farthest)_distance_gap_(?:sum|token_count)$"
)

_RLCSD_DERIVABLE_PREFIXES = ("rlcsd",)


def _finite_values(values: list[Any]) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    return value_array[np.isfinite(value_array)]


def _reduce_mean_metric(key: str, values: list[Any]) -> float:
    if _IGNORE_NAN_KEY.fullmatch(key):
        finite_values = _finite_values(values)
        if finite_values.size == 0:
            return float("nan")
        return float(np.mean(finite_values))
    return np.mean(values)


def _reduce_min_metric(key: str, values: list[Any]) -> float:
    if _IGNORE_NAN_KEY.fullmatch(key):
        finite_values = _finite_values(values)
        if finite_values.size == 0:
            return float("nan")
        return float(np.min(finite_values))
    return np.min(values)


def _reduce_max_metric(key: str, values: list[Any]) -> float:
    if _IGNORE_NAN_KEY.fullmatch(key):
        finite_values = _finite_values(values)
        if finite_values.size == 0:
            return float("nan")
        return float(np.max(finite_values))
    return np.max(values)


def _reduce_sum_metric(key: str, values: list[Any]) -> float:
    if _SUM_KEY.fullmatch(key):
        finite_values = _finite_values(values)
        if finite_values.size == 0:
            return float("nan")
        return float(np.sum(finite_values))
    return np.sum(values)


def _safe_div(numerator: Any, denominator: Any) -> float:
    denominator = float(denominator)
    if not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator) / denominator


def _derive_rlcsd_count_metrics_for_prefix(metrics: dict[str, Any], prefix: str) -> None:
    selected_token_count = metrics.get(f"{prefix}_selected_token_count")
    response_token_count = metrics.get(f"{prefix}_response_token_count")
    sample_count = metrics.pop(f"{prefix}_sample_count", None)

    if selected_token_count is not None and response_token_count is not None:
        metrics[f"{prefix}_selected_token_ratio"] = _safe_div(selected_token_count, response_token_count)
    if selected_token_count is not None and sample_count is not None:
        metrics[f"{prefix}_selected_tokens_per_sample_mean"] = _safe_div(selected_token_count, sample_count)


def _derive_rlcsd_count_metrics(metrics: dict[str, Any]) -> None:
    for prefix in _RLCSD_DERIVABLE_PREFIXES:
        _derive_rlcsd_count_metrics_for_prefix(metrics, prefix)


def _derive_opsd_routing_count_metrics(metrics: dict[str, Any]) -> None:
    eligible_count = metrics.get("distill_routing_eligible_token_count")
    final_count = metrics.get("distill_routing_final_token_count")
    if eligible_count is not None and final_count is not None:
        metrics["distill_routing_final_token_ratio"] = _safe_div(final_count, eligible_count)
    high_count = metrics.get("distill_routing_high_entropy_token_count")
    if high_count is not None and eligible_count is not None:
        metrics["distill_routing_high_entropy_ratio"] = _safe_div(high_count, eligible_count)

    for group in ("high_prefix", "rest_suffix", "high_nearest", "rest_farthest"):
        gap_sum = metrics.get(f"distill_routing_{group}_distance_gap_sum")
        gap_count = metrics.get(f"distill_routing_{group}_distance_gap_token_count")
        if gap_sum is not None and gap_count is not None:
            metrics[f"distill_routing_{group}_distance_gap_mean"] = _safe_div(
                gap_sum, gap_count
            )

    teacher_indices = set()
    teacher_count_pattern = re.compile(
        r"^distill_routing_teacher_(\d+)_(?:assigned|final)_token_count$"
    )
    for key in metrics:
        match = teacher_count_pattern.fullmatch(key)
        if match:
            teacher_indices.add(int(match.group(1)))
    for teacher_idx in teacher_indices:
        prefix = f"distill_routing_teacher_{teacher_idx}"
        assigned_count = metrics.get(f"{prefix}_assigned_token_count")
        teacher_final_count = metrics.get(f"{prefix}_final_token_count")
        if assigned_count is not None and eligible_count is not None:
            metrics[f"{prefix}_pre_filter_ratio"] = _safe_div(assigned_count, eligible_count)
        if teacher_final_count is not None and final_count is not None:
            metrics[f"{prefix}_post_filter_share"] = _safe_div(teacher_final_count, final_count)
        if assigned_count is not None and teacher_final_count is not None:
            metrics[f"{prefix}_retention_ratio"] = _safe_div(teacher_final_count, assigned_count)

    margin_count = metrics.get("distill_routing_margin_token_count")
    near_tie_count = metrics.get("distill_routing_near_tie_token_count")
    if margin_count is not None and near_tie_count is not None:
        metrics["distill_routing_near_tie_ratio"] = _safe_div(near_tie_count, margin_count)

    for prefix in ("distill_routing_adjacent", "distill_routing_final_adjacent"):
        pair_count = metrics.get(f"{prefix}_pair_count")
        switch_count = metrics.get(f"{prefix}_switch_count")
        if pair_count is not None and switch_count is not None:
            metrics[f"{prefix}_switch_rate"] = _safe_div(switch_count, pair_count)


def reduce_metrics(metrics: dict[str, Union["Metric", list[Any]]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if isinstance(val, Metric):
            metrics[key] = val.aggregate()
        elif _SUM_KEY.fullmatch(key):
            metrics[key] = _reduce_sum_metric(key, val)
        elif "max" in key:
            metrics[key] = _reduce_max_metric(key, val)
        elif "min" in key:
            metrics[key] = _reduce_min_metric(key, val)
        else:
            metrics[key] = _reduce_mean_metric(key, val)
    _derive_rlcsd_count_metrics(metrics)
    _derive_opsd_routing_count_metrics(metrics)
    return metrics


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


NumericType = int, float, torch.Tensor, np.ndarray
Numeric = int | float | torch.Tensor | np.ndarray


class Metric:
    """
    A metric aggregator for collecting and aggregating numeric values.

    This class accumulates numeric values (int, float, or scalar tensors) and computes
    an aggregate statistic based on the specified aggregation type (MEAN, SUM, MIN, or MAX).

    Args:
        aggregation: The aggregation method to use. Can be a string ("mean", "sum", "min", "max")
            or an AggregationType enum value.
        value: Optional initial value(s) to add. Can be a single numeric value or a list of values.

    Example:
        >>> metric = Metric(aggregation="mean", value=1.0)
        >>> metric.append(2.0)
        >>> metric.append(3.0)
        >>> metric.aggregate()
        2.0
    """

    def __init__(self, aggregation: str | AggregationType, value: Optional[Numeric | list[Numeric]] = None) -> None:
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.values = []
        if value is not None:
            self.append(value)

    def append(self, value: Union[Numeric, "Metric"]) -> None:
        if isinstance(value, Metric):
            self.extend(value)
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be converted to float")
            value = value.detach().item()
        if not isinstance(value, NumericType):
            raise ValueError(f"Unsupported value type: {type(value)}")
        self.values.append(value)

    def extend(self, values: Union["Metric", list[Numeric]]) -> None:
        if isinstance(values, Metric):
            if values.aggregation != self.aggregation:
                raise ValueError(f"Aggregation type mismatch: {self.aggregation} != {values.aggregation}")
            values = values.values
        for value in values:
            self.append(value)

    def aggregate(self) -> float:
        return self._aggregate(self.values, self.aggregation)

    @classmethod
    def _aggregate(cls, values: list[Numeric], aggregation: AggregationType) -> float:
        match aggregation:
            case AggregationType.MEAN:
                return np.mean(values)
            case AggregationType.SUM:
                return np.sum(values)
            case AggregationType.MIN:
                return np.min(values)
            case AggregationType.MAX:
                return np.max(values)

    @classmethod
    def aggregate_dp(cls, metric_lists: list["Metric"]) -> float:
        if not metric_lists:
            raise ValueError("Cannot aggregate an empty list of metrics.")
        value_lists = [ml.values for ml in metric_lists]
        if not all(len(ls) == len(value_lists[0]) for ls in value_lists):
            raise ValueError(
                f"All Metric instances must have the same number of values "
                f"for dp aggregation: {[len(ls) for ls in value_lists]}"
            )
        value_arrays = np.array(value_lists)  # [num_dp, num_grad_accumulation]
        aggregation = metric_lists[0].aggregation
        match aggregation:
            case AggregationType.SUM | AggregationType.MEAN:
                return cls._aggregate(
                    values=np.mean(value_arrays, axis=0), aggregation=aggregation
                )  # mean over dp ranks
            case AggregationType.MIN | AggregationType.MAX:
                return cls._aggregate(values=value_arrays.flatten(), aggregation=aggregation)  # min/max over all values

    @classmethod
    def from_dict(cls, data: dict[str, Numeric], aggregation: str | AggregationType) -> dict[str, "Metric"]:
        return {key: cls(value=value, aggregation=aggregation) for key, value in data.items()}

    def init_list(self) -> "Metric":
        return Metric(aggregation=self.aggregation)
