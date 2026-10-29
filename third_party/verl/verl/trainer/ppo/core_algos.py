# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
import math
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

# new added
import os

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


# new added
def _debug_print_reward_tensor(reward_tensor: torch.Tensor):
    """Print reward tensor metadata and compact correctness-oriented views."""
    reward_cpu = reward_tensor.detach().float().cpu()
    nonzero_indices = torch.nonzero(reward_cpu, as_tuple=False)
    print("\n" + "=" * 100)
    print("[DEBUG reward_tensor before token_level_scores assignment]")
    print(f"shape={tuple(reward_tensor.shape)}, dtype={reward_tensor.dtype}, device={reward_tensor.device}")
    print(f"content=\n{reward_cpu}")
    print(f"per_rollout_sum={reward_cpu.sum(dim=-1).tolist()}")
    print(f"nonzero_indices={nonzero_indices.tolist()}")
    if nonzero_indices.numel() > 0:
        print(f"nonzero_values={reward_cpu[tuple(nonzero_indices.T)].tolist()}")
    print("=" * 100 + "\n", flush=True)



def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
    GDPO = "gdpo"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GDPO)  # or simply: @register_adv_est("gdpo")
def compute_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    batch: Optional[dict] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GDPO: Group reward-Decoupled Normalization Policy Optimization.

    Instead of summing all reward dimensions first (like GRPO), GDPO normalizes
    each reward dimension independently within each group before aggregation.
    This prevents a dominant reward signal from drowning out weaker ones.

    Mathematical formulation:
        Step 1 – Group-wise decoupled normalization (via GRPO per dimension):
            For each reward dimension k, within each group g:
            A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)

        Step 2 – Weighted aggregation:
            A_sum = Σ_k w_k · A_k

        Step 3 – Batch-level normalization (via masked_whiten):
            A_final = whiten(A_sum, response_mask)

    Args:
        token_level_rewards: (bs, response_length) – standard token-level rewards.
            Used as fallback when per-dimension rewards are not provided.
        response_mask: (bs, response_length)
        index: (bs) – group id per sample (from ``uid``).
        epsilon: Numerical stability constant.
        norm_adv_by_std_in_grpo: Whether to normalize by std in GRPO.
        config: Algorithm configuration (optional).
        non_tensor_batch: Non-tensor batch data containing per-dimension reward scores.
        batch: Batch data containing prompts, attention_mask, etc.

    Note:
        Ref GDPO (https://arxiv.org/abs/2601.05242).

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) – same as advantages (outcome-only).
    """
    score_list = None
    reward_weights = None

    if config is not None and non_tensor_batch is not None and batch is not None:
        gdpo_reward_keys = config.get("gdpo_reward_keys", None)
        assert gdpo_reward_keys, (
            "GDPO requires 'algorithm.gdpo_reward_keys' listing the individual reward "
            "component keys returned by compute_score (e.g. ['format_reward', 'accuracy_reward'])."
        )
        device = token_level_rewards.device
        prompt_length = batch["prompts"].size(1)
        valid_response_length = batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1

        score_list = []
        for key in gdpo_reward_keys:
            assert key in non_tensor_batch, (
                f"GDPO reward key '{key}' not found in non_tensor_batch. "
                f"Available keys: {list(non_tensor_batch.keys())}. "
                f"Make sure your compute_score returns a dict containing '{key}'."
            )
            comp = non_tensor_batch[key]
            rm_score = torch.tensor(np.asarray(comp, dtype=np.float32), device=device)
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(rm_scores.size(0), device=device), valid_response_length] = rm_score
            score_list.append(rm_scores)

        gdpo_weights = config.get("gdpo_reward_weights", None)
        if gdpo_weights is not None:
            reward_weights = list(gdpo_weights)

    if score_list is None:
        score_list = [token_level_rewards]

    num_scores = len(score_list)

    if reward_weights is not None:
        weights = torch.tensor(reward_weights, dtype=torch.float32, device=token_level_rewards.device)
    else:
        weights = torch.ones(num_scores, dtype=torch.float32, device=token_level_rewards.device)

    new_advantage = None

    for i in range(num_scores):
        normalized_score, _ = compute_grpo_outcome_advantage(
            token_level_rewards=score_list[i],
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config)

        if new_advantage is None:
            new_advantage = weights[i] * normalized_score
        else:
            new_advantage += weights[i] * normalized_score

    advantages = verl_F.masked_whiten(new_advantage, response_mask) * response_mask

    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: True
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n)]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                horizon = loss_mask.shape[-1]
                loss_scale_factor = horizon
            loss /= loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-TV.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the TV divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-TV
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    valid_positive_mask = (prob - old_prob) <= clip_divergence_high
    valid_negative_mask = (prob - old_prob) >= -clip_divergence_low
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("dppo_kl")
def compute_policy_loss_dppo_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-KL.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the KL divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-KL
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * torch.log(
        (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
    )
    valid_positive_mask = (binary_kl <= clip_divergence_high) | (prob <= old_prob)
    valid_negative_mask = (binary_kl <= clip_divergence_low) | (prob >= old_prob)
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard DPPO)
    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expected value of KL, but the expected gradient of k1 and k3
    estimator is not the expected gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info)

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item()}

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold)
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights)

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics


# ================================================================== #
# Self-distillation losses (OPSD / SDPO / RLSD)
# Algorithm params are read from config.policy_loss (set via Hydra overrides)
# ================================================================== #

def _sd_token_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1)


def _sd_exclusion_ratio_train_mask(values, valid_mask, ratio, exclude_high):
    """Return a bool train mask after excluding an exact ratio of valid entries."""
    train_flat = valid_mask.bool().flatten().clone()
    valid_flat_indices = train_flat.nonzero(as_tuple=False).squeeze(-1)
    num_rejected = math.ceil(valid_flat_indices.numel() * ratio)
    effective_threshold = 0.0
    if num_rejected > 0:
        valid_values = values.detach().float().flatten()[valid_flat_indices]
        rejected_local_indices = torch.topk(
            valid_values, k=num_rejected, largest=exclude_high, sorted=False
        ).indices
        rejected_flat_indices = valid_flat_indices[rejected_local_indices]
        train_flat[rejected_flat_indices] = False
        rejected_values = values.detach().float().flatten()[rejected_flat_indices]
        effective_threshold = (
            rejected_values.min().item() if exclude_high else rejected_values.max().item()
        )
    return train_flat.view_as(valid_mask), effective_threshold


def _sd_valid_minmax_normalize(values, valid_mask, upper_quantile=None, eps=1e-12):
    """Min-max normalize valid entries, optionally clipping their upper tail first."""
    normalized = torch.zeros_like(values, dtype=torch.float32)
    valid_flat_indices = valid_mask.bool().flatten().nonzero(as_tuple=False).squeeze(-1)
    if valid_flat_indices.numel() == 0:
        return normalized

    valid_values = values.detach().float().flatten()[valid_flat_indices]
    if upper_quantile is not None:
        cap = torch.quantile(valid_values, upper_quantile)
        valid_values = valid_values.clamp(max=cap)
    value_min = valid_values.min()
    value_range = valid_values.max() - value_min
    if value_range.item() > eps:
        normalized_values = (valid_values - value_min) / value_range
    else:
        normalized_values = torch.zeros_like(valid_values)
    normalized.flatten()[valid_flat_indices] = normalized_values
    return normalized


def _sd_soft_or_train_mask(student_entropy, distance, valid_mask, exclusion_ratio):
    """Exclude low Soft-OR scores after per-mask entropy/distance normalization."""
    normalized_entropy = _sd_valid_minmax_normalize(
        student_entropy, valid_mask, upper_quantile=0.98
    )
    normalized_distance = _sd_valid_minmax_normalize(distance, valid_mask)
    soft_or_score = (
        normalized_entropy
        + normalized_distance
        - normalized_entropy * normalized_distance
    )
    train_mask, effective_threshold = _sd_exclusion_ratio_train_mask(
        soft_or_score, valid_mask, exclusion_ratio, exclude_high=False
    )
    return train_mask, effective_threshold, soft_or_score


def _sd_cfg(config, key, default):
    """Read algorithm param from config.policy_loss."""
    if config is None:
        return default
    pl = getattr(config, 'policy_loss', None)
    if pl is None:
        return default
    if hasattr(pl, 'get'):
        value = pl.get(key, default)
    else:
        value = getattr(pl, key, default)
    return default if value is None else value


def _sd_teacher_metrics(teacher_lp, log_prob, response_mask, teacher_entropy=None):
    """Common teacher metrics for all self-distillation losses."""
    metrics = {}
    if teacher_lp is not None:
        metrics["teacher_gap"] = _sd_token_mean((teacher_lp - log_prob).abs(), response_mask).item()
        metrics["teacher_avg_logprob"] = _sd_token_mean(teacher_lp, response_mask).item()
        metrics["student_avg_logprob"] = _sd_token_mean(log_prob, response_mask).item()
    if teacher_entropy is not None:
        metrics["teacher_entropy"] = _sd_token_mean(teacher_entropy, response_mask).item()
    return metrics


def _sd_bool_cfg(config, key, default):
    return bool(_sd_cfg(config, key, default))


def _sd_int_cfg(config, key, default):
    return int(_sd_cfg(config, key, default))


def _sd_group_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (values * weights).sum(dim=1) / denom


def _sd_group_var(values: torch.Tensor, valid_mask: torch.Tensor, mean: torch.Tensor | None = None) -> torch.Tensor:
    if mean is None:
        mean = _sd_group_mean(values, valid_mask)
    weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    centered = values - mean.unsqueeze(1)
    return (centered.square() * weights).sum(dim=1) / denom


def _sd_distribution_metrics(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    prefix: str,
    include_percentiles: bool = True,
) -> dict[str, float]:
    valid_mask = response_mask.bool() & torch.isfinite(values)
    valid_values = torch.masked_select(values.float(), valid_mask)
    if valid_values.numel() == 0:
        base = {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan")}
        if include_percentiles:
            base.update({
                f"{prefix}_p05": float("nan"),
                f"{prefix}_p25": float("nan"),
                f"{prefix}_p50": float("nan"),
                f"{prefix}_p75": float("nan"),
                f"{prefix}_p95": float("nan")})
        return base

    base = {
        f"{prefix}_mean": valid_values.mean().item(),
        f"{prefix}_min": valid_values.min().item(),
        f"{prefix}_max": valid_values.max().item()}
    if not include_percentiles:
        return base
    quantiles = torch.quantile(
        valid_values,
        torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=valid_values.device))
    p05, p25, p50, p75, p95 = [quantile.item() for quantile in quantiles]
    base.update({
        f"{prefix}_p05": p05,
        f"{prefix}_p25": p25,
        f"{prefix}_p50": p50,
        f"{prefix}_p75": p75,
        f"{prefix}_p95": p95})
    return base


def _sd_effective_lam(config, global_steps, default=0.5):
    """Linearly decay lambda over lam_decay_steps, keeping step 1 at the initial value."""
    lam = float(_sd_cfg(config, "lam", default))
    lam_decay_steps = int(_sd_cfg(config, "lam_decay_steps", 0) or 0)
    if lam_decay_steps <= 0 or global_steps is None:
        return lam

    step_index = max(int(global_steps) - 1, 0)
    decay_progress = min(step_index / lam_decay_steps, 1.0)
    return max(lam * (1.0 - decay_progress), 0.0)


def _sd_prepare_distill_log_probs(
    student_all_log_probs,
    teacher_all_log_probs,
    top_k=0,
    add_tail=True,
    top_k_source="student"):
    """Prepare aligned student/teacher log-probs for full-vocab or top-k distillation."""
    assert student_all_log_probs is not None and teacher_all_log_probs is not None, (
        "full_logit_distill=True requires student_all_log_probs and teacher_all_log_probs."
    )

    if top_k <= 0:
        return student_all_log_probs, teacher_all_log_probs

    top_k = min(top_k, student_all_log_probs.shape[-1])
    source_log_probs = student_all_log_probs if top_k_source == "student" else teacher_all_log_probs
    _, topk_indices = torch.topk(source_log_probs.detach(), k=top_k, dim=-1)
    student_topk = torch.gather(student_all_log_probs, dim=-1, index=topk_indices)
    teacher_topk = torch.gather(teacher_all_log_probs, dim=-1, index=topk_indices)

    if add_tail:
        def _tail_log_prob(log_probs):
            log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_mass = torch.clamp(log_mass, max=-1e-7)
            return torch.log(-torch.expm1(log_mass))

        student_tail = _tail_log_prob(student_topk)
        teacher_tail = _tail_log_prob(teacher_topk)
        student_topk = torch.cat([student_topk, student_tail], dim=-1)
        teacher_topk = torch.cat([teacher_topk, teacher_tail], dim=-1)
    else:
        student_topk = student_topk - torch.logsumexp(student_topk, dim=-1, keepdim=True)
        teacher_topk = teacher_topk - torch.logsumexp(teacher_topk, dim=-1, keepdim=True)

    return student_topk, teacher_topk


def _sd_prepare_sparse_topk_log_probs(student_topk_log_probs, teacher_topk_log_probs, add_tail=True):
    """Finalize aligned sparse top-k log-probs by adding a tail bucket or renormalizing."""
    assert student_topk_log_probs is not None and teacher_topk_log_probs is not None, (
        "Sparse top-k distillation requires student_topk_log_probs and teacher_topk_log_probs."
    )

    if add_tail:
        def _tail_log_prob(log_probs):
            log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_mass = torch.clamp(log_mass, max=-1e-7)
            return torch.log(-torch.expm1(log_mass))

        student_tail = _tail_log_prob(student_topk_log_probs)
        teacher_tail = _tail_log_prob(teacher_topk_log_probs)
        student_topk_log_probs = torch.cat([student_topk_log_probs, student_tail], dim=-1)
        teacher_topk_log_probs = torch.cat([teacher_topk_log_probs, teacher_tail], dim=-1)
    else:
        student_topk_log_probs = student_topk_log_probs - torch.logsumexp(
            student_topk_log_probs, dim=-1, keepdim=True
        )
        teacher_topk_log_probs = teacher_topk_log_probs - torch.logsumexp(
            teacher_topk_log_probs, dim=-1, keepdim=True
        )

    return student_topk_log_probs, teacher_topk_log_probs


def _sd_generalized_jsd(student_log_probs, teacher_log_probs, mix, vocab_clip=None):
    """Generalized JSD / KL over log-prob distributions.

    If vocab_clip is a positive float, the per-(token, vocab) divergence
    contribution is clamped at vocab_clip BEFORE summing over the vocab
    dimension. This matches the official OPSD reference implementation, where
    `F.kl_div(..., reduction="none")` (shape [B, S, V]) is clipped per element
    and then summed across both vocab and tokens. With vocab_clip=None (the
    default), behavior is unchanged: KL contributions are summed over the
    vocab dimension first and any clipping is left to the caller.
    """
    apply_vocab_clip = vocab_clip is not None and vocab_clip > 0

    if mix == 0.0:
        per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        if apply_vocab_clip:
            per_vocab = per_vocab.clamp(max=vocab_clip)
        return per_vocab.sum(-1)
    if mix == 1.0:
        per_vocab = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        if apply_vocab_clip:
            per_vocab = per_vocab.clamp(max=vocab_clip)
        return per_vocab.sum(-1)

    mix_tensor = torch.tensor(mix, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log1p(-mix_tensor),
                teacher_log_probs + torch.log(mix_tensor),
            ]
        ),
        dim=0)
    if apply_vocab_clip:
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        per_vocab = torch.lerp(kl_student, kl_teacher, mix_tensor)
        return per_vocab.clamp(max=vocab_clip).sum(-1)
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(-1)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(-1)
    return torch.lerp(kl_student, kl_teacher, mix_tensor)


def _sd_paint_interpolate_teacher(
    student_log_probs,
    teacher_log_probs,
    student_entropy,
    teacher_entropy,
    response_mask,
    strength,
    top_fraction,
):
    """Apply PAINT geometric teacher interpolation at per-response top entropy-ratio positions."""
    if not (0.0 < strength < 1.0):
        raise ValueError(f"paint_interp_strength must be in (0, 1), got {strength}.")
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError(f"paint_top_fraction must be in (0, 1], got {top_fraction}.")
    if student_entropy is None or teacher_entropy is None:
        raise ValueError("PAINT interpolation requires both student_entropy and teacher_entropy.")
    if student_entropy.shape != response_mask.shape or teacher_entropy.shape != response_mask.shape:
        raise ValueError(
            "PAINT entropy tensors must match response_mask shape, got "
            f"student={tuple(student_entropy.shape)}, teacher={tuple(teacher_entropy.shape)}, "
            f"mask={tuple(response_mask.shape)}."
        )

    valid_mask = response_mask.bool()
    entropy_ratio = teacher_entropy.detach().float() / student_entropy.detach().float().clamp(min=1e-12)
    selected_mask = torch.zeros_like(valid_mask)
    for batch_idx in range(valid_mask.shape[0]):
        valid_indices = valid_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            continue
        selected_count = math.ceil(valid_indices.numel() * top_fraction)
        selected_local = torch.topk(
            entropy_ratio[batch_idx, valid_indices],
            k=selected_count,
            largest=True,
            sorted=False,
        ).indices
        selected_mask[batch_idx, valid_indices[selected_local]] = True

    # Energies are -log(p), so energy interpolation is equivalent to a
    # normalized convex combination of teacher and detached student log-probs.
    interpolated = (1.0 - strength) * teacher_log_probs + strength * student_log_probs.detach()
    interpolated = interpolated - torch.logsumexp(interpolated, dim=-1, keepdim=True)
    calibrated_teacher = torch.where(selected_mask.unsqueeze(-1), interpolated, teacher_log_probs)
    return calibrated_teacher, selected_mask, entropy_ratio


@register_policy_loss("opsd")
def compute_policy_loss_opsd(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """OPSD: Forward KL distillation (teacher -> student)."""
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "OPSD requires teacher_log_probs but got None. "
        "Ensure teacher_mode=fixed and use_kl_loss=True so ref_log_prob is computed."
    )
    beta = float(_sd_cfg(config, "beta", 0.0))
    teacher_selection_beta_cfg = _sd_cfg(config, "teacher_selection_beta", None)
    teacher_selection_beta = (
        beta if teacher_selection_beta_cfg is None else float(teacher_selection_beta_cfg)
    )
    if not 0.0 <= teacher_selection_beta <= 1.0:
        raise ValueError(
            "policy_loss.teacher_selection_beta must be in [0, 1], got "
            f"{teacher_selection_beta}."
        )
    jsd_token_clip = float(_sd_cfg(config, "jsd_token_clip", 0.05))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)
    paint_interp_strength = float(_sd_cfg(config, "paint_interp_strength", 0.0) or 0.0)
    paint_top_fraction = float(_sd_cfg(config, "paint_top_fraction", 0.0) or 0.0)
    paint_enabled = paint_interp_strength > 0.0 or paint_top_fraction > 0.0
    if paint_enabled and not (paint_interp_strength > 0.0 and paint_top_fraction > 0.0):
        raise ValueError(
            "PAINT requires both paint_interp_strength and paint_top_fraction to be positive."
        )
    distance_filter_ratio = _sd_cfg(config, "distance_filter_ratio", None)
    distance_filter_high = _sd_bool_cfg(config, "distance_filter_high", True)
    soft_or_filter_enabled = _sd_bool_cfg(config, "soft_or_filter_enabled", False)
    soft_or_filter_ratio = _sd_cfg(config, "soft_or_filter_ratio", None)
    multi_teacher_aggregation = str(
        _sd_cfg(config, "multi_teacher_aggregation", "teacher_mean")
    ).strip().lower()
    valid_multi_teacher_aggregations = {
        "teacher_mean", "token_max_distance", "entropy_grouped_distance"
    }
    if multi_teacher_aggregation not in valid_multi_teacher_aggregations:
        raise ValueError(
            "policy_loss.multi_teacher_aggregation must be one of "
            f"{sorted(valid_multi_teacher_aggregations)}, got {multi_teacher_aggregation!r}."
        )
    if soft_or_filter_enabled:
        if soft_or_filter_ratio is None:
            raise ValueError(
                "policy_loss.soft_or_filter_ratio must be set when soft_or_filter_enabled=True."
            )
        soft_or_filter_ratio = float(soft_or_filter_ratio)
        if (
            not math.isfinite(soft_or_filter_ratio)
            or soft_or_filter_ratio < 0
            or soft_or_filter_ratio >= 1
        ):
            raise ValueError(
                "policy_loss.soft_or_filter_ratio must be in [0, 1), "
                f"got {soft_or_filter_ratio}."
            )
    if not soft_or_filter_enabled and distance_filter_ratio is not None:
        distance_filter_ratio = float(distance_filter_ratio)
        if (
            not math.isfinite(distance_filter_ratio)
            or distance_filter_ratio < 0
            or distance_filter_ratio >= 1
        ):
            raise ValueError(
                "policy_loss.distance_filter_ratio must be in [0, 1), "
                f"got {distance_filter_ratio}."
            )
    entropy_train_mask = kwargs.get("entropy_train_mask")
    if entropy_train_mask is not None:
        if entropy_train_mask.shape != response_mask.shape:
            raise ValueError(
                "entropy_train_mask must match response_mask shape, got "
                f"{tuple(entropy_train_mask.shape)} vs {tuple(response_mask.shape)}."
            )
        entropy_train_mask = entropy_train_mask.to(device=response_mask.device, dtype=torch.bool)
    entropy_router_enabled = multi_teacher_aggregation == "entropy_grouped_distance"
    entropy_router_high_ratio = float(_sd_cfg(config, "entropy_router_high_ratio", 0.2))
    entropy_router_fixed_teacher_type = _sd_bool_cfg(
        config, "entropy_router_fixed_teacher_type", True
    )
    entropy_router_vote_mode = _sd_cfg(config, "entropy_router_vote_mode", None)
    if entropy_router_vote_mode is not None:
        entropy_router_vote_mode = str(entropy_router_vote_mode).strip().lower()
        if entropy_router_vote_mode in {"", "none", "null"}:
            entropy_router_vote_mode = None
    valid_entropy_router_vote_modes = {None, "majority", "minority_high"}
    if entropy_router_vote_mode not in valid_entropy_router_vote_modes:
        raise ValueError(
            "policy_loss.entropy_router_vote_mode must be one of "
            "[null, 'majority', 'minority_high'], got "
            f"{entropy_router_vote_mode!r}."
        )
    if entropy_router_vote_mode is not None and not entropy_router_enabled:
        raise ValueError(
            "policy_loss.entropy_router_vote_mode requires "
            "multi_teacher_aggregation='entropy_grouped_distance'."
        )
    entropy_router_teacher_selection = str(
        _sd_cfg(config, "entropy_router_teacher_selection", "entropy_grouped")
    ).strip().lower()
    valid_entropy_router_teacher_selections = {
        "entropy_grouped", "nearest", "farthest", "random", "mean"
    }
    if entropy_router_teacher_selection not in valid_entropy_router_teacher_selections:
        raise ValueError(
            "policy_loss.entropy_router_teacher_selection must be one of "
            f"{sorted(valid_entropy_router_teacher_selections)}, got "
            f"{entropy_router_teacher_selection!r}."
        )
    if entropy_router_teacher_selection != "entropy_grouped" and (
        not entropy_router_enabled
        or entropy_router_vote_mode != "majority"
        or entropy_router_fixed_teacher_type
    ):
        raise ValueError(
            "Non-default entropy_router_teacher_selection requires "
            "multi_teacher_aggregation='entropy_grouped_distance', "
            "entropy_router_vote_mode='majority', and "
            "entropy_router_fixed_teacher_type=False."
        )
    if entropy_router_teacher_selection == "mean" and (
        distance_filter_ratio is not None or soft_or_filter_enabled
    ):
        raise ValueError(
            "entropy_router_teacher_selection='mean' does not support "
            "distance_filter_ratio or Soft-OR filtering."
        )
    if entropy_router_enabled and (
        not math.isfinite(entropy_router_high_ratio)
        or entropy_router_high_ratio < 0
        or entropy_router_high_ratio >= 1
    ):
        raise ValueError(
            "policy_loss.entropy_router_high_ratio must be in [0, 1), "
            f"got {entropy_router_high_ratio}."
        )
    student_entropy = kwargs.get("student_entropy")
    if soft_or_filter_enabled or entropy_router_enabled or paint_enabled:
        if student_entropy is None:
            if paint_enabled:
                feature = "interpolation"
            else:
                feature = "entropy-grouped routing" if entropy_router_enabled else "Soft-OR filtering"
            raise ValueError(f"OPSD {feature} requires student_entropy.")
        if student_entropy.shape != response_mask.shape:
            raise ValueError(
                "student_entropy must match response_mask shape, got "
                f"{tuple(student_entropy.shape)} vs {tuple(response_mask.shape)}."
            )
        student_entropy = student_entropy.to(device=response_mask.device).detach()
    student_multi_topk = kwargs.get("student_multi_topk_log_probs")
    teacher_multi_topk = kwargs.get("teacher_multi_topk_log_probs")
    teacher_multi_valid_mask = kwargs.get("teacher_multi_valid_mask")
    teacher_multi_log_probs = kwargs.get("teacher_multi_log_probs")
    teacher_multi_entropy = kwargs.get("teacher_multi_entropy")
    teacher_entropy = kwargs.get("teacher_entropy")
    multi_teacher = teacher_multi_topk is not None
    if paint_enabled and (multi_teacher or not full_logit_distill):
        raise ValueError("PAINT interpolation requires single-teacher full_logit_distill OPSD.")

    per_teacher_losses = []
    per_teacher_distance_metrics = []
    raw_per_token_distance = None
    if multi_teacher:
        assert full_logit_distill and top_k_distill > 0, (
            "Multi-teacher OPSD requires full_logit_distill=True and top_k_distill>0."
        )
        assert student_multi_topk is not None and teacher_multi_valid_mask is not None, (
            "Multi-teacher OPSD requires student/teacher top-k tensors and a valid mask."
        )
        assert student_multi_topk.shape == teacher_multi_topk.shape, (
            f"Multi-teacher top-k shape mismatch: student={tuple(student_multi_topk.shape)}, "
            f"teacher={tuple(teacher_multi_topk.shape)}."
        )
        per_token_loss = None
    elif full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="teacher")
        paint_selected_mask = None
        paint_entropy_ratio = None
        if paint_enabled:
            teacher_distill, paint_selected_mask, paint_entropy_ratio = _sd_paint_interpolate_teacher(
                student_log_probs=student_distill,
                teacher_log_probs=teacher_distill,
                student_entropy=student_entropy,
                teacher_entropy=teacher_entropy,
                response_mask=response_mask,
                strength=paint_interp_strength,
                top_fraction=paint_top_fraction,
            )
        # Match official OPSD: clip per-(token, vocab) KL contribution before
        # summing over vocab. The clip can break Gibbs' inequality and produce
        # negative per-token loss values; this matches the reference behavior.
        per_token_loss = _sd_generalized_jsd(
            student_distill,
            teacher_distill,
            beta,
            vocab_clip=jsd_token_clip if jsd_token_clip > 0 else None)
        if (distance_filter_ratio is not None or soft_or_filter_enabled) and jsd_token_clip > 0:
            with torch.no_grad():
                raw_per_token_distance = _sd_generalized_jsd(
                    student_distill.detach(), teacher_distill.detach(), beta, vocab_clip=None
                )
        else:
            raw_per_token_distance = per_token_loss
    else:
        raw_per_token_distance = (teacher_lp.detach() - log_prob).pow(2)
        per_token_loss = raw_per_token_distance
        if jsd_token_clip > 0:
            per_token_loss = per_token_loss.clamp(max=jsd_token_clip)

    if rollout_is_weights is not None and per_token_loss is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    opsd_rollout_mask = kwargs.get("opsd_rollout_mask")
    selected_response_mask = response_mask
    if opsd_rollout_mask is not None:
        if opsd_rollout_mask.ndim == 1:
            opsd_rollout_mask = opsd_rollout_mask.unsqueeze(-1)
        if opsd_rollout_mask.ndim != 2 or opsd_rollout_mask.shape[0] != response_mask.shape[0]:
            raise ValueError(
                "opsd_rollout_mask must have shape [B] or [B, 1], but got "
                f"{tuple(opsd_rollout_mask.shape)} for response_mask={tuple(response_mask.shape)}."
            )
        opsd_rollout_mask = opsd_rollout_mask.to(device=response_mask.device, dtype=response_mask.dtype)
        selected_response_mask = response_mask * opsd_rollout_mask
        # if os.environ.get("VERL_DEBUG_BATCH", "1").lower() not in {"0", "false", "no"}:
        #     # _debug_print_reward_tensor(selected_response_mask)
        #     print(f'opsd_rollout_mask: {opsd_rollout_mask.detach().float().cpu()}')

    def _teacher_loss_mask(raw_distance, teacher_base_mask):
        """Combine common entropy selection with this teacher's distance selection."""
        base_bool = teacher_base_mask.bool()
        distance_train_mask = None
        distance_threshold = 0.0
        distance_metrics = {}
        if soft_or_filter_enabled:
            soft_or_train_mask, soft_or_threshold, soft_or_score = _sd_soft_or_train_mask(
                student_entropy, raw_distance, base_bool, soft_or_filter_ratio
            )
            final_bool = base_bool & soft_or_train_mask
            selected_count = final_bool.sum()
            rejected_mask = base_bool & ~soft_or_train_mask
            rejected_count = rejected_mask.sum()
            base_count = base_bool.sum().clamp(min=1)
            distance_metrics = {
                "soft_or_selected_ratio": (selected_count.float() / base_count).item(),
                "soft_or_rejected_ratio": (rejected_count.float() / base_count).item(),
                "soft_or_effective_threshold": float(soft_or_threshold),
                "selected_soft_or_score": (
                    soft_or_score.masked_select(final_bool).mean().item()
                    if selected_count.item() > 0 else 0.0
                ),
                "rejected_soft_or_score": (
                    soft_or_score.masked_select(rejected_mask).mean().item()
                    if rejected_count.item() > 0 else 0.0
                ),
                "selected_student_entropy": (
                    student_entropy.float().masked_select(final_bool).mean().item()
                    if selected_count.item() > 0 else 0.0
                ),
                "rejected_student_entropy": (
                    student_entropy.float().masked_select(rejected_mask).mean().item()
                    if rejected_count.item() > 0 else 0.0
                ),
                "selected_distance": (
                    raw_distance.detach().float().masked_select(final_bool).mean().item()
                    if selected_count.item() > 0 else 0.0
                ),
                "rejected_distance": (
                    raw_distance.detach().float().masked_select(rejected_mask).mean().item()
                    if rejected_count.item() > 0 else 0.0
                ),
            }
            return final_bool.to(dtype=teacher_base_mask.dtype), distance_metrics

        if distance_filter_ratio is not None:
            distance_train_mask, distance_threshold = _sd_exclusion_ratio_train_mask(
                raw_distance, base_bool, distance_filter_ratio, distance_filter_high
            )

        if entropy_train_mask is not None and distance_train_mask is not None:
            rule_train_mask = entropy_train_mask | distance_train_mask
        elif entropy_train_mask is not None:
            rule_train_mask = entropy_train_mask
        elif distance_train_mask is not None:
            rule_train_mask = distance_train_mask
        else:
            rule_train_mask = base_bool
        final_bool = base_bool & rule_train_mask

        if distance_train_mask is not None:
            distance_selected = base_bool & distance_train_mask
            distance_rejected = base_bool & ~distance_train_mask
            base_count = base_bool.sum().clamp(min=1)
            selected_count = distance_selected.sum()
            rejected_count = distance_rejected.sum()
            detached_distance = raw_distance.detach().float()
            distance_metrics = {
                "distance_selected_ratio": (selected_count.float() / base_count).item(),
                "distance_rejected_ratio": (rejected_count.float() / base_count).item(),
                "selected_distance": (
                    detached_distance.masked_select(distance_selected).mean().item()
                    if selected_count.item() > 0 else 0.0
                ),
                "rejected_distance": (
                    detached_distance.masked_select(distance_rejected).mean().item()
                    if rejected_count.item() > 0 else 0.0
                ),
                "distance_effective_threshold": float(distance_threshold),
                "union_selected_ratio": (final_bool.sum().float() / base_count).item(),
            }
        elif entropy_train_mask is not None:
            distance_metrics["union_selected_ratio"] = (
                final_bool.sum().float() / base_bool.sum().clamp(min=1)
            ).item()
        return final_bool.to(dtype=teacher_base_mask.dtype), distance_metrics

    # The denominator is clamped in _sd_token_mean, so an all-masked
    # micro-batch gives a differentiable zero loss instead of NaN.
    routing_metrics = {}
    num_teachers = teacher_multi_topk.shape[1] if multi_teacher else 1
    if multi_teacher and multi_teacher_aggregation == "teacher_mean":
        for teacher_idx in range(num_teachers):
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_multi_topk[:, teacher_idx],
                teacher_topk_log_probs=teacher_multi_topk[:, teacher_idx],
                add_tail=distill_add_tail)
            teacher_per_token_loss = _sd_generalized_jsd(
                student_distill,
                teacher_distill,
                beta,
                vocab_clip=jsd_token_clip if jsd_token_clip > 0 else None)
            if (distance_filter_ratio is not None or soft_or_filter_enabled) and jsd_token_clip > 0:
                with torch.no_grad():
                    raw_teacher_distance = _sd_generalized_jsd(
                        student_distill.detach(), teacher_distill.detach(), beta, vocab_clip=None
                    )
            else:
                raw_teacher_distance = teacher_per_token_loss
            if rollout_is_weights is not None:
                teacher_per_token_loss = teacher_per_token_loss * rollout_is_weights
            teacher_valid = teacher_multi_valid_mask[:, teacher_idx].to(
                device=selected_response_mask.device, dtype=selected_response_mask.dtype
            ).unsqueeze(-1)
            teacher_base_mask = selected_response_mask * teacher_valid
            teacher_mask, teacher_distance_metrics = _teacher_loss_mask(
                raw_teacher_distance, teacher_base_mask
            )
            per_teacher_losses.append(_sd_token_mean(teacher_per_token_loss, teacher_mask))
            per_teacher_distance_metrics.append(teacher_distance_metrics)
        loss = torch.stack(per_teacher_losses).mean()
    elif multi_teacher:
        all_teacher_losses = []
        all_teacher_distances = []
        # Select teachers with the unclipped, detached divergence, while the
        # gathered (possibly clipped) loss remains differentiable w.r.t. the student.
        for teacher_idx in range(num_teachers):
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_multi_topk[:, teacher_idx],
                teacher_topk_log_probs=teacher_multi_topk[:, teacher_idx],
                add_tail=distill_add_tail)
            teacher_per_token_loss = _sd_generalized_jsd(
                student_distill,
                teacher_distill,
                beta,
                vocab_clip=jsd_token_clip if jsd_token_clip > 0 else None)
            # Teacher routing may use a different KL direction from the
            # differentiable distillation objective. Both operate on the same
            # teacher top-k support (plus the optional tail bucket).
            with torch.no_grad():
                raw_teacher_distance = _sd_generalized_jsd(
                    student_distill.detach(),
                    teacher_distill.detach(),
                    teacher_selection_beta,
                    vocab_clip=None,
                )
            all_teacher_losses.append(teacher_per_token_loss)
            all_teacher_distances.append(raw_teacher_distance)

        teacher_losses = torch.stack(all_teacher_losses, dim=1)
        teacher_distances = torch.stack(all_teacher_distances, dim=1)
        teacher_valid = teacher_multi_valid_mask.to(
            device=selected_response_mask.device, dtype=torch.bool
        ).unsqueeze(-1)
        teacher_eligible = teacher_valid & selected_response_mask.bool().unsqueeze(1)
        detached_distances = teacher_distances.detach().float()
        if entropy_router_enabled:
            if entropy_router_vote_mode is not None and entropy_router_fixed_teacher_type:
                raise ValueError(
                    "entropy_router_vote_mode operates over the full teacher pool and requires "
                    "entropy_router_fixed_teacher_type=False."
                )
            if entropy_router_fixed_teacher_type and num_teachers != 4:
                raise ValueError(
                    "entropy_grouped_distance with fixed teacher types requires exactly "
                    "four teachers ordered as "
                    "prefix_0.5, prefix_0.7, suffix_0.5, suffix_0.7; "
                    f"got {num_teachers}."
                )
            if not entropy_router_fixed_teacher_type and num_teachers < 2:
                raise ValueError(
                    "entropy_grouped_distance over the full teacher pool requires at least "
                    f"two teachers, got {num_teachers}."
                )
            valid_tokens = selected_response_mask.bool()
            valid_flat_indices = valid_tokens.flatten().nonzero(as_tuple=False).squeeze(-1)
            num_high = math.ceil(valid_flat_indices.numel() * entropy_router_high_ratio)
            high_entropy_mask = torch.zeros_like(valid_tokens)
            if num_high > 0:
                valid_entropies = student_entropy.detach().float().flatten()[valid_flat_indices]
                high_local_indices = torch.topk(
                    valid_entropies, k=num_high, largest=True, sorted=False
                ).indices
                high_entropy_mask.flatten()[valid_flat_indices[high_local_indices]] = True
            rest_entropy_mask = valid_tokens & ~high_entropy_mask

            prefix_eligible = teacher_eligible[:, :2]
            suffix_eligible = teacher_eligible[:, 2:]
            if entropy_router_fixed_teacher_type:
                high_candidate_distances = detached_distances[:, :2].masked_fill(
                    ~prefix_eligible, torch.inf
                )
                rest_candidate_distances = detached_distances[:, 2:].masked_fill(
                    ~suffix_eligible, -torch.inf
                )
                high_winner = high_candidate_distances.argmin(dim=1)
                rest_winner = rest_candidate_distances.argmax(dim=1) + 2
                has_high_teacher = prefix_eligible.any(dim=1)
                has_rest_teacher = suffix_eligible.any(dim=1)
                high_candidate_count = prefix_eligible.sum(dim=1)
                rest_candidate_count = suffix_eligible.sum(dim=1)
            else:
                high_candidates = teacher_eligible
                rest_candidates = teacher_eligible
                if entropy_router_vote_mode is not None:
                    if teacher_multi_log_probs is None:
                        raise ValueError(
                            "entropy_router_vote_mode requires teacher_multi_log_probs."
                        )
                    if teacher_multi_log_probs.shape != detached_distances.shape:
                        raise ValueError(
                            "teacher_multi_log_probs must match teacher distances for voting, got "
                            f"{tuple(teacher_multi_log_probs.shape)} vs "
                            f"{tuple(detached_distances.shape)}."
                        )
                    signed_delta = (
                        teacher_multi_log_probs.to(device=log_prob.device).detach().float()
                        - log_prob.detach().float().unsqueeze(1)
                    )
                    positive = teacher_eligible & (signed_delta > 0)
                    negative = teacher_eligible & (signed_delta < 0)
                    positive_count = positive.sum(dim=1)
                    negative_count = negative.sum(dim=1)
                    positive_magnitude = signed_delta.masked_fill(~positive, 0.0).sum(dim=1).abs()
                    negative_magnitude = signed_delta.masked_fill(~negative, 0.0).sum(dim=1).abs()
                    count_tie = positive_count == negative_count
                    majority_is_positive = (positive_count > negative_count) | (
                        count_tie & (positive_magnitude >= negative_magnitude)
                    )
                    majority_candidates = torch.where(
                        majority_is_positive.unsqueeze(1), positive, negative
                    )
                    minority_candidates = torch.where(
                        majority_is_positive.unsqueeze(1), negative, positive
                    )
                    majority_empty = ~majority_candidates.any(dim=1)
                    majority_candidates = torch.where(
                        majority_empty.unsqueeze(1), teacher_eligible, majority_candidates
                    )
                    rest_candidates = majority_candidates
                    if entropy_router_vote_mode == "majority":
                        high_candidates = majority_candidates
                        high_minority_empty = torch.zeros_like(majority_empty)
                        high_minority_singleton = torch.zeros_like(majority_empty)
                    else:
                        minority_count = minority_candidates.sum(dim=1)
                        high_minority_empty = minority_count == 0
                        high_minority_singleton = minority_count == 1
                        high_candidates = torch.where(
                            high_minority_empty.unsqueeze(1), teacher_eligible, minority_candidates
                        )
                    routing_metrics.update({
                        "distill_routing_vote_positive_teacher_count_mean": (
                            positive_count.float().masked_select(valid_tokens).mean().item()
                            if valid_tokens.any() else 0.0
                        ),
                        "distill_routing_vote_negative_teacher_count_mean": (
                            negative_count.float().masked_select(valid_tokens).mean().item()
                            if valid_tokens.any() else 0.0
                        ),
                        "distill_routing_vote_count_tie_token_count": float(
                            (count_tie & valid_tokens).sum().item()
                        ),
                        "distill_routing_vote_magnitude_tie_token_count": float(
                            (
                                count_tie
                                & (positive_magnitude == negative_magnitude)
                                & valid_tokens
                            ).sum().item()
                        ),
                        "distill_routing_vote_majority_empty_fallback_count": float(
                            (majority_empty & valid_tokens).sum().item()
                        ),
                        "distill_routing_high_minority_empty_fallback_count": float(
                            (high_minority_empty & high_entropy_mask).sum().item()
                        ),
                        "distill_routing_high_minority_singleton_count": float(
                            (high_minority_singleton & high_entropy_mask).sum().item()
                        ),
                    })
                high_candidate_distances = detached_distances.masked_fill(
                    ~high_candidates, torch.inf
                )
                rest_candidate_distances = detached_distances.masked_fill(
                    ~rest_candidates, -torch.inf
                )
                if entropy_router_teacher_selection == "nearest":
                    high_winner = high_candidate_distances.argmin(dim=1)
                    rest_winner = detached_distances.masked_fill(
                        ~rest_candidates, torch.inf
                    ).argmin(dim=1)
                elif entropy_router_teacher_selection == "random":
                    random_scores = torch.rand_like(detached_distances)
                    high_winner = random_scores.masked_fill(
                        ~high_candidates, -torch.inf
                    ).argmax(dim=1)
                    rest_winner = random_scores.masked_fill(
                        ~rest_candidates, -torch.inf
                    ).argmax(dim=1)
                else:
                    if entropy_router_teacher_selection == "farthest":
                        high_winner = detached_distances.masked_fill(
                            ~high_candidates, -torch.inf
                        ).argmax(dim=1)
                    else:
                        high_winner = high_candidate_distances.argmin(dim=1)
                    rest_winner = rest_candidate_distances.argmax(dim=1)
                has_high_teacher = high_candidates.any(dim=1)
                has_rest_teacher = rest_candidates.any(dim=1)
                high_candidate_count = high_candidates.sum(dim=1)
                rest_candidate_count = rest_candidates.sum(dim=1)
            has_valid_teacher = torch.where(
                high_entropy_mask, has_high_teacher, has_rest_teacher
            )
            missing_group_teacher = valid_tokens & ~has_valid_teacher
            if missing_group_teacher.any():
                raise ValueError(
                    "entropy_grouped_distance found response tokens without a valid teacher "
                    "in their required candidate pool."
                )
            winner_teacher_idx = torch.where(
                high_entropy_mask, high_winner, rest_winner
            )
            selected_teacher_candidates = torch.where(
                high_entropy_mask.unsqueeze(1), high_candidates, rest_candidates
            ) if not entropy_router_fixed_teacher_type else None
            selection_distances = detached_distances.masked_fill(~teacher_eligible, -torch.inf)
        else:
            selection_distances = detached_distances.masked_fill(~teacher_eligible, -torch.inf)
            has_valid_teacher = teacher_eligible.any(dim=1)
            winner_teacher_idx = selection_distances.argmax(dim=1)
        gather_index = winner_teacher_idx.unsqueeze(1)
        chosen_loss_unweighted = torch.gather(teacher_losses, 1, gather_index).squeeze(1)
        chosen_distance = torch.gather(teacher_distances, 1, gather_index).squeeze(1)
        if entropy_router_enabled and entropy_router_teacher_selection == "mean":
            candidate_weights = selected_teacher_candidates.to(dtype=teacher_losses.dtype)
            candidate_count = candidate_weights.sum(dim=1).clamp(min=1)
            chosen_loss_unweighted = (teacher_losses * candidate_weights).sum(dim=1) / candidate_count
            chosen_distance = (
                teacher_distances * candidate_weights.to(dtype=teacher_distances.dtype)
            ).sum(dim=1) / candidate_count.to(dtype=teacher_distances.dtype)
            routing_metrics["distill_routing_majority_mean_enabled"] = 1.0
            routing_metrics["distill_routing_majority_candidate_count_mean"] = (
                candidate_count.float().masked_select(selected_response_mask.bool()).mean().item()
                if selected_response_mask.bool().any() else 0.0
            )
            for teacher_idx in range(num_teachers):
                teacher_candidate_count = (
                    selected_teacher_candidates[:, teacher_idx]
                    & selected_response_mask.bool()
                ).sum()
                routing_metrics[
                    f"distill_routing_teacher_{teacher_idx}_majority_candidate_count"
                ] = float(teacher_candidate_count.item())
        chosen_loss = chosen_loss_unweighted
        if rollout_is_weights is not None:
            chosen_loss = chosen_loss * rollout_is_weights

        composite_base_mask = selected_response_mask * has_valid_teacher.to(
            dtype=selected_response_mask.dtype
        )
        final_response_mask, filter_metrics = _teacher_loss_mask(
            chosen_distance, composite_base_mask
        )
        loss = _sd_token_mean(chosen_loss, final_response_mask)

        base_bool = composite_base_mask.bool()
        final_bool = final_response_mask.bool()
        eligible_count = base_bool.sum()
        final_count = final_bool.sum()
        routing_metrics.update({
            "distill_routing_eligible_token_count": float(eligible_count.item()),
            "distill_routing_final_token_count": float(final_count.item()),
            "distill_routing_final_token_ratio": (
                final_count.float() / eligible_count.clamp(min=1)
            ).item(),
        })
        if entropy_router_enabled:
            high_base = base_bool & high_entropy_mask
            rest_base = base_bool & rest_entropy_mask
            high_count = high_base.sum()
            rest_count = rest_base.sum()
            nearest_two = torch.topk(
                high_candidate_distances, k=2, dim=1, largest=False
            ).values
            farthest_two = torch.topk(
                rest_candidate_distances, k=2, dim=1, largest=True
            ).values
            high_nearest_gap = nearest_two[:, 1] - nearest_two[:, 0]
            rest_farthest_gap = farthest_two[:, 0] - farthest_two[:, 1]
            valid_high_gap = high_nearest_gap.masked_select(
                high_base & (high_candidate_count >= 2)
            )
            valid_rest_gap = rest_farthest_gap.masked_select(
                rest_base & (rest_candidate_count >= 2)
            )
            routing_metrics.update({
                "distill_routing_fixed_teacher_type_enabled": float(
                    entropy_router_fixed_teacher_type
                ),
                "distill_routing_vote_mode_enabled": float(
                    entropy_router_vote_mode is not None
                ),
                "distill_routing_high_entropy_token_count": float(high_count.item()),
                "distill_routing_rest_entropy_token_count": float(rest_count.item()),
                "distill_routing_high_entropy_ratio": (
                    high_count.float() / eligible_count.clamp(min=1)
                ).item(),
                "distill_routing_high_nearest_distance_gap_sum": (
                    valid_high_gap.sum().item() if valid_high_gap.numel() else 0.0
                ),
                "distill_routing_high_nearest_distance_gap_token_count": float(
                    valid_high_gap.numel()
                ),
                "distill_routing_rest_farthest_distance_gap_sum": (
                    valid_rest_gap.sum().item() if valid_rest_gap.numel() else 0.0
                ),
                "distill_routing_rest_farthest_distance_gap_token_count": float(
                    valid_rest_gap.numel()
                ),
            })
            for teacher_idx in range(num_teachers):
                routing_metrics.update({
                    f"distill_routing_high_teacher_{teacher_idx}_count": float(
                        (high_base & (winner_teacher_idx == teacher_idx)).sum().item()
                    ),
                    f"distill_routing_rest_teacher_{teacher_idx}_count": float(
                        (rest_base & (winner_teacher_idx == teacher_idx)).sum().item()
                    ),
                })
            if valid_high_gap.numel():
                high_gap_quantiles = torch.quantile(
                    valid_high_gap, torch.tensor([0.5, 0.95], device=valid_high_gap.device)
                )
                routing_metrics.update({
                    "distill_routing_high_nearest_distance_gap_mean": valid_high_gap.mean().item(),
                    "distill_routing_high_nearest_distance_gap_p50": high_gap_quantiles[0].item(),
                    "distill_routing_high_nearest_distance_gap_p95": high_gap_quantiles[1].item(),
                })
            if valid_rest_gap.numel():
                rest_gap_quantiles = torch.quantile(
                    valid_rest_gap, torch.tensor([0.5, 0.95], device=valid_rest_gap.device)
                )
                routing_metrics.update({
                    "distill_routing_rest_farthest_distance_gap_mean": valid_rest_gap.mean().item(),
                    "distill_routing_rest_farthest_distance_gap_p50": rest_gap_quantiles[0].item(),
                    "distill_routing_rest_farthest_distance_gap_p95": rest_gap_quantiles[1].item(),
                })
            if entropy_router_fixed_teacher_type:
                routing_metrics.update({
                    "distill_routing_high_prefix_0_5_count": float(
                        (high_base & (winner_teacher_idx == 0)).sum().item()
                    ),
                    "distill_routing_high_prefix_0_7_count": float(
                        (high_base & (winner_teacher_idx == 1)).sum().item()
                    ),
                    "distill_routing_rest_suffix_0_5_count": float(
                        (rest_base & (winner_teacher_idx == 2)).sum().item()
                    ),
                    "distill_routing_rest_suffix_0_7_count": float(
                        (rest_base & (winner_teacher_idx == 3)).sum().item()
                    ),
                    "distill_routing_high_prefix_distance_gap_sum": (
                        valid_high_gap.sum().item() if valid_high_gap.numel() else 0.0
                    ),
                    "distill_routing_high_prefix_distance_gap_token_count": float(
                        valid_high_gap.numel()
                    ),
                    "distill_routing_rest_suffix_distance_gap_sum": (
                        valid_rest_gap.sum().item() if valid_rest_gap.numel() else 0.0
                    ),
                    "distill_routing_rest_suffix_distance_gap_token_count": float(
                        valid_rest_gap.numel()
                    ),
                })
                if valid_high_gap.numel():
                    routing_metrics.update({
                        "distill_routing_high_prefix_distance_gap_mean": valid_high_gap.mean().item(),
                        "distill_routing_high_prefix_distance_gap_p50": high_gap_quantiles[0].item(),
                        "distill_routing_high_prefix_distance_gap_p95": high_gap_quantiles[1].item(),
                    })
                if valid_rest_gap.numel():
                    routing_metrics.update({
                        "distill_routing_rest_suffix_distance_gap_mean": valid_rest_gap.mean().item(),
                        "distill_routing_rest_suffix_distance_gap_p50": rest_gap_quantiles[0].item(),
                        "distill_routing_rest_suffix_distance_gap_p95": rest_gap_quantiles[1].item(),
                    })
        for metric_name, metric_value in filter_metrics.items():
            routing_metrics[f"distill_routing_{metric_name}"] = metric_value

        teacher_shares = []
        for teacher_idx in range(num_teachers):
            assigned = base_bool & (winner_teacher_idx == teacher_idx)
            retained = final_bool & (winner_teacher_idx == teacher_idx)
            assigned_count = assigned.sum()
            retained_count = retained.sum()
            teacher_shares.append(assigned_count.float() / eligible_count.clamp(min=1))
            prefix = f"distill_routing_teacher_{teacher_idx}"
            routing_metrics.update({
                f"{prefix}_assigned_token_count": float(assigned_count.item()),
                f"{prefix}_final_token_count": float(retained_count.item()),
                f"{prefix}_pre_filter_ratio": (
                    assigned_count.float() / eligible_count.clamp(min=1)
                ).item(),
                f"{prefix}_post_filter_share": (
                    retained_count.float() / final_count.clamp(min=1)
                ).item(),
                f"{prefix}_retention_ratio": (
                    retained_count.float() / assigned_count.clamp(min=1)
                ).item(),
                f"{prefix}_winner_distance_mean": (
                    chosen_distance.detach().float().masked_select(assigned).mean().item()
                    if assigned_count.item() > 0 else float("nan")
                ),
                f"{prefix}_winner_loss_mean": (
                    chosen_loss_unweighted.detach().float().masked_select(assigned).mean().item()
                    if assigned_count.item() > 0 else float("nan")
                ),
            })

        share_tensor = torch.stack(teacher_shares)
        positive_shares = share_tensor > 0
        selection_entropy = -(
            share_tensor.masked_select(positive_shares)
            * share_tensor.masked_select(positive_shares).log()
        ).sum()
        routing_metrics.update({
            "distill_routing_dominant_teacher_share": share_tensor.max().item(),
            "distill_routing_teacher_selection_entropy": selection_entropy.item(),
            "distill_routing_teacher_selection_entropy_normalized": (
                selection_entropy / math.log(num_teachers) if num_teachers > 1 else selection_entropy * 0
            ).item(),
            "distill_routing_active_teacher_count": float(positive_shares.sum().item()),
        })
        if teacher_multi_log_probs is not None:
            winner_teacher_log_probs = torch.gather(
                teacher_multi_log_probs.to(device=gather_index.device), 1, gather_index
            ).squeeze(1)
            routing_metrics["distill_routing_winner_teacher_avg_logprob"] = (
                winner_teacher_log_probs.detach().float().masked_select(base_bool).mean().item()
                if eligible_count.item() > 0 else float("nan")
            )
        if teacher_multi_entropy is not None:
            winner_teacher_entropy = torch.gather(
                teacher_multi_entropy.to(device=gather_index.device), 1, gather_index
            ).squeeze(1)
            routing_metrics["distill_routing_winner_teacher_entropy"] = (
                winner_teacher_entropy.detach().float().masked_select(base_bool).mean().item()
                if eligible_count.item() > 0 else float("nan")
            )

        valid_winner_distances = chosen_distance.detach().float().masked_select(base_bool)
        if valid_winner_distances.numel() > 0:
            winner_quantiles = torch.quantile(
                valid_winner_distances,
                torch.tensor([0.5, 0.95], device=valid_winner_distances.device),
            )
            routing_metrics.update({
                "distill_routing_winner_distance_mean": valid_winner_distances.mean().item(),
                "distill_routing_winner_distance_p50": winner_quantiles[0].item(),
                "distill_routing_winner_distance_p95": winner_quantiles[1].item(),
                "distill_routing_winner_distance_max": valid_winner_distances.max().item(),
            })

        valid_teacher_count = teacher_eligible.sum(dim=1)
        if num_teachers >= 2 and not entropy_router_enabled:
            top2_distances = torch.topk(selection_distances, k=2, dim=1).values
            margin_mask = base_bool & (valid_teacher_count >= 2)
            margins = (top2_distances[:, 0] - top2_distances[:, 1]).masked_select(margin_mask)
            if margins.numel() > 0:
                top1 = top2_distances[:, 0].masked_select(margin_mask)
                margin_quantiles = torch.quantile(
                    margins,
                    torch.tensor([0.5, 0.95], device=margins.device),
                )
                routing_metrics.update({
                    "distill_routing_top2_margin_mean": margins.mean().item(),
                    "distill_routing_top2_margin_p50": margin_quantiles[0].item(),
                    "distill_routing_top2_margin_p95": margin_quantiles[1].item(),
                    "distill_routing_relative_margin_mean": (
                        margins / top1.abs().clamp(min=1e-12)
                    ).mean().item(),
                    "distill_routing_near_tie_token_count": float((margins <= 1e-6).sum().item()),
                    "distill_routing_margin_token_count": float(margins.numel()),
                    "distill_routing_near_tie_ratio": (margins <= 1e-6).float().mean().item(),
                })

        adjacent_base = base_bool[:, :-1] & base_bool[:, 1:]
        adjacent_switch = adjacent_base & (
            winner_teacher_idx[:, :-1] != winner_teacher_idx[:, 1:]
        )
        adjacent_final = final_bool[:, :-1] & final_bool[:, 1:]
        adjacent_final_switch = adjacent_final & (
            winner_teacher_idx[:, :-1] != winner_teacher_idx[:, 1:]
        )
        routing_metrics.update({
            "distill_routing_adjacent_pair_count": float(adjacent_base.sum().item()),
            "distill_routing_adjacent_switch_count": float(adjacent_switch.sum().item()),
            "distill_routing_adjacent_switch_rate": (
                adjacent_switch.sum().float() / adjacent_base.sum().clamp(min=1)
            ).item(),
            "distill_routing_final_adjacent_pair_count": float(adjacent_final.sum().item()),
            "distill_routing_final_adjacent_switch_count": float(adjacent_final_switch.sum().item()),
            "distill_routing_final_adjacent_switch_rate": (
                adjacent_final_switch.sum().float() / adjacent_final.sum().clamp(min=1)
            ).item(),
        })
    else:
        final_response_mask, teacher_distance_metrics = _teacher_loss_mask(
            raw_per_token_distance, selected_response_mask
        )
        per_teacher_distance_metrics.append(teacher_distance_metrics)
        loss = _sd_token_mean(per_token_loss, final_response_mask)
    selected_rollout_ratio = (
        opsd_rollout_mask.float().mean().item() if opsd_rollout_mask is not None else 1.0
    )
    metrics = {
        "distill_loss": loss.item(),
        "distill_beta": beta,
        "distill_teacher_selection_beta": teacher_selection_beta,
        "distill_topk": float(top_k_distill),
        "distill_full_logit": float(full_logit_distill),
        "distill_num_teachers": float(num_teachers),
        "distill_multi_teacher_token_routing_enabled": float(
            multi_teacher and multi_teacher_aggregation in {
                "token_max_distance", "entropy_grouped_distance"
            }
        ),
        "distill_selected_rollout_ratio": selected_rollout_ratio,
        "distill_selected_token_count": selected_response_mask.sum().item()}
    if paint_enabled:
        valid_paint = response_mask.bool()
        selected_count = paint_selected_mask.sum()
        metrics.update({
            "paint_interp_strength": paint_interp_strength,
            "paint_top_fraction": paint_top_fraction,
            "paint_selected_token_count": selected_count.item(),
            "paint_selected_token_ratio": (
                selected_count.float() / valid_paint.sum().clamp(min=1)
            ).item(),
            "paint_entropy_ratio_mean": (
                paint_entropy_ratio.masked_select(valid_paint).mean().item()
                if valid_paint.any() else 0.0
            ),
        })
    metrics.update(routing_metrics)
    for teacher_idx, teacher_loss in enumerate(per_teacher_losses):
        metrics[f"distill_teacher_{teacher_idx}_loss"] = teacher_loss.detach().item()
    for teacher_idx, teacher_metrics in enumerate(per_teacher_distance_metrics):
        for metric_name, metric_value in teacher_metrics.items():
            metrics[f"distill_teacher_{teacher_idx}_{metric_name}"] = metric_value
    if per_teacher_distance_metrics:
        for metric_name in (
            "distance_selected_ratio", "distance_rejected_ratio", "union_selected_ratio",
            "soft_or_selected_ratio", "soft_or_rejected_ratio",
        ):
            values = [m[metric_name] for m in per_teacher_distance_metrics if metric_name in m]
            if values:
                metrics[f"distill_{metric_name}"] = sum(values) / len(values)
    metrics.update(
        _sd_teacher_metrics(
            teacher_lp, log_prob, selected_response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("opsd_ectr")
def compute_policy_loss_opsd_ectr(old_log_prob, log_prob, advantages, response_mask,
                                  loss_agg_mode="token_mean", config=None,
                                  rollout_is_weights=None, **kwargs):
    """OPSD with CFG-style contrastive guided target in the top-k subspace.

    Same sparse top-k machinery as OPSD: top-k vocab indices are chosen by the
    correct-context teacher, and student / correct-teacher / wrong-teacher all
    gather their log-probs at those indices. An optional tail bucket captures
    the remaining mass. In this (top-k + tail) log-space:

        guided_logits = (1 + α) · log p_c  -  α · clamp(log p_w, min=floor)
        q_target      = softmax(guided_logits)

    Student matches q_target via the same generalized-JSD machinery as OPSD.
    α = 0 recovers vanilla OPSD; α > 0 amplifies the correct-vs-wrong contrast
    across the top-k support.

    Token-level contrastive masking (controlled by ``opsd_ectr_cw_tau``):
    At each token position the KL divergence between correct and wrong teachers
    is computed.  Positions where correct ≈ wrong (stylistic tokens) receive
    near-zero loss weight so the student's policy is left unchanged there;
    positions where the teachers genuinely disagree (content tokens) receive
    full loss weight.  tau controls the sharpness of this soft mask —
    setting tau=0 disables masking and recovers the original uniform behaviour.
    """
    teacher_correct_lp = kwargs.get("teacher_log_probs")
    student_topk = kwargs.get("student_topk_log_probs")
    teacher_correct_topk = kwargs.get("teacher_topk_log_probs")
    teacher_wrong_topk = kwargs.get("teacher_wrong_topk_log_probs")
    assert (
        student_topk is not None
        and teacher_correct_topk is not None
        and teacher_wrong_topk is not None
    ), (
        "opsd_ectr requires student/teacher_correct/teacher_wrong topk_log_probs "
        "gathered at teacher_correct's top-k indices."
    )

    alpha = float(_sd_cfg(config, "opsd_ectr_alpha", 0.5))
    log_w_floor = float(_sd_cfg(config, "opsd_ectr_log_w_floor", -30.0))
    cw_tau = float(_sd_cfg(config, "opsd_ectr_cw_tau", 1.0))
    beta = float(_sd_cfg(config, "beta", 0.0))
    jsd_token_clip = float(_sd_cfg(config, "jsd_token_clip", 0.05))
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    student_dist, correct_dist = _sd_prepare_sparse_topk_log_probs(
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_correct_topk,
        add_tail=distill_add_tail)
    _, wrong_dist = _sd_prepare_sparse_topk_log_probs(
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_wrong_topk,
        add_tail=distill_add_tail)

    wrong_clamped = wrong_dist.clamp(min=log_w_floor)
    guided_logits = (1.0 + alpha) * correct_dist - alpha * wrong_clamped
    guided_log_probs = guided_logits - torch.logsumexp(guided_logits, dim=-1, keepdim=True)

    per_token_loss = _sd_generalized_jsd(
        student_dist,
        guided_log_probs,
        beta,
        vocab_clip=jsd_token_clip if jsd_token_clip > 0 else None)

    # Token-level contrastive mask: only train where correct and wrong teachers
    # genuinely disagree; leave the student's policy untouched at stylistic
    # positions where the hint carries no information.
    if cw_tau > 0:
        with torch.no_grad():
            cw_kl = F.kl_div(
                wrong_dist, correct_dist, reduction="none", log_target=True).sum(-1)  # [B, S]
            cw_kl = cw_kl.clamp(min=0.0)
            token_weight = cw_kl / (cw_kl + cw_tau)
        per_token_loss = per_token_loss * token_weight

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)

    metrics = {
        "distill_loss": loss.item(),
        "distill_beta": beta,
        "distill_topk": float(teacher_correct_topk.shape[-1]),
        "distill_full_logit": 1.0,
        "opsd_ectr_alpha": alpha,
        "opsd_ectr_log_w_floor": log_w_floor,
        "opsd_ectr_cw_tau": cw_tau}
    if cw_tau > 0:
        active = _sd_token_mean(token_weight, response_mask).item()
        metrics["opsd_ectr_active_ratio"] = active
    metrics.update(_sd_teacher_metrics(teacher_correct_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("rlsd")
def compute_policy_loss_rlsd(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """RLSD: GRPO + evidence-ratio modulated advantages."""
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "RLSD requires teacher_log_probs but got None. "
        "teacher_mode=snapshot requires EMA/snapshot implementation in engine_workers."
    )
    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    epsilon_w = float(_sd_cfg(config, "epsilon_w", 0.2))
    lam = _sd_effective_lam(config, kwargs.get("global_steps"), default=0.5)

    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()

    with torch.no_grad():
        delta_t = teacher_lp - log_prob.detach()
        w_t = (advantages.sign() * delta_t).exp().clamp(1 - epsilon_w, 1 + epsilon_w)
        mod_adv = advantages * ((1 - lam) + lam * w_t)

    surr1 = ratio * mod_adv
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * mod_adv
    per_token_loss = -torch.min(surr1, surr2)
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {"pg_loss": loss.item(),
               "pg_clipfrac": ((ratio - 1).abs() > epsilon).float().mean().item(),
               "rlsd_lam": lam}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("rlsd_ectr")
def compute_policy_loss_rlsd_ectr(old_log_prob, log_prob, advantages, response_mask,
                                  loss_agg_mode="token_mean", config=None,
                                  rollout_is_weights=None, **kwargs):
    """RLSD variant: contrast teacher_correct vs teacher_wrong instead of student.

    Structurally identical to RLSD but the evidence ratio uses the pairwise
    teacher contrast e_ctr = teacher_correct_lp - teacher_wrong_lp in place of
    e_c = teacher_correct_lp - student_lp. Both correct and wrong teacher
    contexts come from the single-pair data path.
    Added a -60/60 clamp on the exponent because e_ctr (teacher vs teacher) can
    be much larger in magnitude than e_c (teacher vs student); epsilon_w still
    bounds the final weight.
    """
    teacher_correct_lp = kwargs.get("teacher_log_probs")
    teacher_wrong_lp = kwargs.get("teacher_wrong_log_probs")
    assert teacher_correct_lp is not None and teacher_wrong_lp is not None, (
        "rlsd_ectr requires teacher_log_probs and teacher_wrong_log_probs."
    )
    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    epsilon_w = float(_sd_cfg(config, "epsilon_w", 0.2))
    lam = _sd_effective_lam(config, kwargs.get("global_steps"), default=0.5)

    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()

    with torch.no_grad():
        delta_t = teacher_correct_lp - teacher_wrong_lp
        scaled = (advantages.sign() * delta_t).clamp(min=-60.0, max=60.0)
        w_t = scaled.exp().clamp(1 - epsilon_w, 1 + epsilon_w)
        mod_adv = advantages * ((1 - lam) + lam * w_t)

    surr1 = ratio * mod_adv
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * mod_adv
    per_token_loss = -torch.min(surr1, surr2)
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {
        "pg_loss": loss.item(),
        "pg_clipfrac": ((ratio - 1).abs() > epsilon).float().mean().item(),
        "rlsd_ectr_lam": lam,
        "rlsd_ectr_e_ctr_abs_mean": verl_F.masked_mean(delta_t.abs(), response_mask).item()}
    metrics.update(_sd_distribution_metrics(delta_t, response_mask, "rlsd_ectr_e_ctr", include_percentiles=False))
    metrics.update(_sd_teacher_metrics(teacher_correct_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("rlcsd")
def compute_policy_loss_rlcsd(old_log_prob, log_prob, advantages, response_mask,
                                loss_agg_mode="token_mean", config=None,
                                rollout_is_weights=None, **kwargs):
    """RLCSD: K-marginal wrong-hint contrast policy loss.

    Wrong-hint term is replaced by the
    log of the uniform mixture of teacher distributions over up to K non-self
    negative siblings:

        e_ctr_t = log P_T(y_t | x, y_c*, y_<t)
                  - log[ (1/K) * sum_k P_T(y_t | x, y_w^(k), y_<t) ]

    Implemented as logsumexp(teacher_wrong_multi_lp) - log(K_valid) per row,
    with -inf masking on padded K-slots so they don't contribute. All other
    hyperparameters reuse the rlcsd_* keys (tau/beta/lam/delta/eta/clip)
    so 
    """
    teacher_correct_lp = kwargs.get("teacher_log_probs")
    teacher_wrong_multi_lp = kwargs.get("teacher_wrong_multi_log_probs")
    teacher_wrong_multi_valid_mask = kwargs.get("teacher_wrong_multi_valid_mask")
    assert teacher_correct_lp is not None, "rlcsd_5 requires teacher_log_probs."
    assert teacher_wrong_multi_lp is not None and teacher_wrong_multi_valid_mask is not None, (
        "rlcsd_5 requires teacher_wrong_multi_log_probs and teacher_wrong_multi_valid_mask."
    )

    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    tau = max(float(_sd_cfg(config, "rlcsd_tau", _sd_cfg(config, "tau", 0.03))), 1e-6)
    beta = float(_sd_cfg(config, "rlcsd_beta", 0.2))
    lam = float(_sd_cfg(config, "rlcsd_lam", _sd_cfg(config, "lam", 0.5)))
    delta = max(float(_sd_cfg(config, "rlcsd_delta", 0.02)), 0.0)
    eta = max(float(_sd_cfg(config, "rlcsd_eta", 1.0)), 0.0)
    residual_clip_low = float(_sd_cfg(config, "rlcsd_residual_clip_low", -0.5))
    residual_clip_high = float(_sd_cfg(config, "rlcsd_residual_clip_high", 0.5))
    if residual_clip_low > residual_clip_high:
        residual_clip_low, residual_clip_high = residual_clip_high, residual_clip_low

    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

    with torch.no_grad():
        # Marginalize wrong-hint logp over K valid siblings:
        # log_marginal_w = logsumexp_k(lp_k) - log(K_valid)
        valid_mask_bool = teacher_wrong_multi_valid_mask.to(dtype=torch.bool)
        masked_w_lp = teacher_wrong_multi_lp.masked_fill(
            ~valid_mask_bool.unsqueeze(-1), float("-inf"))
        K_valid = valid_mask_bool.to(dtype=teacher_wrong_multi_lp.dtype).sum(dim=1).clamp(min=1.0)
        log_marginal_w = torch.logsumexp(masked_w_lp, dim=1) - torch.log(K_valid).unsqueeze(-1)

        e_ctr = teacher_correct_lp - log_marginal_w
        s_t = torch.tanh(e_ctr / tau)
        preclip_residual = beta * lam * s_t
        residual = preclip_residual.clamp(min=residual_clip_low, max=residual_clip_high)
        selected = (residual.abs() > delta) & response_mask.bool()
        selected_float = selected.float()
        selected_count = selected_float.sum()
        residual_for_adv = selected_float * residual
        response_token_count = response_mask.sum()
        if selected_float.dim() >= 2:
            valid_sample_mask = response_mask.sum(dim=-1) > 0
            sample_count = valid_sample_mask.float().sum()
        else:
            sample_count = torch.as_tensor(1.0, dtype=selected_count.dtype, device=selected_count.device)
        raw_mod_adv = advantages + residual_for_adv
        mod_adv = torch.where(
            advantages > 0,
            raw_mod_adv.clamp_min(0.0),
            torch.where(advantages < 0, raw_mod_adv.clamp_max(0.0), torch.zeros_like(raw_mod_adv)))
        K_valid_mean = K_valid.mean()

    ratio_clamped = ratio.clamp(1 - epsilon, 1 + epsilon)
    nonselected_mask = response_mask * (1.0 - selected_float)
    selected_mask = response_mask * selected_float

    surr_nonselected1 = ratio * advantages
    surr_nonselected2 = ratio_clamped * advantages
    nonselected_per_token_loss = -torch.min(surr_nonselected1, surr_nonselected2)

    surr_selected1 = ratio * mod_adv
    surr_selected2 = ratio_clamped * mod_adv
    selected_per_token_loss = -torch.min(surr_selected1, surr_selected2)
    if rollout_is_weights is not None:
        nonselected_per_token_loss = nonselected_per_token_loss * rollout_is_weights
        selected_per_token_loss = selected_per_token_loss * rollout_is_weights
    nonselected_loss = _sd_token_mean(nonselected_per_token_loss, nonselected_mask)
    selected_loss = _sd_token_mean(selected_per_token_loss, selected_mask)
    loss = nonselected_loss + eta * selected_loss

    metrics = {
        "pg_loss": loss.item(),
        "pg_clipfrac": verl_F.masked_mean(((ratio - 1).abs() > epsilon).float(), response_mask).item(),
        "ppo_kl": ppo_kl.item(),
        "rlcsd_selected_token_count": selected_count.item(),
        "rlcsd_response_token_count": response_token_count.item(),
        "rlcsd_sample_count": sample_count.item(),
        "rlcsd_K_valid_mean": K_valid_mean.item()}
    metrics.update(_sd_distribution_metrics(e_ctr, response_mask, "rlcsd_e_ctr", include_percentiles=False))
    metrics.update(_sd_distribution_metrics(residual, selected_mask, "rlcsd_selected_residual"))
    return loss, metrics


@register_policy_loss("sdpo")
def compute_policy_loss_sdpo(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """SDPO: JSD / KL distillation with optional full-vocab or top-k views and IS correction."""
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "SDPO requires teacher_log_probs but got None. "
        "teacher_mode=ema requires EMA implementation in engine_workers."
    )
    alpha = float(_sd_cfg(config, "alpha", 0.5))
    is_clip = float(_sd_cfg(config, "is_clip", 2.0))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    if full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="student")
        per_token_kl = _sd_generalized_jsd(student_distill, teacher_distill, alpha)
    else:
        if alpha != 1.0:
            raise AssertionError("Non-full-logit SDPO only supports alpha=1.0 (reverse KL estimator).")
        log_ratio = log_prob - teacher_lp.detach()
        per_token_kl = log_ratio.detach() * log_prob

    is_ratio = (log_prob - old_log_prob).detach().clamp(-20, 20).exp().clamp(max=is_clip)
    per_token_loss = per_token_kl * is_ratio
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {
        "distill_loss": loss.item(),
        "is_ratio": is_ratio.mean().item(),
        "distill_alpha": alpha,
        "distill_topk": float(top_k_distill),
        "distill_full_logit": float(full_logit_distill)}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("srpo")
def compute_policy_loss_srpo(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """SRPO: Sample-Routed Policy Optimization (arXiv:2604.02288).

    Routes each rollout to the supervision signal best suited to its learning
    status: correct rollouts go through GRPO's PPO-clip surrogate (reward-aligned,
    sequence-level advantage), while incorrect rollouts with available teacher
    information go through SDPO's logit-level distillation, reweighted token-wise
    by teacher entropy (Dynamic-Weighted SDPO).

    The combined objective follows Eq. (Lfinal) of the paper:
        L = (Σ_{i,t} z^GRPO_i ℓ^GRPO_{i,t} + Σ_{i,t} z^SDPO_i ℓ^DW-SDPO_{i,t})
            / (Σ_{i,t} z^GRPO_i + Σ_{i,t} z^SDPO_i)
    where z^SDPO_i = (1 - c_i) m_i and z^GRPO_i = 1 - z^SDPO_i.

    The correctness flag c_i is read from the per-rollout sign of the GRPO
    advantage (a positive group-normalized advantage corresponds to a higher
    reward than the group mean, equivalent to "correct" in the binary-reward
    regime when the group is mixed). Teacher availability m_i is taken to be 1
    whenever teacher_log_probs are present.

    Hyperparameters (read from config.policy_loss):
        srpo_beta:           dynamic-weighting temperature β (default 1.0).
        alpha:               JSD mixing coefficient for the SDPO branch
                             (0 = forward KL, 1 = reverse KL, 0<α<1 = JSD).
        full_logit_distill:  use full-vocab JSD (True) or k1 reverse-KL (False).
        top_k_distill:       sparse top-k support size for the JSD branch.
        is_clip:             upper clamp on the SDPO IS ratio (default 2.0).
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    teacher_ent = kwargs.get("teacher_entropy")
    assert teacher_lp is not None, (
        "SRPO requires teacher_log_probs (feedback-conditioned self-teacher). "
        "Ensure the SDPO teacher pipeline is enabled for this method."
    )
    assert teacher_ent is not None, (
        "SRPO requires teacher_entropy for entropy-aware dynamic weighting."
    )

    beta = float(_sd_cfg(config, "srpo_beta", 1.0))
    alpha = float(_sd_cfg(config, "alpha", 0.5))
    is_clip = float(_sd_cfg(config, "is_clip", 2.0))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    valid_mask = response_mask.to(log_prob.dtype)

    # ---- Sample-level routing: c_i from per-rollout advantage sign --------
    rollout_token_count = valid_mask.sum(dim=1).clamp(min=1.0)
    rollout_adv = (advantages * valid_mask).sum(dim=1) / rollout_token_count
    correct_per_rollout = (rollout_adv > 0).to(log_prob.dtype)         # c_i
    sdpo_per_rollout = 1.0 - correct_per_rollout                        # z^SDPO_i (m_i = 1)
    grpo_per_rollout = 1.0 - sdpo_per_rollout                           # z^GRPO_i

    grpo_mask = grpo_per_rollout.unsqueeze(1) * valid_mask
    sdpo_mask = sdpo_per_rollout.unsqueeze(1) * valid_mask

    # ---- GRPO branch: dual-clip PPO surrogate, per-token (no reduction) ---
    negative_approx_kl = (log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = negative_approx_kl.exp()
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * ratio.clamp(1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    grpo_per_token = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        grpo_per_token = grpo_per_token * rollout_is_weights

    # Diagnostics on the GRPO branch only (clipping is meaningless for SDPO tokens).
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    grpo_diag_mask = grpo_mask if grpo_mask.sum() > 0 else response_mask
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), grpo_diag_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        (torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float()),
        grpo_diag_mask)

    # ---- SDPO branch: logit-level distillation, per-token (no reduction) --
    if full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="student")
        per_token_kl = _sd_generalized_jsd(student_distill, teacher_distill, alpha)
    else:
        if alpha != 1.0:
            raise AssertionError("Non-full-logit SRPO/SDPO only supports alpha=1.0 (k1 reverse-KL estimator).")
        log_ratio = log_prob - teacher_lp.detach()
        per_token_kl = log_ratio.detach() * log_prob

    is_ratio = (log_prob - old_log_prob).detach().clamp(-20, 20).exp().clamp(max=is_clip)
    sdpo_per_token = per_token_kl * is_ratio
    if rollout_is_weights is not None:
        sdpo_per_token = sdpo_per_token * rollout_is_weights

    # ---- Entropy-aware Dynamic Weighting on the SDPO branch ---------------
    # w̃_{i,t} = exp(-β · H_{i,t}); normalize so the mean weight over routed
    # tokens is 1, preserving the overall loss scale (paper §3.2).
    with torch.no_grad():
        teacher_ent_d = teacher_ent.detach().to(log_prob.dtype)
        log_w = -beta * teacher_ent_d
        # Numerically stable exp: subtract the max log-weight over routed tokens.
        # Tokens outside the SDPO route get -1e30 so they don't influence the
        # max; the (log_w - max) difference is then clamped to avoid +inf when
        # no SDPO tokens exist in the batch.
        log_w_for_max = torch.where(
            sdpo_mask > 0, log_w, torch.full_like(log_w, -1.0e30)
        )
        log_w_max = log_w_for_max.amax()
        w_unnorm = ((log_w - log_w_max).clamp(max=50.0)).exp() * sdpo_mask
        n_sdpo_tokens = sdpo_mask.sum().clamp(min=1.0)
        w_mean = w_unnorm.sum() / n_sdpo_tokens
        dw_weights = w_unnorm / w_mean.clamp(min=1e-12)

    sdpo_per_token = sdpo_per_token * dw_weights

    # ---- Final routed loss with unified denominator -----------------------
    grpo_term = (grpo_per_token * grpo_mask).sum()
    sdpo_term = (sdpo_per_token * sdpo_mask).sum()
    denom = (grpo_mask.sum() + sdpo_mask.sum()).clamp(min=1.0)
    loss = (grpo_term + sdpo_term) / denom

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "srpo/correct_rollout_frac": correct_per_rollout.mean().detach().item(),
        "srpo/sdpo_token_frac": (sdpo_mask.sum() / valid_mask.sum().clamp(min=1.0)).detach().item(),
        "srpo/grpo_loss_term": (grpo_term / denom).detach().item(),
        "srpo/sdpo_loss_term": (sdpo_term / denom).detach().item(),
        "srpo/dw_beta": beta,
        "srpo/dw_weight_mean": w_mean.detach().item(),
        "srpo/is_ratio": is_ratio.mean().detach().item(),
        "distill_alpha": alpha,
        "distill_topk": float(top_k_distill),
        "distill_full_logit": float(full_logit_distill)}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, teacher_ent))
    return loss, metrics
