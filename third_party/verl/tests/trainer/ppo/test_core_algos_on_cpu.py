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

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_policy_loss_opsd,
    compute_policy_loss_rlsd,
    compute_policy_loss_sdpo,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


class _DummyActorConfig:
    def __init__(self, policy_loss: dict):
        self.policy_loss = policy_loss


def test_sd_exclusion_ratio_train_mask_excludes_exact_high_or_low_count():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 99.0]])
    valid_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)

    keep_high, high_threshold = verl.trainer.ppo.core_algos._sd_exclusion_ratio_train_mask(
        values, valid_mask, ratio=0.5, exclude_high=True
    )
    keep_low, low_threshold = verl.trainer.ppo.core_algos._sd_exclusion_ratio_train_mask(
        values, valid_mask, ratio=0.5, exclude_high=False
    )

    assert keep_high.tolist() == [[True, True, False, False, False]]
    assert keep_low.tolist() == [[False, False, True, True, False]]
    assert high_threshold == pytest.approx(3.0)
    assert low_threshold == pytest.approx(2.0)


def test_sd_soft_or_entropy_clips_top_two_percent_before_minmax():
    entropy = torch.arange(100, dtype=torch.float32).view(1, 100)
    valid_mask = torch.ones_like(entropy, dtype=torch.bool)

    normalized = verl.trainer.ppo.core_algos._sd_valid_minmax_normalize(
        entropy, valid_mask, upper_quantile=0.98
    )

    # The 98th percentile is the cap, so all values above it saturate at one.
    assert normalized[0, 98].item() == pytest.approx(1.0)
    assert normalized[0, 99].item() == pytest.approx(1.0)
    assert normalized[0, 0].item() == pytest.approx(0.0)


def test_sd_valid_minmax_constant_values_are_zero():
    values = torch.full((1, 4), 3.0)
    valid_mask = torch.tensor([[True, True, False, True]])

    normalized = verl.trainer.ppo.core_algos._sd_valid_minmax_normalize(values, valid_mask)

    assert torch.equal(normalized, torch.zeros_like(values))


def test_opsd_soft_or_replaces_separate_entropy_and_distance_filters():
    shape = (1, 4)
    config = _DummyActorConfig(
        policy_loss={
            "beta": 0.0,
            "jsd_token_clip": 0.0,
            "full_logit_distill": False,
            "distance_filter_ratio": 0.5,
            "distance_filter_high": True,
            "soft_or_filter_enabled": True,
            "soft_or_filter_ratio": 0.5,
        }
    )
    log_prob = -torch.arange(1, 5, dtype=torch.float32).view(shape).requires_grad_()

    loss, metrics = compute_policy_loss_opsd(
        old_log_prob=torch.zeros(shape),
        log_prob=log_prob,
        advantages=torch.ones(shape),
        response_mask=torch.ones(shape),
        config=config,
        teacher_log_probs=torch.zeros(shape),
        student_entropy=torch.tensor([[0.0, 1.0, 2.0, 100.0]]),
        entropy_train_mask=torch.zeros(shape, dtype=torch.bool),
    )

    # Soft-OR excludes its two lowest scores and ignores both legacy masks.
    assert loss.item() == pytest.approx((9.0 + 16.0) / 2.0)
    assert metrics["distill_teacher_0_soft_or_selected_ratio"] == pytest.approx(0.5)
    assert metrics["distill_soft_or_selected_ratio"] == pytest.approx(0.5)


def test_opsd_entropy_distance_training_masks_use_union():
    shape = (1, 4)
    config = _DummyActorConfig(
        policy_loss={
            "beta": 0.0,
            "jsd_token_clip": 0.0,
            "full_logit_distill": False,
            "distance_filter_ratio": 0.5,
            "distance_filter_high": True,
        }
    )
    log_prob = -torch.arange(1, 5, dtype=torch.float32).view(shape).requires_grad_()
    entropy_train_mask = torch.tensor([[False, False, True, True]])

    loss, metrics = compute_policy_loss_opsd(
        old_log_prob=torch.zeros(shape),
        log_prob=log_prob,
        advantages=torch.ones(shape),
        response_mask=torch.ones(shape),
        config=config,
        teacher_log_probs=torch.zeros(shape),
        entropy_train_mask=entropy_train_mask,
    )

    # Distance keeps the two smallest-distance tokens, entropy keeps the other
    # two, so their training-set union contains all four positions.
    assert loss.item() == pytest.approx((1.0 + 4.0 + 9.0 + 16.0) / 4.0)
    assert metrics["distill_teacher_0_distance_selected_ratio"] == pytest.approx(0.5)
    assert metrics["distill_teacher_0_union_selected_ratio"] == pytest.approx(1.0)


def test_opsd_sparse_topk_matches_dense_teacher_support():
    old_log_prob = torch.zeros((1, 1), dtype=torch.float32)
    log_prob = torch.zeros((1, 1), dtype=torch.float32)
    advantages = torch.ones((1, 1), dtype=torch.float32)
    response_mask = torch.ones((1, 1), dtype=torch.float32)
    teacher_log_probs = torch.zeros((1, 1), dtype=torch.float32)
    student_all_log_probs = torch.log_softmax(torch.tensor([[[2.0, 0.0, -1.0]]], dtype=torch.float32), dim=-1)
    teacher_all_log_probs = torch.log_softmax(torch.tensor([[[1.0, 3.0, -2.0]]], dtype=torch.float32), dim=-1)
    _, teacher_topk_indices = torch.topk(teacher_all_log_probs, k=2, dim=-1)
    student_topk_log_probs = torch.gather(student_all_log_probs, dim=-1, index=teacher_topk_indices)
    teacher_topk_log_probs = torch.gather(teacher_all_log_probs, dim=-1, index=teacher_topk_indices)
    config = _DummyActorConfig(
        policy_loss={
            "beta": 0.0,
            "jsd_token_clip": 0.0,
            "full_logit_distill": True,
            "top_k_distill": 2,
            "distill_add_tail": True,
        }
    )

    dense_loss, _ = compute_policy_loss_opsd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        teacher_log_probs=teacher_log_probs,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )
    sparse_loss, _ = compute_policy_loss_opsd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        teacher_log_probs=teacher_log_probs,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
    )

    assert sparse_loss.item() == pytest.approx(dense_loss.item(), rel=1e-6, abs=1e-6)


def test_opsd_paint_selects_top_entropy_ratio_per_response_and_interpolates():
    shape = (2, 4)
    student_logits = torch.tensor(
        [
            [[2.0, 0.0], [1.0, 0.0], [0.5, 0.0], [0.1, 0.0]],
            [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0]],
        ],
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    student_entropy = torch.tensor([[2.0, 1.0, 0.5, 0.25], [1.0, 4.0, 2.0, 0.5]])
    teacher_entropy = torch.ones(shape)
    response_mask = torch.ones(shape)
    config = _DummyActorConfig(
        policy_loss={
            "beta": 0.0,
            "jsd_token_clip": 0.0,
            "full_logit_distill": True,
            "top_k_distill": 0,
            "paint_interp_strength": 0.25,
            "paint_top_fraction": 0.25,
        }
    )

    loss, metrics = compute_policy_loss_opsd(
        old_log_prob=torch.zeros(shape),
        log_prob=torch.zeros(shape),
        advantages=torch.ones(shape),
        response_mask=response_mask,
        config=config,
        teacher_log_probs=torch.zeros(shape),
        student_all_log_probs=student_log_probs,
        teacher_all_log_probs=teacher_log_probs,
        student_entropy=student_entropy,
        teacher_entropy=teacher_entropy,
    )

    # One position is selected independently from each four-token response.
    assert metrics["paint_selected_token_count"] == 2
    assert metrics["paint_selected_token_ratio"] == pytest.approx(0.25)
    assert loss.isfinite()
    loss.backward()
    assert student_logits.grad is not None
    assert student_logits.grad.isfinite().all()


def _multi_teacher_opsd_inputs():
    # Teacher 0 is farther at tokens 0/2; teacher 1 is farther at tokens 1/3.
    student_probs = torch.tensor(
        [[
            [[0.45, 0.45], [0.45, 0.45], [0.45, 0.45], [0.45, 0.45]],
            [[0.45, 0.45], [0.45, 0.45], [0.45, 0.45], [0.45, 0.45]],
        ]],
        dtype=torch.float32,
    )
    close = torch.tensor([0.45, 0.45])
    far = torch.tensor([0.80, 0.10])
    teacher_probs = torch.stack(
        [
            torch.stack([far, close, far, close]),
            torch.stack([close, far, close, far]),
        ],
        dim=0,
    ).unsqueeze(0)
    return {
        "old_log_prob": torch.zeros((1, 4)),
        "log_prob": torch.zeros((1, 4)),
        "advantages": torch.ones((1, 4)),
        "response_mask": torch.ones((1, 4)),
        "teacher_log_probs": torch.zeros((1, 4)),
        "student_multi_topk_log_probs": student_probs.log().requires_grad_(),
        "teacher_multi_topk_log_probs": teacher_probs.log(),
        "teacher_multi_valid_mask": torch.ones((1, 2), dtype=torch.bool),
    }


def _entropy_grouped_opsd_inputs():
    # Token 0 is the sole high-entropy token. Prefix teacher 0 is closer there.
    # Across the remaining tokens, suffix teachers 2 and 3 alternate as farther.
    student_probs = torch.full((1, 4, 5, 2), 0.45, dtype=torch.float32)
    close = torch.tensor([0.45, 0.45])
    medium = torch.tensor([0.60, 0.30])
    far = torch.tensor([0.80, 0.10])
    teacher_probs = torch.stack(
        [
            torch.stack([close, medium, medium, medium, medium]),
            torch.stack([far, close, close, close, close]),
            torch.stack([close, far, close, far, close]),
            torch.stack([close, close, far, close, far]),
        ],
        dim=0,
    ).unsqueeze(0)
    return {
        "old_log_prob": torch.zeros((1, 5)),
        "log_prob": torch.zeros((1, 5)),
        "advantages": torch.ones((1, 5)),
        "response_mask": torch.ones((1, 5)),
        "teacher_log_probs": torch.zeros((1, 5)),
        "student_entropy": torch.tensor([[10.0, 4.0, 3.0, 2.0, 1.0]]),
        "student_multi_topk_log_probs": student_probs.log().requires_grad_(),
        "teacher_multi_topk_log_probs": teacher_probs.log(),
        "teacher_multi_valid_mask": torch.ones((1, 4), dtype=torch.bool),
    }


def test_opsd_multi_teacher_entropy_grouped_distance_routes_by_entropy_and_distance():
    inputs = _entropy_grouped_opsd_inputs()
    student = inputs["student_multi_topk_log_probs"]
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_high_ratio": 0.2,
    })

    loss, metrics = compute_policy_loss_opsd(config=config, **inputs)
    loss.backward()

    assert metrics["distill_routing_high_entropy_token_count"] == 1
    assert metrics["distill_routing_rest_entropy_token_count"] == 4
    assert metrics["distill_routing_high_entropy_ratio"] == pytest.approx(0.2)
    assert metrics["distill_routing_high_prefix_0_5_count"] == 1
    assert metrics["distill_routing_high_prefix_0_7_count"] == 0
    assert metrics["distill_routing_rest_suffix_0_5_count"] == 2
    assert metrics["distill_routing_rest_suffix_0_7_count"] == 2
    assert metrics["distill_routing_high_prefix_distance_gap_token_count"] == 1
    assert metrics["distill_routing_rest_suffix_distance_gap_token_count"] == 4
    assert metrics["distill_routing_high_prefix_distance_gap_mean"] > 0
    assert metrics["distill_routing_rest_suffix_distance_gap_mean"] > 0
    assert torch.count_nonzero(student.grad[:, 0, 0]).item() > 0
    assert torch.count_nonzero(student.grad[:, 1]).item() == 0
    assert torch.count_nonzero(student.grad[:, 2, [1, 3]]).item() > 0
    assert torch.count_nonzero(student.grad[:, 3, [2, 4]]).item() > 0


def test_opsd_entropy_grouped_distance_can_route_over_full_teacher_pool():
    inputs = _entropy_grouped_opsd_inputs()
    teacher_probs = inputs["teacher_multi_topk_log_probs"].exp()
    # The highest-entropy token's nearest teacher is suffix_0.5 (index 2).
    teacher_probs[0, 0, 0] = torch.tensor([0.60, 0.30])
    teacher_probs[0, 1, 0] = torch.tensor([0.80, 0.10])
    teacher_probs[0, 2, 0] = torch.tensor([0.45, 0.45])
    teacher_probs[0, 3, 0] = torch.tensor([0.75, 0.15])
    # A rest-group token's farthest teacher is prefix_0.7 (index 1).
    teacher_probs[0, 0, 1] = torch.tensor([0.45, 0.45])
    teacher_probs[0, 1, 1] = torch.tensor([0.80, 0.10])
    teacher_probs[0, 2, 1] = torch.tensor([0.60, 0.30])
    teacher_probs[0, 3, 1] = torch.tensor([0.45, 0.45])
    inputs["teacher_multi_topk_log_probs"] = teacher_probs.log()
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_high_ratio": 0.2,
        "entropy_router_fixed_teacher_type": False,
    })

    _, metrics = compute_policy_loss_opsd(config=config, **inputs)

    assert metrics["distill_routing_fixed_teacher_type_enabled"] == 0
    assert metrics["distill_routing_high_teacher_2_count"] == 1
    assert metrics["distill_routing_rest_teacher_1_count"] >= 1
    assert "distill_routing_high_prefix_0_5_count" not in metrics
    assert "distill_routing_high_prefix_0_7_count" not in metrics
    assert metrics["distill_routing_high_nearest_distance_gap_token_count"] == 1
    assert metrics["distill_routing_rest_farthest_distance_gap_token_count"] == 4


def _entropy_voting_opsd_inputs():
    inputs = _entropy_grouped_opsd_inputs()
    inputs["log_prob"] = torch.full((1, 5), -2.0)
    # Positive majority (teachers 0, 1, 3), with teacher 2 as the sole minority.
    inputs["teacher_multi_log_probs"] = torch.tensor(
        [[
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.5, -3.0, -3.0, -3.0, -3.0],
            [-3.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
        ]]
    )
    return inputs


@pytest.mark.parametrize(
    ("vote_mode", "expected_high_teacher"),
    [("majority", 0), ("minority_high", 2)],
)
def test_opsd_entropy_grouped_distance_votes_before_distance_routing(
    vote_mode, expected_high_teacher
):
    inputs = _entropy_voting_opsd_inputs()
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_high_ratio": 0.2,
        "entropy_router_fixed_teacher_type": False,
        "entropy_router_vote_mode": vote_mode,
    })

    _, metrics = compute_policy_loss_opsd(config=config, **inputs)

    assert metrics["distill_routing_vote_mode_enabled"] == 1
    assert metrics[f"distill_routing_high_teacher_{expected_high_teacher}_count"] == 1
    assert metrics["distill_routing_high_minority_singleton_count"] == (
        1 if vote_mode == "minority_high" else 0
    )
    # Token 1 is low entropy; among its positive-majority teachers, teacher 2 is farthest.
    assert metrics["distill_routing_rest_teacher_2_count"] >= 1


def test_opsd_entropy_grouped_distance_zero_high_ratio_routes_all_to_rest():
    inputs = _entropy_voting_opsd_inputs()
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_high_ratio": 0.0,
        "entropy_router_fixed_teacher_type": False,
        "entropy_router_vote_mode": "majority",
    })

    _, metrics = compute_policy_loss_opsd(config=config, **inputs)

    assert metrics["distill_routing_high_entropy_token_count"] == 0
    assert metrics["distill_routing_rest_entropy_token_count"] == 5
    assert metrics["distill_routing_high_entropy_ratio"] == 0
    assert sum(
        metrics[f"distill_routing_high_teacher_{idx}_count"] for idx in range(4)
    ) == 0
    assert sum(
        metrics[f"distill_routing_rest_teacher_{idx}_count"] for idx in range(4)
    ) == 5
    assert metrics["distill_routing_rest_teacher_2_count"] >= 1


def _majority_selection_config(selection):
    return _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_high_ratio": 0.0,
        "entropy_router_fixed_teacher_type": False,
        "entropy_router_vote_mode": "majority",
        "entropy_router_teacher_selection": selection,
    })


def test_opsd_majority_random_selects_exactly_one_candidate_per_token():
    inputs = _entropy_voting_opsd_inputs()
    torch.manual_seed(1234)

    _, metrics = compute_policy_loss_opsd(
        config=_majority_selection_config("random"), **inputs
    )

    assert sum(
        metrics[f"distill_routing_rest_teacher_{idx}_count"] for idx in range(4)
    ) == 5
    first_counts = [
        metrics[f"distill_routing_rest_teacher_{idx}_count"] for idx in range(4)
    ]
    torch.manual_seed(1234)
    _, repeated_metrics = compute_policy_loss_opsd(
        config=_majority_selection_config("random"), **_entropy_voting_opsd_inputs()
    )
    assert first_counts == [
        repeated_metrics[f"distill_routing_rest_teacher_{idx}_count"]
        for idx in range(4)
    ]


def test_opsd_majority_nearest_selects_nearest_candidate_for_every_token():
    inputs = _entropy_voting_opsd_inputs()

    _, metrics = compute_policy_loss_opsd(
        config=_majority_selection_config("nearest"), **inputs
    )

    assert metrics["distill_routing_rest_teacher_0_count"] == 1
    assert metrics["distill_routing_rest_teacher_1_count"] == 0
    assert metrics["distill_routing_rest_teacher_2_count"] == 2
    assert metrics["distill_routing_rest_teacher_3_count"] == 2


def test_opsd_majority_can_select_with_reverse_kl_and_train_with_forward_kl():
    # On the teacher top-k + tail distributions below, teacher 0 has the
    # larger forward KL while teacher 1 has the larger reverse KL.
    student_topk = torch.tensor([[[[0.6, 0.3]], [[0.6, 0.3]]]])
    teacher_topk = torch.tensor([[[[0.02, 0.09]], [[0.01, 0.18]]]])
    inputs = {
        "old_log_prob": torch.zeros((1, 1)),
        "log_prob": torch.full((1, 1), -2.0),
        "advantages": torch.ones((1, 1)),
        "response_mask": torch.ones((1, 1)),
        "teacher_log_probs": torch.zeros((1, 1)),
        "student_entropy": torch.ones((1, 1)),
        "student_multi_topk_log_probs": student_topk.log().requires_grad_(),
        "teacher_multi_topk_log_probs": teacher_topk.log(),
        "teacher_multi_valid_mask": torch.ones((1, 2), dtype=torch.bool),
        # Both teachers are in the positive majority on the observed token.
        "teacher_multi_log_probs": torch.full((1, 2, 1), -1.0),
    }
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "teacher_selection_beta": 1.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_high_ratio": 0.0,
        "entropy_router_fixed_teacher_type": False,
        "entropy_router_vote_mode": "majority",
    })

    loss, metrics = compute_policy_loss_opsd(config=config, **inputs)
    student_distill, teacher_distill = verl.trainer.ppo.core_algos._sd_prepare_sparse_topk_log_probs(
        inputs["student_multi_topk_log_probs"][:, 1],
        inputs["teacher_multi_topk_log_probs"][:, 1],
        add_tail=True,
    )
    expected_forward_loss = verl.trainer.ppo.core_algos._sd_generalized_jsd(
        student_distill, teacher_distill, mix=0.0, vocab_clip=None
    ).mean()

    assert metrics["distill_routing_rest_teacher_0_count"] == 0
    assert metrics["distill_routing_rest_teacher_1_count"] == 1
    assert metrics["distill_beta"] == 0.0
    assert metrics["distill_teacher_selection_beta"] == 1.0
    assert loss.detach().item() == pytest.approx(expected_forward_loss.detach().item())


def test_opsd_majority_mean_uses_every_majority_candidate():
    inputs = _entropy_voting_opsd_inputs()

    loss, metrics = compute_policy_loss_opsd(
        config=_majority_selection_config("mean"), **inputs
    )
    per_teacher = []
    for teacher_idx in range(4):
        student_lp, teacher_lp = verl.trainer.ppo.core_algos._sd_prepare_sparse_topk_log_probs(
            inputs["student_multi_topk_log_probs"][:, teacher_idx],
            inputs["teacher_multi_topk_log_probs"][:, teacher_idx],
            add_tail=True,
        )
        per_teacher.append(
            verl.trainer.ppo.core_algos._sd_generalized_jsd(
                student_lp, teacher_lp, mix=0.0, vocab_clip=None
            )
        )
    per_teacher = torch.stack(per_teacher, dim=1)
    majority_mask = torch.zeros_like(per_teacher, dtype=torch.bool)
    majority_mask[:, [0, 1, 3], 0] = True
    majority_mask[:, [0, 2, 3], 1:] = True
    expected = (
        (per_teacher * majority_mask).sum(dim=1)
        / majority_mask.sum(dim=1)
    ).mean()

    assert loss.detach().item() == pytest.approx(expected.detach().item())
    assert metrics["distill_routing_majority_mean_enabled"] == 1
    assert metrics["distill_routing_majority_candidate_count_mean"] == pytest.approx(3.0)


@pytest.mark.parametrize("filter_config", [
    {"distance_filter_ratio": 0.1},
    {"soft_or_filter_enabled": True, "soft_or_filter_ratio": 0.1},
])
def test_opsd_majority_mean_rejects_distance_based_filters(filter_config):
    inputs = _entropy_voting_opsd_inputs()
    config = _majority_selection_config("mean")
    config.policy_loss.update(filter_config)

    with pytest.raises(ValueError, match="does not support"):
        compute_policy_loss_opsd(config=config, **inputs)


def test_opsd_entropy_grouped_distance_empty_minority_falls_back_to_all_teachers():
    inputs = _entropy_voting_opsd_inputs()
    inputs["teacher_multi_log_probs"] = torch.full((1, 4, 5), -1.0)
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_high_ratio": 0.2,
        "entropy_router_fixed_teacher_type": False,
        "entropy_router_vote_mode": "minority_high",
    })

    _, metrics = compute_policy_loss_opsd(config=config, **inputs)

    assert metrics["distill_routing_high_minority_empty_fallback_count"] == 1
    assert metrics["distill_routing_high_teacher_0_count"] == 1


def test_opsd_entropy_grouped_distance_vote_count_tie_uses_larger_magnitude_sum():
    inputs = _entropy_voting_opsd_inputs()
    # Two teachers are positive and two negative. The negative side has the
    # larger absolute sum, so it is the majority side for routing purposes.
    deltas = torch.tensor([0.2, 0.3, -0.6, -0.1]).view(1, 4, 1)
    inputs["teacher_multi_log_probs"] = -2.0 + deltas.expand(1, 4, 5)
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_high_ratio": 0.2,
        "entropy_router_fixed_teacher_type": False,
        "entropy_router_vote_mode": "majority",
    })

    _, metrics = compute_policy_loss_opsd(config=config, **inputs)

    assert metrics["distill_routing_vote_count_tie_token_count"] == 5
    assert metrics["distill_routing_vote_magnitude_tie_token_count"] == 0
    assert metrics["distill_routing_high_teacher_2_count"] == 1


def test_opsd_entropy_grouped_distance_vote_rejects_fixed_teacher_types():
    inputs = _entropy_voting_opsd_inputs()
    config = _DummyActorConfig(policy_loss={
        "full_logit_distill": True,
        "top_k_distill": 2,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_vote_mode": "majority",
    })

    with pytest.raises(ValueError, match="full teacher pool"):
        compute_policy_loss_opsd(config=config, **inputs)


@pytest.mark.parametrize("num_teachers", [3, 5])
def test_opsd_entropy_grouped_distance_supports_arbitrary_full_teacher_pool(num_teachers):
    num_tokens = 5
    student_probs = torch.full((1, num_teachers, num_tokens, 2), 0.45)
    medium = torch.tensor([0.60, 0.30])
    far = torch.tensor([0.80, 0.10])
    close = torch.tensor([0.45, 0.45])
    teacher_probs = torch.stack(
        [torch.stack([medium] * num_tokens) for _ in range(num_teachers)], dim=0
    ).unsqueeze(0)
    teacher_probs[0, 1] = far
    teacher_probs[0, 2] = close
    inputs = {
        "old_log_prob": torch.zeros((1, num_tokens)),
        "log_prob": torch.zeros((1, num_tokens)),
        "advantages": torch.ones((1, num_tokens)),
        "response_mask": torch.ones((1, num_tokens)),
        "teacher_log_probs": torch.zeros((1, num_tokens)),
        "student_entropy": torch.tensor([[10.0, 4.0, 3.0, 2.0, 1.0]]),
        "student_multi_topk_log_probs": student_probs.log().requires_grad_(),
        "teacher_multi_topk_log_probs": teacher_probs.log(),
        "teacher_multi_valid_mask": torch.ones((1, num_teachers), dtype=torch.bool),
    }
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "entropy_grouped_distance",
        "entropy_router_high_ratio": 0.2,
        "entropy_router_fixed_teacher_type": False,
    })

    _, metrics = compute_policy_loss_opsd(config=config, **inputs)

    assert metrics["distill_routing_high_teacher_2_count"] == 1
    assert metrics["distill_routing_rest_teacher_1_count"] == 4
    assert sum(
        metrics[f"distill_routing_high_teacher_{idx}_count"]
        for idx in range(num_teachers)
    ) == 1
    assert sum(
        metrics[f"distill_routing_rest_teacher_{idx}_count"]
        for idx in range(num_teachers)
    ) == 4


def test_opsd_entropy_grouped_distance_requires_four_teachers():
    inputs = _multi_teacher_opsd_inputs()
    config = _DummyActorConfig(policy_loss={
        "full_logit_distill": True,
        "top_k_distill": 2,
        "multi_teacher_aggregation": "entropy_grouped_distance",
    })

    with pytest.raises(ValueError, match="exactly four teachers"):
        compute_policy_loss_opsd(
            config=config,
            student_entropy=torch.ones((1, 4)),
            **inputs,
        )


def test_opsd_multi_teacher_token_max_distance_routes_per_token_and_backpropagates():
    inputs = _multi_teacher_opsd_inputs()
    student = inputs["student_multi_topk_log_probs"]
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "token_max_distance",
    })

    loss, metrics = compute_policy_loss_opsd(config=config, **inputs)
    loss.backward()

    assert metrics["distill_routing_teacher_0_assigned_token_count"] == 2
    assert metrics["distill_routing_teacher_1_assigned_token_count"] == 2
    assert metrics["distill_routing_teacher_0_pre_filter_ratio"] == pytest.approx(0.5)
    assert metrics["distill_routing_teacher_1_pre_filter_ratio"] == pytest.approx(0.5)
    assert metrics["distill_routing_adjacent_switch_rate"] == pytest.approx(1.0)
    assert metrics["distill_routing_near_tie_ratio"] == pytest.approx(0.0)
    assert torch.count_nonzero(student.grad[:, 0, 1::2]).item() == 0
    assert torch.count_nonzero(student.grad[:, 1, 0::2]).item() == 0
    assert torch.count_nonzero(student.grad[:, 0, 0::2]).item() > 0
    assert torch.count_nonzero(student.grad[:, 1, 1::2]).item() > 0


def test_opsd_multi_teacher_token_max_distance_filters_once_globally():
    inputs = _multi_teacher_opsd_inputs()
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "token_max_distance",
        "distance_filter_ratio": 0.5,
        "distance_filter_high": False,
    })
    # Make the winning teachers progressively farther so the global filter has
    # a deterministic bottom half to reject across both teachers.
    teacher = inputs["teacher_multi_topk_log_probs"].exp()
    teacher[0, 0, 0] = torch.tensor([0.55, 0.35])
    teacher[0, 1, 1] = torch.tensor([0.65, 0.25])
    teacher[0, 0, 2] = torch.tensor([0.75, 0.15])
    teacher[0, 1, 3] = torch.tensor([0.85, 0.05])
    inputs["teacher_multi_topk_log_probs"] = teacher.log()

    _, metrics = compute_policy_loss_opsd(config=config, **inputs)

    assert metrics["distill_routing_final_token_count"] == 2
    assert metrics["distill_routing_distance_selected_ratio"] == pytest.approx(0.5)
    assert metrics["distill_routing_teacher_0_final_token_count"] == 1
    assert metrics["distill_routing_teacher_1_final_token_count"] == 1


def test_opsd_multi_teacher_token_max_distance_ignores_invalid_teacher():
    inputs = _multi_teacher_opsd_inputs()
    inputs["teacher_multi_valid_mask"] = torch.tensor([[True, False]])
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "token_max_distance",
    })

    _, metrics = compute_policy_loss_opsd(config=config, **inputs)

    assert metrics["distill_routing_teacher_0_assigned_token_count"] == 4
    assert metrics["distill_routing_teacher_1_assigned_token_count"] == 0
    assert metrics["distill_routing_active_teacher_count"] == 1


def test_opsd_multi_teacher_token_max_distance_applies_soft_or_after_routing():
    inputs = _multi_teacher_opsd_inputs()
    config = _DummyActorConfig(policy_loss={
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
        "multi_teacher_aggregation": "token_max_distance",
        "soft_or_filter_enabled": True,
        "soft_or_filter_ratio": 0.5,
    })

    _, metrics = compute_policy_loss_opsd(
        config=config,
        student_entropy=torch.tensor([[0.0, 1.0, 2.0, 100.0]]),
        **inputs,
    )

    assert metrics["distill_routing_final_token_count"] == 2
    assert metrics["distill_routing_soft_or_selected_ratio"] == pytest.approx(0.5)
    assert metrics["distill_routing_teacher_0_final_token_count"] == 1
    assert metrics["distill_routing_teacher_1_final_token_count"] == 1


def test_opsd_multi_teacher_default_matches_explicit_teacher_mean():
    implicit_inputs = _multi_teacher_opsd_inputs()
    explicit_inputs = _multi_teacher_opsd_inputs()
    base_policy_loss = {
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "full_logit_distill": True,
        "top_k_distill": 2,
        "distill_add_tail": True,
    }
    implicit_loss, implicit_metrics = compute_policy_loss_opsd(
        config=_DummyActorConfig(policy_loss=base_policy_loss), **implicit_inputs
    )
    explicit_loss, explicit_metrics = compute_policy_loss_opsd(
        config=_DummyActorConfig(policy_loss={
            **base_policy_loss,
            "multi_teacher_aggregation": "teacher_mean",
        }),
        **explicit_inputs,
    )

    assert implicit_loss.item() == pytest.approx(explicit_loss.item())
    assert implicit_metrics["distill_teacher_0_loss"] == pytest.approx(
        explicit_metrics["distill_teacher_0_loss"]
    )
    assert implicit_metrics["distill_teacher_1_loss"] == pytest.approx(
        explicit_metrics["distill_teacher_1_loss"]
    )


def test_opsd_rejects_unknown_multi_teacher_aggregation():
    inputs = _multi_teacher_opsd_inputs()
    config = _DummyActorConfig(policy_loss={
        "full_logit_distill": True,
        "top_k_distill": 2,
        "multi_teacher_aggregation": "unknown",
    })

    with pytest.raises(ValueError, match="multi_teacher_aggregation"):
        compute_policy_loss_opsd(config=config, **inputs)


def test_sdpo_sparse_topk_matches_dense_student_support():
    old_log_prob = torch.zeros((1, 1), dtype=torch.float32)
    log_prob = torch.zeros((1, 1), dtype=torch.float32)
    advantages = torch.ones((1, 1), dtype=torch.float32)
    response_mask = torch.ones((1, 1), dtype=torch.float32)
    teacher_log_probs = torch.zeros((1, 1), dtype=torch.float32)
    student_all_log_probs = torch.log_softmax(torch.tensor([[[3.0, 0.5, -1.0]]], dtype=torch.float32), dim=-1)
    teacher_all_log_probs = torch.log_softmax(torch.tensor([[[1.0, 2.0, -0.5]]], dtype=torch.float32), dim=-1)
    _, student_topk_indices = torch.topk(student_all_log_probs, k=2, dim=-1)
    student_topk_log_probs = torch.gather(student_all_log_probs, dim=-1, index=student_topk_indices)
    teacher_topk_log_probs = torch.gather(teacher_all_log_probs, dim=-1, index=student_topk_indices)
    config = _DummyActorConfig(
        policy_loss={
            "alpha": 0.5,
            "is_clip": 2.0,
            "full_logit_distill": True,
            "top_k_distill": 2,
            "distill_add_tail": True,
        }
    )

    dense_loss, _ = compute_policy_loss_sdpo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        teacher_log_probs=teacher_log_probs,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )
    sparse_loss, _ = compute_policy_loss_sdpo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        teacher_log_probs=teacher_log_probs,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
    )

    assert sparse_loss.item() == pytest.approx(dense_loss.item(), rel=1e-6, abs=1e-6)


def test_rlsd_lambda_decay_reduces_teacher_modulation():
    old_log_prob = torch.zeros((1, 2), dtype=torch.float32)
    log_prob = torch.zeros((1, 2), dtype=torch.float32)
    advantages = torch.ones((1, 2), dtype=torch.float32)
    response_mask = torch.ones((1, 2), dtype=torch.float32)
    teacher_log_probs = torch.full((1, 2), 0.4, dtype=torch.float32)
    config = _DummyActorConfig(
        policy_loss={
            "epsilon": 0.2,
            "epsilon_w": 0.2,
            "lam": 0.5,
            "lam_decay_steps": 10,
        }
    )

    loss_step_1, metrics_step_1 = compute_policy_loss_rlsd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        teacher_log_probs=teacher_log_probs,
        global_steps=1,
    )
    loss_step_11, metrics_step_11 = compute_policy_loss_rlsd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        teacher_log_probs=teacher_log_probs,
        global_steps=11,
    )

    assert metrics_step_1["rlsd_lam"] == pytest.approx(0.5)
    assert metrics_step_11["rlsd_lam"] == pytest.approx(0.0)
    assert loss_step_1.item() < loss_step_11.item()


def test_rlsd_applies_rollout_is_weights():
    old_log_prob = torch.zeros((1, 2), dtype=torch.float32)
    log_prob = torch.zeros((1, 2), dtype=torch.float32)
    advantages = torch.ones((1, 2), dtype=torch.float32)
    response_mask = torch.ones((1, 2), dtype=torch.float32)
    teacher_log_probs = torch.full((1, 2), 0.2, dtype=torch.float32)
    rollout_is_weights = torch.full((1, 2), 0.5, dtype=torch.float32)
    config = _DummyActorConfig(
        policy_loss={
            "epsilon": 0.2,
            "epsilon_w": 0.2,
            "lam": 0.0,
            "lam_decay_steps": 0,
        }
    )

    loss_no_is, _ = compute_policy_loss_rlsd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        teacher_log_probs=teacher_log_probs,
        global_steps=1,
    )
    loss_with_is, _ = compute_policy_loss_rlsd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        teacher_log_probs=teacher_log_probs,
        rollout_is_weights=rollout_is_weights,
        global_steps=1,
    )

    assert loss_with_is.item() == pytest.approx(loss_no_is.item() * 0.5)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_compute_policy_loss_flow_grpo() -> None:
    """Test flow-GRPO policy loss computation."""

    # prepare input
    batch_size = 8
    steps = 10
    rollout_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    advantages = torch.randn((batch_size, steps), dtype=torch.float32)
    response_mask = torch.ones((batch_size, steps), dtype=torch.int32)
    import os

    from hydra import compose, initialize_config_dir

    from verl.trainer.ppo.diffusion_algos import compute_policy_loss_flow_grpo
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config.actor import FSDPActorConfig

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
        cfg = compose(
            config_name="dp_actor",
            overrides=[
                "strategy=fsdp",
                "clip_ratio=0.0001",
                "clip_ratio_high=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPActorConfig = omega_conf_to_dataclass(cfg)

    for step in range(steps):
        pg_loss, pg_metrics = compute_policy_loss_flow_grpo(
            old_log_prob=rollout_log_probs[:, step],
            log_prob=current_log_probs[:, step],
            advantages=advantages[:, step],
            response_mask=response_mask[:, step],
            loss_agg_mode="token-mean",
            config=actor_config,
        )

        assert pg_loss.shape == ()
        assert isinstance(pg_loss.item(), float)
        assert "actor/ppo_kl" in pg_metrics.keys()


if __name__ == "__main__":
    unittest.main()
