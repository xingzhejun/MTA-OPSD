"""Shared OPSD-style prompt formatting and math answer grading helpers."""

from __future__ import annotations

import hashlib
import random
import re
from decimal import Decimal, InvalidOperation
from typing import Optional


LEGACY_DAPO_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: $Answer "
    '(without quotes) where $Answer is the answer to the problem.\n\n'
    'Remember to put your answer on its own line after "Answer:".'
)

BOXED_ANSWER_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# Variant for Base models: explicitly instruct the model to use <think> tags.
THINKING_BOXED_ANSWER_INSTRUCTION = (
    "Please first think step by step inside <think>...</think> tags, "
    "and then put your final answer within \\boxed{}."
)

# Legacy: R1-Zero-style system prompt that asked the model to wrap reasoning
# in <think>/</think>. A cold-start diagnostic on Qwen3-1.7B-Base showed the
# base model never emits </think> regardless of priming, so the structure
# could not be learned through RL alone. Base configs now share the exact
# same prompt as Instruct configs -- a single user message carrying the
# \boxed{} instruction, no system prompt, no <think> requirement. This
# constant is kept only for back-compat with any external imports.
R1_ZERO_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> tags "
    "and \\boxed{} respectively, i.e., "
    "<think> reasoning process here </think> \\boxed{answer here}."
)

# Unified lighter wording. Originally rlcsd_* used a lighter prompt while
# opsd / sdpo / rlsd / srpo used a longer "derive a fresh solution" framing,
# but the longer wording was unnecessary even for those methods, so all
# methods now share the same transition prompt. The two names are kept as
# aliases for backwards compatibility with existing imports.
TEACHER_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you understand "
    "the reasoning behind each step.\n"
)
RLCSD_TEACHER_TRANSITION_PROMPT = TEACHER_TRANSITION_PROMPT


# Named wrapper variants for the teacher prompt.  Each variant supplies a
# (framing, transition) pair; the rest of the user-message scaffolding
# (Problem header, === Reference Solution Begin/End ===, final boxed-answer
# instruction) is shared so positive and negative passes still align at the
# token level -- a hard requirement for RLCSD's CFG-style cancellation.
#
# "neutral" reproduces the current production wording verbatim, so the default
# behavior of build_teacher_messages is unchanged.  "verbose" and "terse" are
# the v1 robustness-experiment perturbations validated by the
# diagnose_prompt_robustness_dists.py probe.
TEACHER_WRAPPER_VARIANTS = {
    "neutral": {
        "framing": "Here is a reference solution to this problem:",
        "transition": TEACHER_TRANSITION_PROMPT,
    },
    "verbose": {
        "framing": (
            "Below is a fully worked-out solution by a senior mathematician "
            "known for exceptionally detailed, rigorous, and pedagogically "
            "thorough explanations that justify every algebraic step and "
            "discuss the underlying intuition:"
        ),
        "transition": (
            "\n\nNow, in the same elaborate, careful, fully-justified style "
            "-- explaining your intuition at each step, motivating every "
            "move, and double-checking each algebraic manipulation -- write "
            "your own solution.\n"
        ),
    },
    "terse": {
        "framing": (
            "A reference solution is shown below. Solutions in this "
            "collection are written in an extremely terse style: minimal "
            "prose, no restating the problem, no intuition, no verification "
            "-- just the essential computation in as few steps as possible:"
        ),
        "transition": (
            "\n\nWrite your own solution in the same terse, compact style. "
            "Be as brief as possible while still arriving at the correct "
            "result.\n"
        ),
    },
}


def normalize_wrapper_variant(variant: str) -> str:
    name = str(variant or "neutral").strip().lower()
    if name not in TEACHER_WRAPPER_VARIANTS:
        raise ValueError(
            f"Unsupported teacher_wrapper_variant={variant!r}. "
            f"Supported: {sorted(TEACHER_WRAPPER_VARIANTS)}."
        )
    return name

EVAL_DATA_SOURCES = {
    "amc23", "aime24", "aime25", "math500", "amo-bench", "minerva", "hmmt25",
    "kk_3to7_test", "kk_4to8_test", "kk_8", "kk_9", "kk_10", "kk_11",
}

KK_DATA_SOURCES = {
    "kk_3to7", "kk_3to7_test",
    "kk_4to8", "kk_4to8_test",
    "kk_8", "kk_9", "kk_10", "kk_11",
}

_SIMPLE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DEGREE_SUFFIX_RE = re.compile(r"(?:\^\{?\\circ\}?|\\degree|\\deg|\u00b0)$")
_PAINT_LATEX_SPAN_RE = re.compile(
    r"(?:\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\]|\\\(.*?\\\))",
    re.DOTALL,
)
_PAINT_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?")
_PAINT_COMMAND_RE = re.compile(
    r"(?:\\(?:frac|sqrt|binom)\s*\{[^{}]+\}(?:\s*\{[^{}]+\})?|"
    r"[A-Za-z0-9]+\s*(?:=|<=|>=|<|>|\\leq?|\\geq?)\s*[^,.;:\n]+)"
)


def strip_legacy_math_prompt(text: str) -> str:
    """Strip legacy/local prompt wrappers and keep only the raw problem text."""
    if text is None:
        return ""

    text = str(text).strip()
    legacy_prefix = f"{LEGACY_DAPO_INSTRUCTION}\n\n"
    if text.startswith(legacy_prefix):
        return text[len(legacy_prefix):].strip()

    train_prefix = "Problem: "
    for instruction in (BOXED_ANSWER_INSTRUCTION, THINKING_BOXED_ANSWER_INSTRUCTION):
        suffix = f"\n\n{instruction}"
        if text.startswith(train_prefix) and text.endswith(suffix):
            return text[len(train_prefix):-len(suffix)].strip()
        if text.endswith(suffix):
            return text[:-len(suffix)].strip()

    return text


def normalize_paint_anchor(text: str) -> str:
    """Normalize a mathematical anchor for PAINT's recall-style overlap."""
    normalized = str(text or "").lower()
    normalized = normalized.replace("\u2212", "-").replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("$", "")
    return normalized.strip(".,;:")


def extract_paint_anchors(solution: str, answer: str = "") -> list[str]:
    """Extract de-duplicated answer, formula, and numeric anchors from a reference trace."""
    candidates = []
    boxed = extract_boxed_answer(solution)
    if boxed:
        candidates.append(boxed)
    if str(answer or "").strip():
        candidates.append(str(answer))
    candidates.extend(match.group(0) for match in _PAINT_LATEX_SPAN_RE.finditer(str(solution or "")))
    candidates.extend(match.group(0) for match in _PAINT_COMMAND_RE.finditer(str(solution or "")))
    candidates.extend(match.group(0) for match in _PAINT_NUMBER_RE.finditer(str(solution or "")))

    anchors = []
    seen = set()
    for candidate in candidates:
        anchor = normalize_paint_anchor(candidate)
        if not anchor or anchor in seen:
            continue
        seen.add(anchor)
        anchors.append(anchor)
    return anchors


def compute_paint_anchor_overlap(solution: str, answer: str, rollout: str) -> tuple[float, list[str]]:
    """Return PAINT anchor recall and the anchors used to compute it."""
    anchors = extract_paint_anchors(solution, answer)
    if not anchors:
        return 0.0, []
    normalized_rollout = normalize_paint_anchor(rollout)
    matched = sum(anchor in normalized_rollout for anchor in anchors)
    return matched / len(anchors), anchors


def _answer_instruction(thinking: bool = False) -> str:
    return THINKING_BOXED_ANSWER_INSTRUCTION if thinking else BOXED_ANSWER_INSTRUCTION


def build_train_rollout_prompt(problem: str, thinking: bool = False) -> str:
    return f"Problem: {strip_legacy_math_prompt(problem)}\n\n{_answer_instruction(thinking)}"


def build_eval_rollout_prompt(problem: str, thinking: bool = False) -> str:
    return f"{strip_legacy_math_prompt(problem)}\n\n{_answer_instruction(thinking)}"


def build_rollout_prompt(problem: str, data_source: Optional[str] = None, thinking: bool = False) -> str:
    problem = strip_legacy_math_prompt(problem)
    if str(data_source or "").lower() in EVAL_DATA_SOURCES:
        return build_eval_rollout_prompt(problem, thinking=thinking)
    return build_train_rollout_prompt(problem, thinking=thinking)


def build_rollout_messages(problem: str, data_source: Optional[str] = None, thinking: Optional[bool] = None) -> list[dict]:
    # `thinking` was used to fork between an R1-Zero-style Base prompt
    # (system + <think> requirement) and the plain Instruct prompt. The Base
    # branch was retired (see module docstring on R1_ZERO_SYSTEM_PROMPT), so
    # both Base and Instruct now produce the same single-user-message prompt.
    # The argument is accepted but ignored.
    del thinking
    return [{"role": "user", "content": build_rollout_prompt(problem, data_source=data_source)}]


def normalize_privileged_text_mode(mode: str) -> str:
    normalized = str(mode or "solution_answer").strip().lower().replace("-", "_")
    if normalized in {"solution+answer", "solution_and_answer"}:
        normalized = "solution_answer"
    if normalized in {"answer", "answer_only"}:
        normalized = "answer_only"
    if normalized not in {"solution_answer", "answer_only"}:
        raise ValueError(
            f"Unsupported privileged_text_mode={mode!r}. "
            "Supported values: solution_answer (alias: solution+answer), answer_only."
        )
    return normalized


PI_CROP_MODES = {"prefix", "suffix", "random_span", "random_chars"}


def normalize_pi_crop_mode(mode: str) -> str:
    """Normalize the strategy used to retain a fixed fraction of a solution."""
    normalized = str(mode or "prefix").strip().lower().replace("-", "_")
    aliases = {
        "left": "prefix",
        "keep_left": "prefix",
        "right": "suffix",
        "keep_right": "suffix",
        "random_contiguous": "random_span",
        "random_characters": "random_chars",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PI_CROP_MODES:
        raise ValueError(
            f"Unsupported pi_crop_mode={mode!r}. Supported values: {sorted(PI_CROP_MODES)}."
        )
    return normalized


def parse_pi_teacher_specs(value: str) -> list[dict]:
    """Parse semicolon-separated ``beta:crop_mode`` teacher specifications."""
    entries = [entry.strip() for entry in str(value or "").split(";") if entry.strip()]
    if not entries:
        raise ValueError("pi_teacher_specs must contain at least one 'beta:crop_mode' entry.")

    specs = []
    seen = set()
    for teacher_idx, entry in enumerate(entries):
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pi_teacher_specs entry {entry!r}; expected 'beta:crop_mode'."
            )
        beta = float(parts[0])
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"PI teacher beta must be in [0, 1], got {beta} in {entry!r}.")
        crop_mode = normalize_pi_crop_mode(parts[1])
        key = (beta, crop_mode)
        if key in seen:
            raise ValueError(f"Duplicate PI teacher spec {entry!r} is not allowed.")
        seen.add(key)
        specs.append({"teacher_idx": teacher_idx, "pi_beta": beta, "pi_crop_mode": crop_mode})
    return specs


def _stable_pi_rng(seed: int, crop_key: str) -> random.Random:
    """Return a process-independent RNG fixed by the configured seed and problem."""
    payload = f"{int(seed)}\0{crop_key}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))


def truncate_privileged_solution(
    solution: str,
    beta: float,
    crop_mode: str = "prefix",
    crop_seed: int = 42,
    crop_key: str = "",
    random_span_anchor: Optional[float] = None,
) -> str:
    """Retain ``beta`` of the solution characters according to ``crop_mode``.

    Random modes use a local RNG derived from ``crop_seed`` and ``crop_key``;
    callers pass the problem as the key so every occurrence/rollout of the same
    problem receives the same crop without depending on process RNG state.
    """
    solution = str(solution or "").strip()
    beta = float(beta)
    crop_mode = normalize_pi_crop_mode(crop_mode)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"pi_beta must be in [0, 1], got {beta}.")
    if beta >= 1.0 or not solution:
        return solution

    keep_chars = int(len(solution) * beta)
    if keep_chars <= 0:
        return "... [reference reasoning fully hidden] ..."

    if crop_mode == "prefix":
        visible = solution[:keep_chars]
        return f"{visible}\n... [reference reasoning truncated on the right] ..."

    if crop_mode == "suffix":
        visible = solution[-keep_chars:]
        return f"... [reference reasoning truncated on the left] ...\n{visible}"

    rng = _stable_pi_rng(crop_seed, crop_key)
    if crop_mode == "random_span":
        if random_span_anchor is None:
            start = rng.randint(0, len(solution) - keep_chars)
        else:
            anchor = float(random_span_anchor)
            if not 0.0 <= anchor <= 1.0:
                raise ValueError(f"random_span_anchor must be in [0, 1], got {anchor}.")
            start = int(round(anchor * (len(solution) - keep_chars)))
        end = start + keep_chars
        parts = []
        if start > 0:
            parts.append("... [reference reasoning omitted before this span] ...")
        parts.append(solution[start:end])
        if end < len(solution):
            parts.append("... [reference reasoning omitted after this span] ...")
        return "\n".join(parts)

    # Select characters without replacement, then restore their original order.
    # This is intentionally a destructive/non-contiguous PI ablation.
    indices = sorted(rng.sample(range(len(solution)), keep_chars))
    visible = "".join(solution[index] for index in indices)
    return (
        "[Randomly sampled characters from the reference reasoning; "
        "omitted characters are not shown.]\n"
        f"{visible}"
    )


def build_teacher_privileged_text(
    answer: str,
    solution: str,
    mode: str,
    pi_beta: float = 1.0,
    pi_crop_mode: str = "prefix",
    pi_crop_seed: int = 42,
    pi_crop_key: str = "",
    pi_expose_answer: bool = True,
    pi_random_span_anchor: Optional[float] = None,
) -> tuple[str, str]:
    mode = normalize_privileged_text_mode(mode)
    answer = str(answer).strip()
    solution = str(solution or "").strip()
    if mode == "solution_answer":
        if not solution:
            raise ValueError("privileged_text_mode=solution_answer requires a non-empty GT solution")
        visible_solution = truncate_privileged_solution(
            solution=solution,
            beta=pi_beta,
            crop_mode=pi_crop_mode,
            crop_seed=pi_crop_seed,
            crop_key=pi_crop_key,
            random_span_anchor=pi_random_span_anchor,
        )
        if pi_expose_answer:
            return f"{visible_solution}\n\nCorrect final answer: {answer}", mode
        return visible_solution, mode
    if not pi_expose_answer:
        raise ValueError(
            "privileged_text_mode=answer_only cannot be combined with "
            "pi_expose_answer=False because it would produce empty privileged information."
        )
    return f"Correct final answer: {answer}", mode


def build_teacher_messages(
    problem: str, answer: str, solution: str, mode: str,
    transition_prompt: Optional[str] = None,
    framing_prompt: Optional[str] = None,
    wrapper_variant: str = "neutral",
    thinking: bool = False,
    pi_beta: float = 1.0,
    pi_crop_mode: str = "prefix",
    pi_crop_seed: int = 42,
    pi_crop_key: Optional[str] = None,
    pi_expose_answer: bool = True,
    pi_random_span_anchor: Optional[float] = None,
) -> tuple[str, list[dict], str]:
    """Build the user-side teacher message.

    `wrapper_variant` selects a named (framing, transition) pair from
    TEACHER_WRAPPER_VARIANTS.  "neutral" reproduces the original production
    wording.  `framing_prompt` and `transition_prompt` are escape hatches that
    override the variant on a per-call basis (used by the robustness probe
    script).  If both `wrapper_variant` and an override are supplied, the
    override wins.
    """
    variant = TEACHER_WRAPPER_VARIANTS[normalize_wrapper_variant(wrapper_variant)]
    if framing_prompt is None:
        framing_prompt = variant["framing"]
    if transition_prompt is None:
        transition_prompt = variant["transition"]
    problem = strip_legacy_math_prompt(problem)
    privileged_text, effective_mode = build_teacher_privileged_text(
        answer=answer,
        solution=solution,
        mode=mode,
        pi_beta=pi_beta,
        pi_crop_mode=pi_crop_mode,
        pi_crop_seed=pi_crop_seed,
        pi_crop_key=problem if pi_crop_key is None else pi_crop_key,
        pi_expose_answer=pi_expose_answer,
        pi_random_span_anchor=pi_random_span_anchor,
    )
    # `thinking` previously toggled <think>/</think> requirement on the Base
    # path; that path was retired (Qwen3-Base cannot bootstrap </think>), so
    # the teacher prompt now always uses the plain \boxed{} instruction and
    # has no system message -- identical for Base and Instruct configs.
    del thinking
    user_content = (
        f"Problem: {problem}\n\n"
        f"{framing_prompt}\n"
        "=== Reference Solution Begin ===\n"
        f"{privileged_text}\n"
        "=== Reference Solution End ==="
        f"{transition_prompt}\n"
        f"{BOXED_ANSWER_INSTRUCTION}"
    )
    messages = [{"role": "user", "content": user_content}]
    return privileged_text, messages, effective_mode


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} answer using the official OPSD logic."""
    if text is None:
        return None

    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def _normalize_fallback_string(text: str) -> str:
    return str(text).replace("$", "").replace(" ", "").lower().strip()


def _strip_answer_wrappers(text: str) -> str:
    text = str(text).strip()
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        text = boxed
    text = text.strip().strip("$").strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    return text.replace(r"\left", "").replace(r"\right", "").strip()


def _canonical_simple_number(text: str) -> Optional[Decimal]:
    text = _strip_answer_wrappers(text)
    text = _DEGREE_SUFFIX_RE.sub("", text.replace(" ", "").replace(",", "").replace("\u2212", "-"))
    if not _SIMPLE_NUMBER_RE.fullmatch(text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _simple_numeric_answers_match(predicted: str, ground_truth: str) -> Optional[bool]:
    pred_number = _canonical_simple_number(predicted)
    gt_number = _canonical_simple_number(ground_truth)
    if pred_number is None or gt_number is None:
        return None
    return pred_number == gt_number


_KK_PAIR_RE = re.compile(
    r"([A-Za-z][A-Za-z\-']*)\s*(?:=|:|is(?:\s+(?:a|an))?)\s*(knight|knave)",
    re.IGNORECASE,
)


def parse_kk_assignment(text: Optional[str]) -> dict[str, str]:
    """Parse a K&K assignment string into a {name_lower: 'knight'|'knave'} dict.

    Accepts the natural-language form ("Alice is a knight, Bob is a knave"),
    equals form ("Alice=knight"), or colon form. Strips a leading
    `\\boxed{...}` wrapper if present. First-occurrence wins per name.
    """
    if text is None:
        return {}
    text = str(text).strip()
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        text = boxed
    result: dict[str, str] = {}
    for name, role in _KK_PAIR_RE.findall(text):
        key = name.lower()
        if key not in result:
            result[key] = role.lower()
    return result


def grade_kk_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """Compare predicted vs ground-truth K&K assignments (order-insensitive)."""
    gt = parse_kk_assignment(ground_truth)
    if not gt:
        return False
    pred = parse_kk_assignment(predicted)
    return pred == gt


def grade_boxed_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """Grade a boxed answer using official OPSD math_verify logic with string fallback."""
    if predicted is None:
        return False

    numeric_match = _simple_numeric_answers_match(predicted, ground_truth)
    if numeric_match is not None:
        return numeric_match

    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ModuleNotFoundError(
            "math_verify is required for OPSD-aligned grading. Install `math-verify` in the training env."
        ) from exc

    try:
        pred_text = predicted if "$" in predicted else f"${predicted}$"
        gt_text = ground_truth if "$" in str(ground_truth) else f"${ground_truth}$"
        pred_parsed = parse(pred_text, fallback_mode="no_fallback")
        gt_parsed = parse(gt_text, fallback_mode="no_fallback")
        try:
            return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
        except Exception:
            return bool(verify(gt_parsed, pred_parsed, timeout_seconds=None))
    except Exception:
        return _normalize_fallback_string(predicted) == _normalize_fallback_string(ground_truth)
