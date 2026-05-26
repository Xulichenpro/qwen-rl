import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_answer_reward_accepts_equivalent_fraction_decimal_and_rejects_wrong_answer() -> None:
    from src.rl.grpo.rewards import answer_reward

    completions = [
        "<think>\n1. 半数是1/2。\n</think>\n<answer>\n1/2\n</answer>",
        "<think>\n1. 半数是0.5。\n</think>\n<answer>\n0.5\n</answer>",
        "<think>\n1. 算错了。\n</think>\n<answer>\n2\n</answer>",
    ]

    rewards = answer_reward(completions=completions, gold_answer=["0.5", "1/2", "3"])

    assert rewards == [1.0, 1.0, 0.0]


def test_format_reward_requires_single_think_and_answer_with_bare_numeric_answer() -> None:
    from src.rl.grpo.rewards import format_reward

    completions = [
        "<think>\n1. 105 × 3 = 315。\n</think>\n<answer>\n315\n</answer>",
        "<think>1. 105 × 3 = 315。</think><answer>315千克</answer>",
        "答案是315",
    ]

    assert format_reward(completions=completions) == [1.0, 0.0, 0.0]


if __name__ == "__main__":
    test_answer_reward_accepts_equivalent_fraction_decimal_and_rejects_wrong_answer()
    test_format_reward_requires_single_think_and_answer_with_bare_numeric_answer()
