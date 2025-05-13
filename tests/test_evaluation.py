import pandas as pd
import pytest

from dragon_eval.evaluation import (TASK_TYPE, EvalType, score_multi_label_f1,
                                    score_rsmape, select_entity_labels)


@pytest.mark.parametrize(
    "y_true, y_pred, epsilon, expected_score",
    [
        ([1.0, 2.0, 3.0], [1.1, 2.0, 2.9], 0.1, 0.9792929292929293),
        ([1.0, 0.0, 4.0], [1.0, 0.0, 3.9], 0.01, 0.992),
        ([0.0, 0.0, 0.0], [0.1, 0.2, 0.3], 0.001, 0.00606612453777744),  # example to handle division by zero
        ([0.0, 0.0, 0.0], [0.2, 0.3, 0.4], 0.001, 0.0035970497001190926),  # example to handle division by zero
        ([[1.0, 2.0, 3.0], [4.0, 5.0, None], [6.0, None, None]], [[7.0, None, None], [8.0, None, None], [9.0, 10.0, None]], [11.0, 12.0, 13.0], 0.7950730129228245),  # example with multiple epsilons
    ],
)
def test_score_rsmape(y_true, y_pred, epsilon, expected_score):
    score = score_rsmape(y_true=y_true, y_pred=y_pred, epsilon=epsilon, ignore_missing_targets=True)
    assert pytest.approx(score, rel=1e-2) == expected_score


@pytest.mark.parametrize(
    "labels, entity_lbl, expected_labels",
    [
        (
            [["B-FOO", "B-BAR"], ["B-BAZ"], ["B-BAR", "B-BAZ"], ["O"]],
            "FOO",
            ["B-FOO", "O", "O", "O"],
        ),
    ],
)
def test_select_entity_labels(labels, entity_lbl, expected_labels):
    result = select_entity_labels(labels, entity_lbl)
    assert result == expected_labels


@pytest.mark.parametrize(
    "y_true, y_pred, average, expected_score",
    [
        (
            pd.Series([[["B-FOO", "B-BAR"], ["I-FOO"], ["O"]]]),
            pd.Series([[["B-FOO", "B-BAR"], ["I-FOO"], ["O"]]]),
            "weighted",
            1.0,
        ),
        (
            pd.Series([[["B-FOO"], ["I-FOO", "B-BAR"], ["I-FOO"]]]),
            pd.Series([[["B-FOO"], ["I-FOO"], ["I-FOO"]]]),
            "weighted",
            0.5,
        ),
    ],
)
def test_score_multi_label_f1(y_true, y_pred, average, expected_score):
    score = score_multi_label_f1(y_true=y_true, y_pred=y_pred, average=average)
    assert pytest.approx(score, rel=1e-2) == expected_score


@pytest.mark.parametrize(
    "task_name, expected_eval_type",
    [
        ("Task101_Example_sl_bin_clf", EvalType.BINARY_CLASSIFICATION),
        ("Task106_Example_sl_reg", EvalType.REGRESSION),
        ("Task108_Example_sl_ner", EvalType.SINGLE_LABEL_NER),
    ],
)
def test_task_type(task_name, expected_eval_type):
    assert TASK_TYPE[task_name] == expected_eval_type


if __name__ == "__main__":
    pytest.main()
