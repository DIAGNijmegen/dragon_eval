#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import seqeval.metrics
from evalutils.evalutils import (DEFAULT_EVALUATION_OUTPUT_FILE_PATH,
                                 DEFAULT_GROUND_TRUTH_PATH, DEFAULT_INPUT_PATH,
                                 ClassificationEvaluation)
from evalutils.io import FileLoader
from evalutils.validators import ExpectedColumnNamesValidator
from sklearn.metrics import cohen_kappa_score, roc_auc_score


class EvalType(Enum):
    """Problem type of the task"""
    SINGLE_LABEL_NER = "single-label named entity recognition (macro F1)"
    MULTI_LABEL_NER = "multi-label named entity recognition (weighted F1)"
    REGRESSION = "regression (R-SMAPE)"
    BINARY_CLASSIFICATION = "binary classification (AUC)"
    BINARY_CLASSIFICATION_NON_SHARED_TASK = "binary classification different objective across labels (Unweighted Cohen's kappa)"
    ORDINAL_MULTI_CLASS_CLASSIFICATION = "ordinal multi-class classification (Linear Cohen's kappa)"
    NONORDINAL_MULTI_CLASS_CLASSIFICATION = "non-ordinal multi-class classification (Unweighted Cohen's kappa)"


TASK_TYPE = {
    # example tasks
    "Task101_Example_sl_bin_clf": EvalType.BINARY_CLASSIFICATION,
    "Task102_Example_sl_mc_clf": EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task103_Example_mednli": EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task104_Example_ml_bin_clf": EvalType.BINARY_CLASSIFICATION,
    "Task105_Example_ml_mc_clf": EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task106_Example_sl_reg": EvalType.REGRESSION,
    "Task107_Example_ml_reg": EvalType.REGRESSION,
    "Task108_Example_sl_ner": EvalType.SINGLE_LABEL_NER,
    "Task109_Example_ml_ner": EvalType.MULTI_LABEL_NER,

    # tasks from the DRAGON benchmark
    "Task001_adhesion_clf": EvalType.BINARY_CLASSIFICATION,
    "Task002_nodule_clf": EvalType.BINARY_CLASSIFICATION,
    "Task003_kidney_clf": EvalType.BINARY_CLASSIFICATION,
    "Task004_skin_case_selection_clf": EvalType.BINARY_CLASSIFICATION,
    "Task005_recist_timeline_clf": EvalType.BINARY_CLASSIFICATION,
    "Task006_pathology_tumor_origin_clf": EvalType.BINARY_CLASSIFICATION,
    "Task007_nodule_diameter_presence_clf": EvalType.BINARY_CLASSIFICATION,
    "Task008_pdac_size_presence_clf": EvalType.BINARY_CLASSIFICATION,
    "Task009_pdac_diagnosis_clf": EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task010_prostate_radiology_clf": EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task011_prostate_pathology_clf": EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task012_pathology_tissue_type_clf": EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task013_pathology_tissue_origin_clf": EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task014_textual_entailment_clf": EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task015_colon_pathology_clf": EvalType.BINARY_CLASSIFICATION_NON_SHARED_TASK,
    "Task016_recist_lesion_size_presence_clf": EvalType.BINARY_CLASSIFICATION,
    "Task017_pdac_attributes_clf": EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task018_osteoarthritis_clf": EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION,
    "Task019_prostate_volume_reg": EvalType.REGRESSION,
    "Task020_psa_reg": EvalType.REGRESSION,
    "Task021_psad_reg": EvalType.REGRESSION,
    "Task022_pdac_size_reg": EvalType.REGRESSION,
    "Task023_nodule_diameter_reg": EvalType.REGRESSION,
    "Task024_recist_lesion_size_reg": EvalType.REGRESSION,
    "Task025_anonymisation_ner": EvalType.SINGLE_LABEL_NER,
    "Task026_medical_terminology_ner": EvalType.SINGLE_LABEL_NER,
    "Task027_prostate_biopsy_ner": EvalType.MULTI_LABEL_NER,
    "Task028_skin_pathology_ner": EvalType.MULTI_LABEL_NER,
}


class JSONLoader(FileLoader):
    """
    Custom file loader for JSON files.
    """

    def load(self, fname: Path) -> pd.DataFrame:
        if fname.is_dir():
            # skip directories
            return None

        with open(fname) as fp:
            return pd.read_json(fp, dtype={"uid": str})


def score_rsmape(
    *, y_true, y_pred, epsilon: float, ignore_missing_targets: bool = False,
) -> float:
    """Robust symmetric mean absolute percentage score (R-SMAPE)
    The R-SMAPE is a robust version of the symmetric mean absolute percentage error (SMAPE) by adding epsilon to the denominator.
    SMAPE is a symmetric version of the mean absolute percentage error (MAPE) by adding the absolute value of the predicted values to the denominator.
    This results in a score that is more robust to outliers and makes sure that swapping the true and predicted values does not change the score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # flatten arrays and maybe ignore missing targets
    if ignore_missing_targets:
        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    else:
        y_pred = np.ravel(y_pred)
        y_true = np.ravel(y_true)

    # compute R-SMAPE
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    rsmape = numerator / denominator
    return float(1 - np.mean(rsmape))


def select_entity_labels(labels: List[List[str]], entity_lbl: str) -> List[str]:
    labels = [[lbl for lbl in token_labels if entity_lbl in lbl] for token_labels in labels]
    return [(token_labels[0] if len(token_labels) > 0 else "O") for token_labels in labels]


def score_multi_label_f1(
    *, y_true: pd.Series, y_pred: pd.Series, average: str = "weighted", out_label: str = "O",
) -> float:
    """Multi-label F1 score"""
    label_values = sorted(set([re.sub(r"^[BI]-", "", lbl) for lbl in y_true.explode().explode().unique() if lbl != out_label]))

    per_lbl_score, per_lbl_support = [], []
    for entity_lbl in label_values:
        # select labels for the current entity
        y_true_lbl = y_true.apply(lambda labels: select_entity_labels(labels, entity_lbl=entity_lbl))
        y_pred_lbl = y_pred.apply(lambda labels: select_entity_labels(labels, entity_lbl=entity_lbl))
        support = len(seqeval.metrics.sequence_labeling.get_entities(y_true_lbl))

        # calculate F1 score
        score = seqeval.metrics.f1_score(
            y_true=y_true_lbl,
            y_pred=y_pred_lbl,
            average=average,
        )
        per_lbl_score.append(score)
        per_lbl_support.append(support)

    # calculate average of F1 scores
    if average == "macro":
        score = np.mean(per_lbl_score)
    elif average == "weighted":
        score = np.average(per_lbl_score, weights=per_lbl_support)
    else:
        raise ValueError(f"Unsupported average: {average}")

    return float(score)


class DragonEval(ClassificationEvaluation):
    def __init__(self, folds: Iterable[int] = range(5), tasks: Optional[Iterable[str]] = None, **kwargs):
        super().__init__(
            file_loader=JSONLoader(),
            validators=(
                ExpectedColumnNamesValidator(
                    expected=("uid", ), extra_cols_check=False,
                ),
            ),
            join_key="uid",
            **kwargs,
        )
        self._scores: Dict[str, Dict[str, float]] = {}
        self.folds = folds

        if tasks is None:
            # get all tasks
            self.tasks: list[str] = sorted([
                path.stem
                for path in self._ground_truth_path.glob("*.json")
            ])
            if not self.tasks:
                raise ValueError("Could not find any tasks!")
        else:
            # collect task names (possibly using partial names)
            task_names = []
            for task in tasks:
                if (self._ground_truth_path / f"{task}.json").exists():
                    task_names.append(task)
                    continue

                files_found = [path.stem for path in self._ground_truth_path.glob(f"*{task}*.json")]
                if not files_found:
                    raise ValueError(f"Could not find task: {task}")

                if len(files_found) > 1:
                    raise ValueError(f"Found multiple tasks matching {task}: {files_found}")

                task_names.append(files_found[0])
            if len(set(task_names)) != len(list(tasks)):
                raise ValueError(f"Duplicate tasks found: {task_names}")
            self.tasks = task_names

        print(f"Evaluating {len(self.tasks)} tasks: {self.tasks}")

    def evaluate(self):
        for task_name in self.tasks:
            for fold in self.folds:
                job_name = f"{task_name}-fold{fold}"
                self.load(task_name=task_name, job_name=job_name)
                self.validate()
                self.merge_ground_truth_and_predictions()
                self.cross_validate()
                self.score(task_name=task_name, job_name=job_name)
        self.aggregate_scores()
        self.save()

    def load(self, *, task_name: str, job_name: str):
        """Loads ground truth and predictions for a given job name"""
        self._ground_truth_cases = self._file_loader.load(
            self._ground_truth_path / f"{task_name}.json"
        )
        self._predictions_cases = self._file_loader.load(
            self._predictions_path / job_name / "nlp-predictions-dataset.json"
        )

    def score(self, *, task_name: str, job_name: str):
        """Scores the predictions for a given task / job

        Args:
            task_name: Name of the task
            job_name: Name of the job (task_name-foldX)
        """
        print(f"Evaluating {job_name}")
        # select ground truth and prediction columns
        label_column = [col for col in self._cases.columns if col.endswith("_target")][0]
        prediction_column = label_column.replace("_target", "")
        if prediction_column not in self._cases.columns:
            raise ValueError(f"Could not find prediction column for {label_column} (job: {job_name})")

        y_true = self._cases[label_column]
        y_pred = self._cases[prediction_column]

        if TASK_TYPE[task_name] == EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION:
            # evaluate ordinal multi-class classification tasks
            # metric: Linear-weighted Cohen's kappa
            score = cohen_kappa_score(
                y1=y_true,
                y2=y_pred,
                weights="linear",
            )

        elif TASK_TYPE[task_name] == EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION:
            # evaluate non-ordinal (multi-class) classification tasks
            # note: each subtask is the same, so we pool the labels and predictions
            #       (this is not actually true for the example task, but it is for the real tasks)
            # metric: Unweighted Cohen's kappa
            score = cohen_kappa_score(
                y1=y_true.explode(),
                y2=y_pred.explode(),
                weights=None,
            )

        elif TASK_TYPE[task_name] == EvalType.BINARY_CLASSIFICATION:
            # evaluate (multi-label) binary classification tasks
            # note: each subtask is the same, so we pool the labels and predictions
            # metric: AUC
            score = roc_auc_score(
                y_true=y_true.explode().explode().values.astype(int),
                y_score=y_pred.explode().explode().values.astype(float),
            )

        elif TASK_TYPE[task_name] == EvalType.BINARY_CLASSIFICATION_NON_SHARED_TASK:
            # evaluate binary classification tasks with different objectives across labels
            # metric: mean AUC per objective
            score = np.mean([
                roc_auc_score(
                    y_true=y_true.apply(lambda values: values[i]),
                    y_score=y_pred.apply(lambda values: values[i]),
                )
                for i in range(len(y_true.iloc[0]))
            ])

        elif TASK_TYPE[task_name] == EvalType.REGRESSION:
            # evaluate regression tasks
            # note: for the multi-label regression task, each subtask is the same,
            #       so we pool the labels and predictions
            # metric: R-SMAPE
            epsilon = {
                # example tasks
                "Task106_Example_sl_reg": 4,
                "Task107_Example_ml_reg": 4,

                # DRAGON benchmark tasks
                "Task019_prostate_volume_reg": 4,
                "Task020_psa_reg": 0.4,
                "Task021_psad_reg": 0.04,
                "Task022_pdac_size_reg": 4,
                "Task023_nodule_diameter_reg": 4,
                "Task024_recist_lesion_size_reg": 4,
            }[task_name]

            score = score_rsmape(
                y_true=y_true.explode().astype(float),
                y_pred=y_pred.explode().astype(float),
                epsilon=epsilon,
                ignore_missing_targets=True,
            )

        elif TASK_TYPE[task_name] == EvalType.SINGLE_LABEL_NER:
            # evaluate single-label named entity recognition tasks
            # metric: F1 score
            score = seqeval.metrics.f1_score(
                y_true=y_true,
                y_pred=y_pred,
                average="macro",
            )

        elif TASK_TYPE[task_name] == EvalType.MULTI_LABEL_NER:
            # evaluate multi-label named entity recognition tasks
            # metric: weighted F1 score
            score = score_multi_label_f1(
                y_true=y_true,
                y_pred=y_pred,
                average="weighted",
            )

        else:
            raise ValueError(f"Unexpexted task: {task_name}")

        # save score for the current job
        if task_name not in self._scores:
            self._scores[task_name] = {}
        self._scores[task_name][job_name] = score

    @property
    def _metrics(self) -> Dict[str, Any]:
        """Returns the calculated case and aggregate results"""
        return {
            "case": self._scores,
            "aggregates": self._aggregate_results,
            "version": "0.2.6",
        }

    @staticmethod
    def calculate_aggregate_results(scores):
        """Calculates the mean and std of the scores"""
        # calculate mean and std for each task
        aggregate_results = {}
        for task_name, scores in scores.items():
            aggregate_results[task_name] = {
                "mean": np.mean(list(scores.values())),
                "std": np.std(list(scores.values())),
            }

        return aggregate_results

    def aggregate_scores(self):
        """Aggregates the scores"""
        # calculate mean and std for each task
        self._aggregate_results = self.calculate_aggregate_results(self._scores)

        # calculate overall score
        self._aggregate_results["overall"] = {
            "mean": np.mean([score["mean"] for score in self._aggregate_results.values()]),
            "std": np.mean([score["std"] for score in self._aggregate_results.values()]),
        }

        print("Aggregate results:")
        for task_name, scores in self._aggregate_results.items():
            print(f"  {task_name}: {scores['mean']:.3f} ± {scores['std']:.3f}")


if __name__ == "__main__":
    DragonEval(
        ground_truth_path=DEFAULT_GROUND_TRUTH_PATH if DEFAULT_GROUND_TRUTH_PATH.exists() else Path("ground-truth"),
        predictions_path=DEFAULT_INPUT_PATH if DEFAULT_INPUT_PATH.exists() else Path("test-predictions"),
        output_file=DEFAULT_EVALUATION_OUTPUT_FILE_PATH if DEFAULT_EVALUATION_OUTPUT_FILE_PATH.parent.exists() else Path("test-output/metrics.json"),
    ).evaluate()
