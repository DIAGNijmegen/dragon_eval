# Adapted from evalutils version 0.4.2
import json
import logging
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple, Union

from pandas import DataFrame, Series, concat, merge

from dragon_eval.evalutils.exceptions import (ConfigurationError,
                                              FileLoaderError, ValidationError)
from dragon_eval.evalutils.io import (CSVLoader, FileLoader,
                                      first_int_in_filename_key)
from dragon_eval.evalutils.validators import DataFrameValidator

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("/input/")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/output/results.json")
DEFAULT_GROUND_TRUTH_PATH = Path("/opt/app/ground-truth/")
DEFAULT_EVALUATION_OUTPUT_FILE_PATH = Path("/output/metrics.json")


class BaseEvaluation(ABC):
    def __init__(
        self,
        *,
        ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
        predictions_path: Path = DEFAULT_INPUT_PATH,
        file_sorter_key: Callable = first_int_in_filename_key,
        file_loader: FileLoader,
        validators: Tuple[DataFrameValidator, ...],
        join_key: Optional[str] = None,
        aggregates: Optional[Set[str]] = None,
        output_file: PathLike = DEFAULT_EVALUATION_OUTPUT_FILE_PATH,
    ):
        """
        The base class for all evaluations. Sets the environment and controls
        the flow of the evaluation once `evaluate` is called.


        Parameters
        ----------
        ground_truth_path
            The path in the container where the ground truth will be loaded
            from
        predictions_path
            The path in the container where the submission will be loaded from
        file_sorter_key
            A function that determines how files are sorted and matched
            together
        file_loader
            The loader that will be used to get all files
        validators
            A tuple containing all the validators that will be used on the
            loaded data
        join_key
            The column that will be used to join the predictions and ground
            truth tables
        aggregates
            The set of aggregates that will be calculated by
            `pandas.DataFrame.describe`
        output_file
            The path to the location where the results will be written
        """
        if aggregates is None:
            aggregates = {
                "mean",
                "std",
                "min",
                "max",
                "25%",
                "50%",
                "75%",
                "count",
                "uniq",
                "freq",
            }

        self._ground_truth_path = ground_truth_path
        self._predictions_path = predictions_path
        self._file_sorter_key = file_sorter_key
        self._file_loader = file_loader
        self._validators = validators
        self._join_key = join_key
        self._aggregates = aggregates
        self._output_file = output_file

        self._ground_truth_cases = DataFrame()
        self._predictions_cases = DataFrame()

        self._cases = DataFrame()

        self._case_results = DataFrame()
        self._aggregate_results: Dict[str, Union[float, int, str, None]] = {}

        super().__init__()

        if isinstance(self._file_loader, CSVLoader) and self._join_key is None:
            raise ConfigurationError(
                f"You must set a `join_key` when using {self._file_loader}."
            )

    @property
    def _metrics(self) -> Dict:
        """Returns the calculated case and aggregate results"""
        return {
            "case": self._case_results.to_dict(),
            "aggregates": self._aggregate_results,
        }

    def evaluate(self):
        self.load()
        self.validate()
        self.merge_ground_truth_and_predictions()
        self.cross_validate()
        self.score()
        self.save()

    def load(self):
        self._ground_truth_cases = self._load_cases(
            folder=self._ground_truth_path
        )
        self._predictions_cases = self._load_cases(
            folder=self._predictions_path
        )

    def _load_cases(self, *, folder: Path) -> DataFrame:
        cases = None

        for f in sorted(folder.glob("**/*"), key=self._file_sorter_key):
            try:
                new_cases = self._file_loader.load(fname=f)
            except FileLoaderError:
                logger.warning(
                    f"Could not load {f.name} using {self._file_loader}."
                )
            else:
                if cases is None:
                    cases = new_cases
                else:
                    cases += new_cases

        if cases is None:
            raise FileLoaderError(
                f"Could not load any files in {folder} with "
                f"{self._file_loader}."
            )

        return DataFrame(cases)

    def validate(self):
        """Validates each dataframe separately"""
        self._validate_data_frame(df=self._ground_truth_cases)
        self._validate_data_frame(df=self._predictions_cases)

    def _validate_data_frame(self, *, df: DataFrame):
        for validator in self._validators:
            validator.validate(df=df)

    @abstractmethod
    def merge_ground_truth_and_predictions(self):
        pass

    @abstractmethod
    def cross_validate(self):
        """Validates both dataframes"""
        pass

    def _raise_missing_predictions_error(self, *, missing=None):
        if missing is not None:
            message = (
                "Predictions missing: you did not submit predictions for "
                f"{missing}. Please try again."
            )
        else:
            message = (
                "Predictions missing: you did not submit enough predictions, "
                "please try again."
            )

        raise ValidationError(message)

    def _raise_extra_predictions_error(self, *, extra=None):
        if extra is not None:
            message = (
                "Too many predictions: we do not have the ground truth data "
                f"for {extra}. Please try again."
            )
        else:
            message = (
                "Too many predictions: you submitted too many predictions, "
                "please try again."
            )

        raise ValidationError(message)

    @abstractmethod
    def score(self):
        pass

    # noinspection PyUnusedLocal
    def score_case(self, *, idx: int, case: DataFrame) -> Dict:
        return {}

    def score_aggregates(self) -> Dict:
        aggregate_results = {}

        for col in self._case_results.columns:
            aggregate_results[col] = self.aggregate_series(
                series=self._case_results[col]
            )

        return aggregate_results

    def aggregate_series(self, *, series: Series) -> Dict:
        summary = series.describe()
        valid_keys = [a for a in self._aggregates if a in summary]

        series_summary = {}

        for k in valid_keys:
            value = summary[k]

            # % in keys could cause problems when looking up values later
            key = k.replace("%", "pc")

            try:
                json.dumps(value)
            except TypeError:
                logger.warning(
                    f"Could not serialize {key}: {value} as json, "
                    f"so converting {value} to int."
                )
                value = int(value)

            series_summary[key] = value

        return series_summary

    def save(self):
        with open(self._output_file, "w") as f:
            f.write(json.dumps(self._metrics))


class ClassificationEvaluation(BaseEvaluation):
    """
    ClassificationEvaluations have the same number of predictions as the
    number of ground truth cases. These can be things like, what is the
    stage of this case, or segment some things in this case.
    """

    def merge_ground_truth_and_predictions(self):
        if self._join_key:
            kwargs = {"on": self._join_key}
        else:
            kwargs = {"left_index": True, "right_index": True}

        print("Merging ground truth and predictions")
        print(self._ground_truth_cases)
        print(self._predictions_cases)
        self._cases = merge(
            left=self._ground_truth_cases,
            right=self._predictions_cases,
            indicator=True,
            how="outer",
            suffixes=("_ground_truth", "_prediction"),
            **kwargs,
        )

    def cross_validate(self):
        missing = [
            p for _, p in self._cases.iterrows() if p["_merge"] == "left_only"
        ]

        if missing:
            if self._join_key:
                missing = [p[self._join_key] for p in missing]
            self._raise_missing_predictions_error(missing=missing)

        extra = [
            p for _, p in self._cases.iterrows() if p["_merge"] == "right_only"
        ]

        if extra:
            if self._join_key:
                extra = [p[self._join_key] for p in extra]
            self._raise_extra_predictions_error(extra=extra)

    def score(self):
        self._case_results = DataFrame()
        for idx, case in self._cases.iterrows():
            self._case_results = concat(
                [
                    self._case_results,
                    DataFrame.from_records(
                        [self.score_case(idx=idx, case=case)]
                    ),
                ],
                ignore_index=True,
            )
        self._aggregate_results = self.score_aggregates()
