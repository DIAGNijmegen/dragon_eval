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

import argparse
from pathlib import Path

from dragon_eval import DragonEval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--ground-truth-path", type=Path, default=Path("/opt/app/ground-truth/"),
                        help="Path to the ground truth directory.")
    parser.add_argument("-i", "--predictions-path", type=Path, default=Path("/input/"),
                        help="Path to the predictions directory.")
    parser.add_argument("-o", "--output-file", type=Path, default=Path("/output/metrics.json"),
                        help="Path to the output file.")
    parser.add_argument("-f", "--folds", nargs="+", type=int, default=(0, 1, 2, 3, 4),
                        help="Folds to evaluate. Default: all folds.")
    parser.add_argument("-t", "--tasks", nargs="+", type=str, default=None,
                        help="Tasks to evaluate. Default: all tasks."),
    args = parser.parse_args()

    DragonEval(
        ground_truth_path=Path(args.ground_truth_path),
        predictions_path=Path(args.predictions_path),
        output_file=Path(args.output_file),
        folds=args.folds,
        tasks=args.tasks,
    ).evaluate()
