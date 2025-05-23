# DRAGON Evaluation

Evaluation method for the DRAGON (Diagnostic Report Analysis: General Optimization of NLP) challenge. 

If you are using DRAGON resources, please cite the following article:

> J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, and H. Huisman on behalf of the DRAGON consortium. The DRAGON benchmark for clinical NLP. *npj Digital Medicine* 8, 289 (2025). [https://doi.org/10.1038/s41746-025-01626-x](https://doi.org/10.1038/s41746-025-01626-x)

Download the citation file for your reference manager: [BibTeX](https://github.com/DIAGNijmegen/dragon/blob/main/citation.bib) | [RIS](https://github.com/DIAGNijmegen/dragon/blob/main/citation.ris)


## Installation
A pre-built Docker container with the DRAGON evaluation method is available:

```
docker pull joeranbosma/dragon_eval:latest
```

The DRAGON evaluation method can be pip-installed:

```
pip install dragon_eval
```

Or, alternatively, it can be installed from source:

```
pip install git+https://github.com/DIAGNijmegen/dragon_eval
```

The evaluation method was tested with Python 3.10. See [requirements.txt](requirements.txt) for a full list of exact package versions.

## Usage
The Docker container can be used to evaluate the synthetic datasets as specified in [evaluate.sh](evaluate.sh). To evaluate the synthetic tasks, place the predictions to evaluate in the `test-predictions` folder and run `./evaluate.sh`.

The DRAGON evaluation method can also be used from the command line (if installed with pip):

```
python -m dragon_eval --ground-truth-path="ground-truth" --predictions-path=test-predictions --output-file=metrics.json --folds 0 1 2 3 4 --tasks 000 001 002 003 004 005 006 007
```

The command above should work when executed from the `dragon_eval` folder, which needs to be cloned locally for the ground truth and prediction files to be present. Change the paths above when executing the command from a different place or storing the files in a different place. The tasks and folds to evaluate can be changed with the respective parameters.

## Managed By
Diagnostic Image Analysis Group, Radboud University Medical Center, Nijmegen, The Netherlands

## Contact Information
Joeran Bosma: Joeran.Bosma@radboudumc.nl
