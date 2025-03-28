# Adapted from evalutils version 0.4.2
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from pandas import read_csv
from pandas.errors import EmptyDataError, ParserError

from dragon_eval.evalutils.exceptions import FileLoaderError

logger = logging.getLogger(__name__)


def get_first_int_in(s: str) -> int:
    """
    Gets the first integer in a string.

    Parameters
    ----------
    s
        The string to search for an int

    Returns
    -------
        The first integer found in the string

    Raises
    ------
    AttributeError
        If there is not an int contained in the string

    """
    r = re.compile(r"\D*((?:\d+\.?)+)\D*")
    m = r.search(s)

    if m is not None:
        return int(m.group(1).replace(".", ""))
    else:
        raise AttributeError(f"No int found in {s}")


def first_int_in_filename_key(fname: Path) -> str:
    try:
        return f"{get_first_int_in(fname.stem):>64}"
    except AttributeError:
        logger.warning(f"Could not find an int in the string {fname.stem!r}.")
        return fname.stem


class FileLoader(ABC):
    @abstractmethod
    def load(self, *, fname: Path) -> List[Dict]:
        """
        Tries to load the file given by the path fname.

        Notes
        -----
        For this to work with the validators you must:

            If you load an image it must save the hash in the `hash` column

            If you reference a Path it must be saved in the `path` column


        Parameters
        ----------
        fname
            The file that the loader will try to load

        Returns
        -------
            A list containing all of the cases in this file

        Raises
        ------
        FileLoaderError
            If a file cannot be loaded as the specified type

        """
        raise FileLoaderError


class CSVLoader(FileLoader):
    def load(self, *, fname: Path):
        try:
            return read_csv(
                fname, skipinitialspace=True, encoding="utf-8"
            ).to_dict(orient="records")
        except UnicodeDecodeError:
            raise FileLoaderError(f"Could not load {fname} using {__name__}.")
        except (ParserError, EmptyDataError):
            raise ValueError(
                f"CSV file could not be loaded: we could not load "
                f"{fname.name} using `pandas.read_csv`."
            )
