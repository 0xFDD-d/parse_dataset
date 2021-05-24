#!/usr/bin/env python3
import argparse
import pathlib
from typing import Iterable, List
import numpy as np


def read_dataset(file: pathlib.Path, delimiter=" ") -> Iterable[List[str]]:
    with open(file) as fh:
        for line in fh:
            yield line.strip("\n").split(delimiter)


def numpy_vectorize(file: pathlib.Path) -> np.ndarray:
    """
    Iterates over the file line-by-line, returns the joined sequence of two flattened arr
    """
    arr_hex: np.array = np.array([], dtype='<U65')
    arr_floats: np.array = np.array([], dtype='<f8')

    for row in read_dataset(file):
        n_rows = int(row[3])
        _floats, _hex = np.array(row[n_rows * -2:]).reshape(2, n_rows)
        arr_hex = np.append(arr_hex, _hex)
        arr_floats = np.append(arr_floats, _floats)

    return np.stack((arr_hex, arr_floats), axis=-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", "-f", type=pathlib.Path, required=True, help="input filepath to a data set file."
    )
    args = parser.parse_args()
    if not args:
        parser.print_help()
        parser.exit()
    return args


def main():
    args = parse_args()
    arr = numpy_vectorize(file=args.file)

    result_dict = {}
    for idx, r in enumerate(arr):
        result_dict[r[0]] = (idx, r[1])

    print(result_dict)


if __name__ == "__main__":
    main()
