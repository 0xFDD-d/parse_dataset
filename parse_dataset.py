#!/usr/bin/env python3
import sys
import argparse
import pathlib
from typing import Iterable, List
import numpy as np


def bytes_to_mb(value: int):
    return round(value / 1024 / 1024, 2)


def read_dataset(file: pathlib.Path, delimiter=" ") -> Iterable[List[str]]:
    with open(file) as fh:
        for line in fh:
            yield line.strip().split(delimiter)


def numpy_vectorize(file: pathlib.Path) -> np.ndarray:
    """
    Iterates over the file, returns the ndarray
    """
    # Preallocate arrays (memory efficient)
    # since for np.append() python needs to make room in the memory again and again for each append
    entries = 1_000_000  # 1 million entries
    hex_arr: np.array = np.zeros(entries, dtype='<U65')
    float_arr: np.array = np.zeros(entries, dtype='<f8')
    arr = np.stack((hex_arr, float_arr), axis=-1)

    print(f"size of pre-allocated array {bytes_to_mb(arr.nbytes)}mb")

    idx = 0
    for row in read_dataset(file):
        n_rows = int(row[3])
        _hex, _floats = np.array(row[n_rows * -2:]).reshape(2, n_rows)

        for i in range(0, _hex.size):
            arr[idx] = _floats[i], _hex[i]
            idx += 1

    return arr[:idx]


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

    print(f"size of array {bytes_to_mb(arr.nbytes)}mb")

    result_dict = {}
    for idx, r in enumerate(arr):
        result_dict[r[0]] = (idx, r[1])

    print(result_dict)

    print(f"size of result dictionary {bytes_to_mb(sys.getsizeof(result_dict))}mb")


if __name__ == "__main__":
    main()
