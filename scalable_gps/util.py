from typing import List, Tuple

import numpy as np


def _to_list(a):
    return [a]


def _append(a, b):
    a.append(b)
    return a


def _extend(a, b):
    a.extend(b)
    return a


def _sort_row(row: List[Tuple[int, float]]) -> List[float]:
    row_sorted = sorted(row, key=lambda r: r[0])
    row_sorted = [r[1] for r in row_sorted]
    return row_sorted
