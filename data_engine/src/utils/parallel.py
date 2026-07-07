"""Multiprocessing utilities."""

from multiprocessing import Pool
from tqdm import tqdm
from typing import Callable, TypeVar

T = TypeVar('T')
R = TypeVar('R')


def parallel_map(func: Callable[[T], R], items: list[T],
                 num_workers: int = 10, desc: str = "") -> list[R]:
    """Map a function across items using multiprocessing with tqdm progress."""
    if num_workers <= 1:
        return [func(item) for item in tqdm(items, desc=desc)]

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(func, items), total=len(items), desc=desc))
    return results
