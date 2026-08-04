"""Optional progress reporting shared by inference stages.

Keeping this utility in ``runtime`` prevents data loading, recognition, and detection
modules from each importing or implementing their own progress-bar wrapper.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

Item = TypeVar("Item")


def iter_with_progress(
    items: Iterable[Item], *, description: str, enabled: bool
) -> Iterator[Item]:
    """Yield ``items`` and wrap them in tqdm only when progress is requested."""
    if enabled:
        try:
            from tqdm import tqdm
        except ModuleNotFoundError:
            pass
        else:
            yield from tqdm(items, desc=description)
            return
    yield from items