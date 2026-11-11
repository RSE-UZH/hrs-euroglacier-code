import copy
import logging
from typing import Any

logger = logging.getLogger("spot5")


def recursive_to_dotdict(x: Any) -> Any:
    """
    Recursively convert nested dictionaries (and lists) to DotDict instances.

    If x is already a DotDict, it is returned as-is.
    """
    if isinstance(x, DotDict):
        return x
    if isinstance(x, dict):
        return DotDict({k: recursive_to_dotdict(v) for k, v in x.items()})
    elif isinstance(x, list):
        return [recursive_to_dotdict(item) for item in x]
    return x


def recursive_merge(default: dict[Any, Any], user: dict[Any, Any]) -> dict[Any, Any]:
    logger.warning("Recursion is not working properly... need to fix it")
    result = copy.deepcopy(default)
    for k, v in user.items():
        if isinstance(result.get(k), dict) and isinstance(v, dict):
            result[k] = recursive_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, mapping: dict[Any, Any] = None, **kwargs):
        import warnings

        warnings.warn(
            "DotDict is deprecated. Use omegaconf instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__()

        mapping = mapping or {}
        mapping.update(kwargs)
        converted = {k: recursive_to_dotdict(v) for k, v in mapping.items()}
        super().__init__(converted)

    def merge(self, other: dict[Any, Any]) -> None:
        """
        Recursively merge the 'other' dictionary into the current DotDict.

        For matching keys whose values are dictionaries, the merge is performed recursively.
        Otherwise, the value from 'other' replaces the value in the current DotDict.
        """
        new = recursive_merge(dict(self), dict(other))
        # Convert all nested dictionaries back to DotDict
        new = recursive_to_dotdict(new)
        self.clear()
        self.update(new)


if __name__ == "__main__":
    from pprint import pprint

    a = {
        "a": {
            "a1": "a1",
        },
        "b": {
            "b1": 0.8,
        },
        "c": {
            "c1": {
                "c2": {"c21": 0.1, "c22": "x"},
                "c3": {"c31": 0.1, "c32": "y"},
            },
        },
    }

    b = {
        "a": {
            "a1": "AAAAAAAAA",
        },
        "c": {
            "c1": {
                "c2": {"c21": 1, "c22": "W"},
                "c5": {"c31": 1, "c32": "T"},
            },
        },
    }

    dd = DotDict(a)
    pprint(dd)

    dd.merge(b)
    pprint(dd)
