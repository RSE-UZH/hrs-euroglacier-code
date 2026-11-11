from pathlib import Path


def check_path(path: str | Path) -> Path:
    """Check if a path exists and return a Path object.

    Args:
        path (str | Path): The path to check.

    Returns:
        Path: The Path object.

    Raises:
        FileNotFoundError: If the path does not exist.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")
    return path


def check_dir(path: str | Path) -> Path:
    """Check if a directory exists and return a Path object.

    Args:
        path (str | Path): The directory to check.

    Returns:
        Path: The Path object.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory {path} does not exist")
    return path


def make_dir(path: str | Path, parents: bool = True, exists_ok: bool = True) -> Path:
    """Create a directory if it does not exist and return a Path object.

    Args:
        path (str | Path): The directory to create.

    Returns:
        Path: The Path object.
    """

    path = Path(path)
    path.mkdir(parents=parents, exist_ok=exists_ok)
    return path
