import logging
from pathlib import Path
from zipfile import ZipFile

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger("spot5")


def unzip_scenes(zip_files: list[Path], extract_dir: Path, workers: int = -1) -> bool:
    """Unzip multiple scene files in parallel.

    Args:
        zip_files (list[Path]): List of zip file paths.
        extract_dir (Path): Directory to extract the zip files.
        workers (int, optional): Maximum number of workers for parallel processing. Defaults to -1.

    Returns:
        bool: True if all files were extracted successfully
    """

    def _unzip(zip_path: Path, extract_dir: Path) -> bool:
        """Unzip a scene file if not already extracted."""
        out_dir = extract_dir / zip_path.stem
        if not out_dir.exists():
            try:
                with ZipFile(zip_path) as zf:
                    zf.extractall(out_dir)
                return True
            except Exception as e:
                logger.error(f"Failed to extract {zip_path}: {e}")
                return False

    extract_dir.mkdir(exist_ok=True)
    with logging_redirect_tqdm([logger]):
        Parallel(n_jobs=workers, backend="threading")(
            delayed(_unzip)(zip_file, extract_dir) for zip_file in tqdm(zip_files)
        )

    return True
