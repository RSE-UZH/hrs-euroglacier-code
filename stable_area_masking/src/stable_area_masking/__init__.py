"""
This package provides functions and classes for creating stable area masks from land cover data and vector geometries.
"""

from .land_cover import download_landcover
from .masking import MaskCreator
from .utils.logger import setup_logger

__all__ = ["MaskCreator", "download_landcover"]

logger = setup_logger(
    level="INFO",
    name="stable_area_masking",
    log_to_file=False,
    log_folder=None,
)
