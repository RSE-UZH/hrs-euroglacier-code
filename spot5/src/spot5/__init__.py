import logging

# Import modules
from . import plot, spatial, utils

# Import classes
from .datastore import SpotDatastore

# Import functions
from .plot import (
    create_count_raster,
    plot_pairs_temporal_distribution,
    plot_scene_distribution,
    plot_scenes_polar_distribution,
)

# Import utilities
from .utils.timer import Timer

__version__ = "0.0.1"


# Setup logger and timer for the package
logger = utils.logger.setup_logger(level=logging.INFO, name="spot5")
timer = Timer(logger=logger)
