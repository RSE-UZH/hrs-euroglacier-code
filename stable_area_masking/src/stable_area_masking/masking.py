import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import TypeVar

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import rasterio
import xdem
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, median_filter
from shapely.geometry import box
from tqdm import tqdm

logger = logging.getLogger("stable_area_masking")

T = TypeVar("T", bound="MaskCreator")  # For self-reference return type

DEFAULT_LAND_COVER_CLASS_LEGEND = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and Ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def _create_mask_from_vector(
    vector: gu.Vector,
    reference_raster: gu.Raster | None = None,
    invert: bool = True,
    crs: CRS | None = None,
    xres: float | None = None,
    yres: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    buffer: int | float = 0,
) -> gu.Raster:
    """Helper function to create mask from vector."""
    mask = vector.create_mask(
        reference_raster,
        crs=crs,
        xres=xres,
        yres=yres,
        bounds=bounds,
        buffer=buffer,
    )

    if invert:
        mask = ~mask

    return mask


def _load_land_cover_from_csv(
    classes_df: pd.DataFrame, code_col: str, class_col: str
) -> dict:
    """Helper function to load land cover class mapping from DataFrame."""
    return {k: v for k, v in classes_df[[code_col, class_col]].values}


def _create_mask_from_land_cover(
    land_cover: gu.Raster,
    codes_to_mask: list[int],
) -> gu.Raster:
    """Helper function to create mask from land cover data."""
    # Create a mask based on the specified codes
    mask = np.isin(land_cover.data.data, codes_to_mask).astype(bool)
    mask = gu.Mask.from_array(mask, land_cover.transform, land_cover.crs)

    # Invert the mask to get the stable areas
    return ~mask


def _initialize_empty_mask(
    reference_raster: gu.Raster, initial_value: bool = False
) -> gu.Raster:
    """Helper function to initialize an empty mask."""
    # Create a uniform mask with the specified initial value
    mask_data = np.full(
        (reference_raster.height, reference_raster.width), initial_value, dtype=bool
    )

    # Convert to a GeoUtils Mask
    return gu.Mask.from_array(
        mask_data, reference_raster.transform, reference_raster.crs
    )


def _apply_morphological_operations(
    mask: gu.Raster, structure_size: int = 3, iterations: int = 2
) -> gu.Raster:
    """Helper function to apply morphological operations."""
    morpho_structure = np.ones((structure_size, structure_size))
    mask_data = mask.data.data.copy().astype(bool)

    for _ in range(iterations):
        # Apply binary opening to remove small objects
        mask_data = binary_opening(mask_data, structure=morpho_structure)
        # Apply binary closing to fill small holes
        mask_data = binary_closing(mask_data, structure=morpho_structure)

    # Apply binary opening again to remove small objects
    mask_data = binary_opening(mask_data, structure=morpho_structure)

    # Convert the processed mask data back to a Mask object
    return gu.Mask.from_array(mask_data, mask.transform, mask.crs)


def _apply_median_filter(mask: gu.Raster, filter_size: int = 7) -> gu.Raster:
    """Helper function to apply median filter."""
    mask_data = mask.data.data.copy().astype(float)
    mask_data = median_filter(mask_data, size=filter_size).astype(bool)
    return gu.Mask.from_array(mask_data, mask.transform, mask.crs)


def _apply_gaussian_filter(
    mask: gu.Raster, sigma: float = 1.5, threshold: float = 0.5
) -> gu.Raster:
    """Helper function to apply Gaussian filter."""
    mask_data = mask.data.data.copy().astype(float)
    mask_data = gaussian_filter(mask_data, sigma=sigma) > threshold
    return gu.Mask.from_array(mask_data, mask.transform, mask.crs)


def prepare_glacier_outlines(
    path: str | Path, bounds: tuple[float, float, float, float], crs, buffer: float = 0
) -> gu.Vector | None:
    """Load glacier outlines for a region: in the region's CRS and buffered once.

    Only features intersecting `bounds` (+ buffer) are read. Buffering is done here, once, on the
    whole region (not per tile): glaciers lying just outside a tile must still erode it, and the
    buffer must not depend on how the region is cut. Returns None (with a warning) if there is no
    glacier in the region.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Glacier outlines file not found: {path}")

    crs = CRS.from_user_input(crs)
    left, bottom, right, top = bounds
    aoi = gpd.GeoSeries(
        [box(left - buffer, bottom - buffer, right + buffer, top + buffer)], crs=crs
    )
    file_crs = gpd.read_file(path, rows=1).crs
    gdf = gpd.read_file(path, bbox=tuple(aoi.to_crs(file_crs).total_bounds))

    if gdf.empty:
        logger.warning(
            f"No glacier outlines from {path} intersect the area of interest: "
            "no glacier will be excluded from the mask."
        )
        return None

    if file_crs != crs:
        logger.warning(
            f"Glacier outlines CRS ({file_crs.to_string()}) differs from the DEM CRS "
            f"({crs.to_string()}): reprojecting the outlines."
        )
        gdf = gdf.to_crs(crs)

    outlines = gu.Vector(gdf)
    return outlines.buffer_metric(buffer) if buffer > 0 else outlines


def _exclude_glaciers(mask: gu.Raster, glacier_outlines: gu.Vector | None) -> gu.Raster:
    """Helper function to exclude glacier areas (outlines in the mask CRS, already buffered)."""
    if glacier_outlines is None:
        return mask

    # Keep only the outlines falling in the mask extent
    outlines = glacier_outlines.crop(mask, inplace=False)
    if len(outlines.ds) == 0:
        return mask

    # Create a mask from the glacier outlines
    glacier_mask = outlines.create_mask(mask)

    # Invert the glacier mask to get the areas to exclude
    glacier_mask = ~glacier_mask

    # Apply the glacier mask to the stable mask
    result_mask = mask.copy()
    result_mask.data = np.ma.MaskedArray(np.logical_and(mask.data, glacier_mask.data))
    return result_mask


def _exclude_by_threshold(
    mask: gu.Raster,
    raster: gu.Raster,
    threshold: float,
    mode: str = "<=",
) -> gu.Raster:
    """Helper function to exclude areas based on threshold."""

    # Validate the mode parameter
    valid_modes = ["<=", ">=", "==", "<", ">", "!="]
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of {', '.join(valid_modes)}"
        )

    # Reproject the raster to match the mask
    raster.reproject(mask, inplace=True)

    # Apply the threshold based on the selected mode
    if mode == "<=":
        valid_area_mask = raster.data.data <= threshold
    elif mode == ">=":
        valid_area_mask = raster.data.data >= threshold
    elif mode == "==":
        valid_area_mask = raster.data.data == threshold
    elif mode == "<":
        valid_area_mask = raster.data.data < threshold
    elif mode == ">":
        valid_area_mask = raster.data.data > threshold
    elif mode == "!=":
        valid_area_mask = raster.data.data != threshold

    # Apply the threshold mask to the current mask
    result_mask = mask.copy()
    result_mask.data = np.ma.MaskedArray(np.logical_and(mask.data, valid_area_mask))

    return result_mask


def _save_mask(
    mask: gu.Raster,
    output_path: str | Path,
    driver: str = "GTiff",
    dtype: str = "uint8",
    **kwargs,
) -> bool:
    """Helper function to save mask to file."""
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mask.save(output_path, driver=driver, dtype=dtype, **kwargs)
        logger.info(f"Mask saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving mask: {e}")
        return False


class MaskCreator:
    """Class for creating and processing masks from raster data and vector geometries.

    This class implements a customizable pipeline for mask generation and processing,
    allowing for the creation of complex masks based on land cover data, vector
    geometries, and various processing operations.
    """

    def __init__(self, debug: bool = False, debug_dir: Path | str = None) -> None:
        """Initialize the MaskCreator.

        Args:
            debug: If True, save intermediate results for debugging
            debug_dir: Directory where debug outputs will be saved
        """
        # Land cover data
        self.land_cover: gu.Raster | None = None
        self.codes_to_mask: dict | None = None
        self.class_map: dict[int, str] | None = None

        # Mask data
        self.mask: gu.Raster | None = None

        # Debugging options
        self.debug: bool = debug
        self.debug_dir: Path = Path(debug_dir) if debug_dir else Path("./debug")
        if self.debug:
            self.debug_dir.mkdir(exist_ok=True, parents=True)

    def get_mask(self) -> gu.Raster:
        """Get the current mask object.

        Returns:
            gu.Raster: The current mask raster object

        Raises:
            ValueError: If mask has not been created yet
        """
        if self.mask is None:
            raise ValueError("Mask not created yet.")
        return self.mask.copy()

    def get_mask_as_array(self, *, as_type=bool) -> np.ndarray:
        """Get the mask data as a NumPy array.

        Args:
            as_type: Type to convert the mask data to

        Returns:
            np.ndarray: Mask data as a NumPy array of specified type

        Raises:
            ValueError: If mask has not been created yet
        """
        if self.mask is None:
            raise ValueError("Mask not created yet.")
        return self.mask.data.data.copy().astype(as_type)

    def create_mask_from_vector(
        self,
        vector: gu.Vector,
        reference_raster: gu.Raster | None = None,
        *,
        invert: bool = True,
        crs: CRS | None = None,
        xres: float | None = None,
        yres: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        buffer: int | float = 0,
        overwrite: bool = False,
    ) -> T:
        """Create a mask from a vector file.

        Args:
            vector: Vector file to create the mask from
            reference_raster: Raster file to align the mask with
            invert: If True, invert the mask (the area within the vector will be masked out). Default is True.
            crs: Coordinate reference system for the output mask
            xres: X resolution (pixel width) for the output mask
            yres: Y resolution (pixel height) for the output mask
            bounds: Output bounds as (left, bottom, right, top)
            buffer: Buffer distance to apply to the vector features
            overwrite: Whether to overwrite existing mask if one exists

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask already exists and overwrite is False
        """
        if self.mask is not None and not overwrite:
            raise ValueError("Mask already created. Use overwrite=True to replace it.")

        self.mask = _create_mask_from_vector(
            vector, reference_raster, invert, crs, xres, yres, bounds, buffer
        )

        logger.info("Mask created from vector data.")

        if self.debug:
            _save_mask(self.mask, self.debug_dir / "0_mask_from_vector.tif")

        return self

    def reproject(self, **kwargs) -> T:
        """Reproject the mask to a different coordinate reference system.

        This is a wrapper around the geoutils.Raster.reproject method.
        All the arguments are passed directly to that method. Refer to the geoutils documentation for more details.

        Args:
            **kwargs: Keyword arguments for the reproject method

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask has not been created yet
        """
        if self.mask is None:
            raise ValueError("Mask not created yet.")

        # Set default values for resampling
        if kwargs.get("resampling") is None:
            kwargs["resampling"] = "nearest"

        self.mask.reproject(inplace=True, **kwargs)

        return self

    def load_land_cover(
        self,
        land_cover_path: str | Path | gu.Raster,
        *,
        mapping_path: str | Path | None = None,
        mapping_dict: dict | None = None,
        code_col: str = "Map code",
        class_col: str = "Land Cover Class",
        use_default_mapping: bool = True,
    ) -> T:
        """Load land cover raster data and optionally class mapping from file or dictionary.

        Args:
            land_cover_path: Path to land cover raster file (or an already loaded Raster, e.g. a tile)
            mapping_path: Path to CSV file with class mapping
            mapping_dict: Dictionary with class mapping
            code_col: Column name for codes in CSV
            class_col: Column name for class labels in CSV
            use_default_mapping: Whether to use default CORINE mapping if no other mapping provided

        Returns:
            MaskCreator: Self reference for method chaining
        """
        self.land_cover = (
            land_cover_path
            if isinstance(land_cover_path, gu.Raster)
            else gu.Raster(land_cover_path)
        )

        if mapping_path:
            classes_df = pd.read_csv(mapping_path, delimiter=";")
            self.class_map = _load_land_cover_from_csv(classes_df, code_col, class_col)
        elif mapping_dict:
            self.class_map = mapping_dict
        elif use_default_mapping:
            self.class_map = DEFAULT_LAND_COVER_CLASS_LEGEND
            logger.info("Using default CORINE Land Cover class mapping")

        return self

    def create_mask_from_land_cover(
        self,
        *,
        classes_to_mask: list[str] | None = None,
        codes_to_mask: list[int] | None = None,
        overwrite: bool = False,
    ) -> T:
        """Create a mask from land cover data based on specified classes or codes.

        Args:
            classes_to_mask: List of class labels to mask
            codes_to_mask: List of codes to mask
            overwrite: Whether to overwrite existing mask if one exists

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If land cover data not loaded, mask already exists and
                       overwrite is False, or no classes or codes specified
        """

        def _get_classes_to_mask(class_map: dict, classes_list: list[str]) -> list[int]:
            """Helper to find codes that match the given class names."""
            return [code for code, cls in class_map.items() if cls in classes_list]

        if self.land_cover is None:
            raise ValueError(
                "Land cover data not loaded. Load them first with the 'load_land_cover()' method."
            )

        if self.mask is not None and not overwrite:
            raise ValueError("Mask already created. Use overwrite=True to replace it.")

        if classes_to_mask is None and codes_to_mask is None:
            raise ValueError("No classes or codes specified for masking")

        if classes_to_mask is not None and self.class_map:
            codes_to_mask = _get_classes_to_mask(self.class_map, classes_to_mask)

        if not codes_to_mask:
            raise ValueError("No valid codes specified for masking")

        self.mask = _create_mask_from_land_cover(self.land_cover, codes_to_mask)

        if self.debug:
            _save_mask(self.mask, self.debug_dir / "0_mask_from_land_cover.tif")

        return self

    def initialize_empty_mask(
        self, reference_raster: gu.Raster, initial_value: bool = True
    ) -> T:
        """Initialize an empty mask based on a reference raster.

        Creates a mask with uniform values (True or False) matching the reference raster's geometry,which can then be refined using other methods. False indicates masked out areas.

        Args:
            reference_raster: Raster to use as reference for dimensions, projection and transform
            initial_value: Initial value for all pixels (True=valid areas, False=masked areas). If True, the mask will be initialized with valid areas, and you can then apply further processing to refine it. Default is True.

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            TypeError: If reference is not a GeoUtils Raster object
        """
        if not isinstance(reference_raster, gu.Raster):
            raise TypeError("Reference must be a GeoUtils Raster object")

        self.mask = _initialize_empty_mask(reference_raster, initial_value)
        logger.info(f"Empty mask initialized with value {initial_value}")

        if self.debug:
            _save_mask(self.mask, self.debug_dir / "0_empty_mask.tif")

        return self

    def apply_morphological_operations(
        self, structure_size: int = 3, iterations: int = 2
    ) -> T:
        """Apply morphological operations to refine the mask.

        This applies a sequence of binary opening and closing operations to remove
        noise and small features from the mask.

        Args:
            structure_size: Size of the structuring element (square)
            iterations: Number of iterations to apply the operations

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask has not been created yet
        """
        if self.mask is None:
            raise ValueError("No mask available. Create a mask first.")

        logger.info(f"Applying sequence of {iterations} binary opening and closing...")
        self.mask = _apply_morphological_operations(
            self.mask, structure_size, iterations
        )
        logger.info("Morphological operations completed.")

        if self.debug:
            _save_mask(self.mask, self.debug_dir / "1_binary_morpho.tif")

        return self

    def apply_median_filter(self, filter_size: int = 7) -> T:
        """Apply median filter to the mask.

        This smoothes the mask by replacing each pixel with the median value in its
        neighborhood.

        Args:
            filter_size: Size of the median filter window

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask has not been created yet
        """
        if self.mask is None:
            raise ValueError("No mask available. Create a mask first.")

        self.mask = _apply_median_filter(self.mask, filter_size)
        logger.info("Median filter applied.")

        if self.debug:
            _save_mask(self.mask, self.debug_dir / "2_binary_median.tif")

        return self

    def apply_gaussian_filter(self, sigma: float = 1.5, threshold: float = 0.5) -> T:
        """Apply Gaussian filter to smooth the mask edges.

        This applies a Gaussian smoothing filter to the mask and then thresholds
        the result to create a binary mask.

        Args:
            sigma: Standard deviation for the Gaussian kernel
            threshold: Threshold value for converting back to binary mask

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask has not been created yet
        """
        if self.mask is None:
            raise ValueError("No mask available. Create a mask first.")

        self.mask = _apply_gaussian_filter(self.mask, sigma, threshold)
        logger.info("Gaussian filter applied.")

        if self.debug:
            _save_mask(self.mask, self.debug_dir / "3_binary_gaussian.tif")

        return self

    def invert_mask(self) -> T:
        """Invert the mask so that masked areas are False and valid areas are True.

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask has not been created yet
        """
        if self.mask is None:
            raise ValueError("No mask available. Create a mask first.")

        self.mask = ~self.mask
        logger.info("Mask inverted.")

        return self

    def exclude_glaciers(
        self, glacier_outlines: str | Path | gu.Vector | None, buffer: int = 0
    ) -> T:
        """Exclude glacier areas from the stable mask.

        Args:
            glacier_outlines: Path to a glacier outlines vector file, or a Vector already in the
                mask CRS and buffered (see `prepare_glacier_outlines`; lets tiled processing load
                and buffer the outlines only once). None (e.g. no glacier in the area) leaves the
                mask unchanged.
            buffer: Buffer distance in meters around the outlines (only used with a path)

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask has not been created yet
            FileNotFoundError: If glacier outlines file does not exist
        """
        if self.mask is None:
            raise ValueError("No mask available. Create a mask first.")

        if isinstance(glacier_outlines, (str, Path)):
            glacier_outlines = prepare_glacier_outlines(
                glacier_outlines, self.mask.bounds, self.mask.crs, buffer
            )

        self.mask = _exclude_glaciers(self.mask, glacier_outlines)
        logger.info("Glacier areas excluded from the mask.")

        if self.debug:
            _save_mask(self.mask, self.debug_dir / "5_excluded_glaciers.tif")

        return self

    def exclude_by_threshold(
        self,
        raster: gu.Raster,
        threshold: float,
        mode: str = "<=",
    ) -> T:
        """Exclude areas from the mask based on a threshold applied to any raster.

        Args:
            raster: Input raster to compare against threshold
            threshold: Threshold value for filtering
            mode: Comparison operator ("<=", ">=", "==", "<", ">", "!=")

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask has not been created yet or mode is invalid
        """
        if self.mask is None:
            raise ValueError(
                "No mask available. Create a mask first or initialize an empty one."
            )

        # Use the helper function
        self.mask = _exclude_by_threshold(self.mask, raster, threshold, mode)
        logger.info(f"Excluded areas where input raster {mode} {threshold}")

        if self.debug:
            _save_mask(self.mask, self.debug_dir / "threshold_excluded.tif")

        return self

    def exclude_steep_areas(self, dem: xdem.DEM, slope_threshold: float = 30.0) -> T:
        """Exclude steep areas from the mask based on a digital elevation model.

        Computes slopes from the provided DEM and excludes areas where the slope
        exceeds the specified threshold.

        Args:
            dem: Digital elevation model (xdem.DEM object)
            slope_threshold: Maximum slope in degrees to include in the mask

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask has not been created yet
            TypeError: If dem is not an xdem.DEM object
        """
        if not isinstance(dem, xdem.DEM):
            raise TypeError("DEM must be an xdem.DEM object")

        # Calculate slope raster
        logger.info("Calculating slopes from DEM...")
        slope_raster = dem.slope()

        # Use the general threshold-based exclusion with the slope raster
        return self.exclude_by_threshold(
            raster=slope_raster,
            threshold=slope_threshold,
            mode="<=",
        )

    def apply_custom_processing(
        self, processing_func: Callable[[gu.Raster], gu.Raster]
    ) -> T:
        """Apply custom processing function to the mask.

        Args:
            processing_func: Function that takes a mask as input and returns a
                           processed mask

        Returns:
            MaskCreator: Self reference for method chaining

        Raises:
            ValueError: If mask has not been created yet
        """
        if self.mask is None:
            raise ValueError("No mask available. Create a mask first.")

        self.mask = processing_func(self.mask)
        return self

    def save_mask(
        self,
        output_path: str | Path,
        *,
        driver: str = "GTiff",
        dtype: str = "uint8",
        **kwargs,
    ) -> bool:
        """Save the mask to a file.

        Args:
            output_path: Path where the mask will be saved (will be converted to .tif if no extension is provided)
            driver: Raster driver to use for saving
            dtype: Data type for the output raster
            **kwargs: Additional keyword arguments for the save method

        Returns:
            bool: True if save was successful, False otherwise

        Raises:
            ValueError: If mask has not been created yet
        """
        if self.mask is None:
            raise ValueError("No mask available. Create a mask first.")

        return _save_mask(self.mask, output_path, driver, dtype, **kwargs)


# =============================================================================
# Tiled processing (for DEMs too large to fit in memory)
# =============================================================================


def _tile_halo(
    structure_size: int, iterations: int, median_size: int, gaussian_sigma: float
) -> int:
    """Overlap (pixels) so that the core of a tile is identical to a full-raster run.

    All steps after rasterization are local filters, so their support radii simply add up:
    3x3 slope kernel + opening/closing chain + median + (truncated) Gaussian. Beyond this
    distance from a tile edge, the artificial tile border cannot influence the result.
    """
    slope = 1
    # (opening + closing) x iterations, then one more opening; each is an erosion + a dilation
    morpho = (structure_size // 2) * 2 * (2 * iterations + 1)
    median = median_size // 2
    gaussian = int(4.0 * gaussian_sigma + 0.5)  # scipy's default truncate=4.0
    return slope + morpho + median + gaussian


def _iter_windows(width: int, height: int, tile_size: int):
    """Yield the (non-overlapping) core windows covering the raster."""
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            yield Window(
                col, row, min(tile_size, width - col), min(tile_size, height - row)
            )


def _pad_window(core: Window, halo: int, width: int, height: int) -> Window:
    """Grow a window by `halo` pixels, clipped to the raster (so true raster edges stay true edges)."""
    col0, row0 = max(core.col_off - halo, 0), max(core.row_off - halo, 0)
    col1 = min(core.col_off + core.width + halo, width)
    row1 = min(core.row_off + core.height + halo, height)
    return Window(col0, row0, col1 - col0, row1 - row0)


def _check_window_has_no_nodata(dem_path: str | Path, win: Window) -> None:
    """Raise if the DEM has nodata/NaN pixels in this window."""
    with rasterio.open(dem_path) as src:
        arr = src.read(1, window=win, masked=True)
    bad = np.ma.getmaskarray(arr) | ~np.isfinite(arr.data)
    if bad.any():
        raise ValueError(
            f"The DEM contains nodata/NaN pixels ({bad.sum()} in the window starting at row "
            f"{win.row_off}, col {win.col_off}; possibly more elsewhere). Slope and filtering "
            "need a complete DEM: fill the gaps (or crop the DEM) and try again."
        )


def _run_parallel(pool: ThreadPoolExecutor, func: Callable, items: list, desc: str):
    """Apply `func` to all items in the pool, with a progress bar; stop at the first error."""
    futures = [pool.submit(func, item) for item in items]
    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        raise


def create_stable_mask_tiled(
    dem_path: str | Path,
    landcover_path: str | Path,
    output_path: str | Path,
    glacier_path: str | Path | None = None,
    *,
    lc_classes_to_mask: list[str],
    glacier_buffer: float = 0,
    slope_threshold: float = 50.0,
    morph_structure_size: int = 5,
    morph_iterations: int = 2,
    median_filter_size: int = 7,
    gaussian_sigma: float = 1.8,
    gaussian_threshold: float = 0.5,
    tile_size: int = 2048,
    n_workers: int = 1,
) -> Path:
    """Compute the stable-terrain mask tile by tile, with bounded memory.

    Runs the same MaskCreator pipeline as the in-memory workflow on overlapping tiles, keeps only
    the core of each tile (overlap = `_tile_halo`, so the result is identical to a full-raster
    run) and writes it straight to the output GeoTIFF. Memory scales with `tile_size`**2 and
    `n_workers` (peak ~0.8 GB per worker for 2048 px tiles, measured), not with the DEM size.

    Tiles are independent and processed by `n_workers` threads: the numpy/scipy/xdem/GDAL calls
    doing the work release the GIL. Each tile opens its own file handles (rasterio datasets are
    not thread-safe); only the output writes are serialized.

    The land cover is warped on the fly to the DEM grid (nearest), tile by tile, and may thus
    be in another CRS/resolution. Glacier outlines are reprojected (with a warning) if needed;
    if none falls within the DEM extent, a warning is issued and no glacier is excluded.

    Args:
        tile_size: Core tile size in pixels (a multiple of 512 matches the output tiling).
        n_workers: Number of threads (tiles processed at the same time).

    Raises:
        ValueError: If the DEM contains nodata/NaN pixels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    halo = _tile_halo(
        morph_structure_size, morph_iterations, median_filter_size, gaussian_sigma
    )
    logger.info(
        f"Tiled processing: {tile_size}px tiles with a {halo}px overlap, {n_workers} thread(s)"
    )

    with rasterio.open(dem_path) as dem_src, rasterio.open(landcover_path) as lc_src:
        width, height, crs = dem_src.width, dem_src.height, dem_src.crs
        dem_transform, dem_bounds = dem_src.transform, dem_src.bounds
        dem_nodata, lc_crs = dem_src.nodata, lc_src.crs
    cores = list(_iter_windows(width, height, tile_size))

    if lc_crs != crs:
        logger.warning(
            f"Land cover CRS ({lc_crs}) differs from the DEM CRS ({crs}): "
            "reprojecting the land cover on the fly."
        )

    # Outlines: loaded, reprojected and buffered once for the whole DEM extent, then shared
    # (read-only) by the threads. The spatial index is built now to avoid a race on first use.
    outlines = None
    if glacier_path is not None:
        outlines = prepare_glacier_outlines(
            glacier_path, dem_bounds, crs, glacier_buffer
        )
        if outlines is not None:
            _ = outlines.ds.sindex

    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1, "width": width,
        "height": height, "crs": crs, "transform": dem_transform,
        "compress": "deflate", "tiled": True, "blockxsize": 512,
        "blockysize": 512, "BIGTIFF": "IF_SAFER",
    }  # fmt: skip
    write_lock = threading.Lock()

    def process_tile(core: Window) -> None:
        win = _pad_window(core, halo, width, height)
        transform = rasterio.windows.transform(win, dem_transform)

        # One set of handles per tile: rasterio datasets must not be shared between threads
        with (
            rasterio.open(dem_path) as dem_src,
            rasterio.open(landcover_path) as lc_src,
            WarpedVRT(
                lc_src, crs=crs, transform=dem_transform, width=width,
                height=height, resampling=Resampling.nearest,
            ) as lc_vrt,
        ):  # fmt: skip
            dem = xdem.DEM.from_array(
                dem_src.read(1, window=win), transform, crs, nodata=dem_nodata
            )
            land_cover = gu.Raster.from_array(
                lc_vrt.read(1, window=win), transform, crs
            )

        # Same sequence of steps as the in-memory workflow, on the padded tile
        mc = MaskCreator()
        mc.load_land_cover(land_cover)
        mc.create_mask_from_land_cover(classes_to_mask=lc_classes_to_mask)
        mc.exclude_steep_areas(dem, slope_threshold=slope_threshold)
        mc.exclude_glaciers(outlines)
        mc.invert_mask()
        mc.apply_morphological_operations(
            structure_size=morph_structure_size, iterations=morph_iterations
        )
        mc.apply_median_filter(filter_size=median_filter_size)
        mc.apply_gaussian_filter(sigma=gaussian_sigma, threshold=gaussian_threshold)
        mc.invert_mask()

        # Write the core only; the overlap is discarded
        r0, c0 = core.row_off - win.row_off, core.col_off - win.col_off
        stable = mc.get_mask_as_array()[r0 : r0 + core.height, c0 : c0 + core.width]
        with write_lock:
            dst.write(stable.astype("uint8"), 1, window=core)

    log_level = logger.level
    logger.setLevel(logging.WARNING)  # silence the per-tile messages of MaskCreator
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            _run_parallel(
                pool,
                partial(_check_window_has_no_nodata, dem_path),
                cores,
                "Checking DEM for nodata",
            )
            with rasterio.open(output_path, "w", **profile) as dst:
                _run_parallel(pool, process_tile, cores, "Stable mask tiles")
    finally:
        logger.setLevel(log_level)

    logger.info(f"Mask saved to {output_path}")
    return output_path


if __name__ == "__main__":
    debug_dir = Path("/home/fioli/rse_aletsch/sandbox/mask")

    # ===== Create a mask from the rgi vector only

    # Initialize the mask creator
    # mask_creator = MaskCreator(debug=False)

    # # Load the RGI vector and reference raster
    # rgi = gu.Vector("data/RGI2000-v7.0-G-11_central_europe.geojson")
    # reference_raster = gu.Raster("data/landcover_32632.tif")

    # # Create a mask from the RGI vector
    # mask_creator.create_mask_from_vector(vector=rgi, reference_raster=reference_raster)
    # mask_creator.save_mask(debug_dir / "rgi_mask.tif")

    # # Create a mask from the rgi vector with a buffer
    # mask_creator.create_mask_from_vector(
    #     vector=rgi, reference_raster=reference_raster, buffer=100, overwrite=True
    # )
    # mask_creator.save_mask(debug_dir / "rgi_mask_with_buffer.tif")

    # ===== Create a mask from land cover data
    # mask_creator = MaskCreator(debug=False, debug_dir=debug_dir)

    # # Load land cover data and class mapping
    # mask_creator.load_land_cover(
    #     "data/landcover_32632.tif", mapping_path="data/lccs_classes_legend.csv"
    # )

    # # Create basic binary mask
    # mask_creator.create_mask_from_land_cover(
    #     classes_to_mask=[
    #         "Tree cover",
    #         "Snow and Ice",
    #         "Permanent water bodies",
    #     ]
    # )
    # mask_creator.save_mask(debug_dir / "mask_from_land_cover.tif")

    # # Refine the mask (work on the inverted mask for morphological operations)
    # mask_creator.invert_mask()
    # mask_creator.apply_morphological_operations(structure_size=5, iterations=2)
    # mask_creator.apply_median_filter(filter_size=7)
    # mask_creator.apply_gaussian_filter(sigma=1.8, threshold=0.5)
    # mask_creator.invert_mask()
    # mask_creator.save_mask(debug_dir / "mask_from_land_cover_refined.tif")

    # # Update the mask to exclude glacier areas
    # mask_creator.exclude_glaciers(
    #     "data/RGI2000-v7.0-G-11_central_europe.geojson", buffer=50
    # )
    # mask_creator.save_mask(debug_dir / "mask_from_land_cover_refined_and_rgi.tif")

    # Example of using the new methods (commented out)
    import xdem

    dem = xdem.DEM("data/COP-DEM_GLO-30-DGED_2023_1_32632_ellh.tif")

    # # Method 1: Create a mask from scratch using only slope
    mask_creator = MaskCreator(debug=True, debug_dir=debug_dir)
    mask_creator.initialize_empty_mask(dem)
    mask_creator.exclude_steep_areas(dem, slope_threshold=25.0)
    mask_creator.apply_morphological_operations(structure_size=5, iterations=2)
    mask_creator.save_mask(debug_dir / "slope_mask.tif")

    # Method 2: Add slope criteria to an existing mask
    mask_creator.exclude_steep_areas(dem, slope_threshold=25.0)
    mask_creator.save_mask(debug_dir / "combined_mask_with_slope.tif")

    print("done")
