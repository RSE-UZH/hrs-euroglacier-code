import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.transform import Affine, array_bounds, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform_bounds
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

logger = logging.getLogger("stable_area_masking")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def metre2deg(m_val):
    """Convert meters to decimal degrees (approximate at equator)"""
    return m_val / 111320


BLOCK_SIZE = 4096  # pixels; blocks are warped, pasted and written one at a time


def _floor(x: float) -> int:
    return math.floor(round(x, 6))  # tolerant to float noise on exact pixel edges


def _ceil(x: float) -> int:
    return math.ceil(round(x, 6))


def _target_grid(
    src: DatasetReader,
    bbox: list[float],
    dst_crs: str | int | None,
    resolution: float | tuple[float, float] | None,
    dst_bounds: tuple[float, float, float, float] | None,
    reproject_to_utm: bool,
) -> tuple[CRS, Affine, int, int]:
    """Define the output grid (CRS, transform, width, height) from the options."""
    # Native grid: the part of the source grid covering bbox (WorldCover tiles share one grid)
    win = from_bounds(*bbox, transform=src.transform)
    col0, row0 = _floor(win.col_off), _floor(win.row_off)
    native_w = _ceil(win.col_off + win.width) - col0
    native_h = _ceil(win.row_off + win.height) - row0
    native_transform = src.transform * Affine.translation(col0, row0)

    if dst_crs is not None and resolution is not None:
        # Case 1: User defined both dst_crs and resolution
        crs = CRS.from_user_input(dst_crs)
        if isinstance(resolution, (int, float)):
            x_res, y_res = resolution, resolution
        else:
            x_res, y_res = resolution
        logger.info(
            f"Reprojecting to user-specified CRS: {crs} with resolution {x_res}, {y_res}"
        )

        # The extent is the one given (e.g. the DEM footprint), else that of the bbox
        left, bottom, right, top = dst_bounds or transform_bounds(
            "EPSG:4326", crs, *bbox, densify_pts=21
        )
        width, height = _ceil((right - left) / x_res), _ceil((top - bottom) / y_res)
        return crs, from_origin(left, top, x_res, y_res), width, height

    if reproject_to_utm and dst_crs is None and resolution is None:
        # Case 2: Auto UTM reprojection
        bbox_geom = gpd.GeoSeries.from_xy(bbox[::2], bbox[1::2], crs="EPSG:4326")
        crs = CRS.from_epsg(bbox_geom.estimate_utm_crs().to_epsg())
        logger.info(
            f"Reprojecting to estimated UTM projection: {crs} with default resolution"
        )
        bounds = array_bounds(native_h, native_w, native_transform)
        transform, width, height = calculate_default_transform(
            src.crs, crs, native_w, native_h, *bounds
        )
        return crs, transform, width, height

    # Case 3: Keep original projection and resolution
    if dst_crs is not None or resolution is not None:
        logger.warning(
            "Both dst_crs and resolution must be provided to reproject. Keeping original projection."
        )
    else:
        logger.info("Keeping original projection and resolution")
    return src.crs, native_transform, native_w, native_h


def _paste_item(item, dst: DatasetWriter) -> bool:
    """Warp one item onto the output grid and paste it, block by block (bounded memory).

    Returns whether any data was written.
    """
    # Sign here, per item: the tokens expire, and a large download can take hours
    href = planetary_computer.sign(item.assets["map"].href)

    # Blocks of the output grid that the item can touch
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", dst.crs, *item.bbox, densify_pts=21
    )
    win = from_bounds(left, bottom, right, top, transform=dst.transform)
    col0, row0 = max(_floor(win.col_off), 0), max(_floor(win.row_off), 0)
    col1 = min(_ceil(win.col_off + win.width), dst.width)
    row1 = min(_ceil(win.row_off + win.height), dst.height)
    blocks = [
        Window(c, r, min(c + BLOCK_SIZE, col1) - c, min(r + BLOCK_SIZE, row1) - r)
        for r in range(max(row0, 0), row1, BLOCK_SIZE)
        for c in range(max(col0, 0), col1, BLOCK_SIZE)
    ]  # fmt: skip

    found = False
    with (
        rasterio.open(href) as src,
        WarpedVRT(
            src, crs=dst.crs, transform=dst.transform, width=dst.width,
            height=dst.height, resampling=Resampling.nearest, src_nodata=0, nodata=0,
        ) as vrt,
    ):  # fmt: skip
        for block in tqdm(blocks, desc=f"Land cover {item.id}"):
            new = vrt.read(1, window=block)
            if not new.any():
                continue
            # Items share borders: 0 is nodata in WorldCover, so only fill what is still empty
            current = dst.read(1, window=block)
            dst.write(
                np.where(current == 0, new, current).astype("uint8"), 1, window=block
            )
            found = True
    return found


def download_landcover(
    bbox: list[float],
    output_path: str | Path,
    collection: str = "esa-worldcover",
    version: str = "v200",
    dst_crs: str | int | None = None,
    resolution: float | tuple[float, float] | None = None,
    reproject_to_utm: bool = True,
    dst_bounds: tuple[float, float, float, float] | None = None,
) -> Path:
    """Download land cover data from Microsoft Planetary Computer.

    The data is streamed to disk block by block (warped to the output grid on the fly), so memory
    use is bounded by BLOCK_SIZE, whatever the size of the area.

    Args:
        bbox: Bounding box in format [minx, miny, maxx, maxy] (in EPSG:4326)
        output_path: Path to save the downloaded land cover data
        collection: Collection name to query. Defaults to "esa-worldcover".
        version: Version of the collection to query. Defaults to "v200".
        reproject_to_utm: Whether to estimate and use UTM projection when dst_crs
            is not provided. Defaults to True.
        dst_crs: Destination CRS to reproject the data to (e.g. 'EPSG:32633' or 32633).
            Defaults to None.
        resolution: Resolution in units of dst_crs (if scalar, used for both x and y;
            if tuple, (x_res, y_res)). Defaults to None.
        dst_bounds: Output extent (left, bottom, right, top) in dst_crs; only used when dst_crs
            and resolution are given. Pass the DEM bounds and resolution to get exactly the DEM
            grid. Defaults to the extent of bbox.

    Returns:
        Path to the downloaded land cover data

    Raises:
        ValueError: If no items found for the given bbox in the collection
        ValueError: If no items found with the specified version
        ValueError: If no 'map' asset found in the item
        ValueError: If the downloaded data is empty
    """
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to Planetary Computer
    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace
    )

    # Search for items
    search = catalog.search(collections=[collection], bbox=bbox)
    items = list(search.items())
    if not items:
        raise ValueError(
            f"No items found for the given bbox in collection {collection}"
        )

    # Filter items by version
    items = [item for item in items if version in item.id]
    if not items:
        raise ValueError(
            f"No items found for the given bbox in collection {collection} with version {version}. Try using v100 instead."
        )

    for item in items:
        if "map" not in item.assets:
            logger.info(f"Available assets: {list(item.assets.keys())}")
            raise ValueError("No 'map' asset found in the item")

    # Output grid, then stream every item into it
    with rasterio.open(planetary_computer.sign(items[0].assets["map"].href)) as src:
        crs, transform, width, height = _target_grid(
            src, bbox, dst_crs, resolution, dst_bounds, reproject_to_utm
        )
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1, "width": width,
        "height": height, "crs": crs, "transform": transform, "nodata": 0,
        "compress": "deflate", "tiled": True, "blockxsize": 512,
        "blockysize": 512, "BIGTIFF": "IF_SAFER", "SPARSE_OK": "TRUE",
    }  # fmt: skip

    # Write to a temporary name: an interrupted download must not leave a file that looks valid
    part = output_path.with_name(output_path.name + ".part")
    logger.info(
        f"Downloading {len(items)} item(s) to {output_path} ({width} x {height} px)"
    )
    try:
        with rasterio.Env(GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2"):
            with rasterio.open(part, "w", **profile):
                pass
            with rasterio.open(part, "r+") as dst:
                found = [_paste_item(item, dst) for item in items]
        if not any(found):
            raise ValueError("Downloaded data is empty. Please check the bounding box.")
        part.replace(output_path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    logger.info(f"Land cover data saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage
    # Define a bounding box (in WGS84/EPSG:4326)
    bbox = [-19, 64, -18, 65.0]  # [minx, miny, maxx, maxy]

    output_path = Path("./esa_worldcover.tif")

    download_landcover(bbox, output_path)
