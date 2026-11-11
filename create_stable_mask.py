import time
from pathlib import Path

import geoutils as gu
import xdem
from stable_area_masking import MaskCreator, download_landcover, logger

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Input/Output paths
data_dir = Path(".")
ref_dem_path = data_dir / "DEM_RASTER"
rgi_path = data_dir / "GLACIERS_VECTOR"
output_path = data_dir / "OUTPUT_RASTER"

# Landcover configuration
# If you already have the landcover data, you can set the path here
# Otherwise, it will be downloaded and save to this path (do not remove the path if you want to use the downloaded data)
landcover_path = data_dir / "LANDCOVER_RASTER"
lc_classes_to_mask = [
    "Tree cover",
    "Snow and Ice",
    "Permanent water bodies",
]  # You can add more classes to this list if needed

# If you want to force the download of the landcover data, set this to True
FORCE_DOWNLOAD_LC = False
landcover_collection = "esa-worldcover"
landcover_version = "v200"

# Mask refinement parameters
glacier_buffer = 100  # meters
slope_threshold = 50  # degrees
morph_structure_size = 5
morph_iterations = 2
median_filter_size = 7
gaussian_sigma = 1.8
gaussian_threshold = 0.5

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

start_time = time.time()
logger.info(f"Input DEM: {ref_dem_path}")
logger.info(f"Input RGI: {rgi_path}")
logger.info(f"Landcover path: {landcover_path}")
logger.info(f"Output mask will be saved to: {output_path}")
logger.info(f"Land cover classes to mask: {lc_classes_to_mask}")

# Load reference DEM
ref_dem = xdem.DEM(ref_dem_path)
logger.info(f"DEM loaded with shape: {ref_dem.shape}, resolution: {ref_dem.res}")

if not landcover_path.exists() or FORCE_DOWNLOAD_LC:
    import geopandas as gpd
    import shapely
    from pyproj import CRS

    logger.info(f"Landcover data not found at {landcover_path}. Downloading...")
    gdf = gpd.GeoDataFrame(
        geometry=[
            shapely.geometry.box(
                minx=ref_dem.bounds.left,
                miny=ref_dem.bounds.bottom,
                maxx=ref_dem.bounds.right,
                maxy=ref_dem.bounds.top,
            )
        ],
        crs=CRS.from_epsg(ref_dem.crs.to_epsg()),
    ).to_crs(epsg=4326)
    landcover_path = download_landcover(
        bbox=list(gdf.total_bounds),
        output_path=landcover_path,
        collection=landcover_collection,
        version=landcover_version,
        dst_crs=ref_dem.crs.to_epsg(),
        resolution=ref_dem.res,
    )
    logger.info(f"Landcover data downloaded to {landcover_path}")
else:
    logger.info(f"Using existing landcover data from {landcover_path}")

# Check if the landcover data exists
if not landcover_path.exists():
    raise FileNotFoundError(f"Landcover data not found at {landcover_path}")

# Load the landcover data and warp it to the DEM's CRS
landcover = gu.Raster(landcover_path)
landcover.reproject(ref_dem, resampling="nearest")
landcover.save(landcover_path)  # Save the reprojected landcover data

# Create a MaskCreator instance
logger.info("Creating MaskCreator instance")
mask_creator = MaskCreator()

# Load land cover data and class mapping and create a mask
logger.info("Loading land cover data and class mapping...")
mask_creator.load_land_cover(landcover_path)
mask_creator.create_mask_from_land_cover(classes_to_mask=lc_classes_to_mask)
logger.info("Initial mask created from land cover data")

# Exclude steep areas from the mask
logger.info(f"Excluding steep areas with slope > {slope_threshold} degrees")
mask_creator.exclude_steep_areas(ref_dem, slope_threshold=slope_threshold)

# Update the mask to exclude glacier areas
logger.info(f"Excluding glacier areas from RGI file with {glacier_buffer}m buffer")
mask_creator.exclude_glaciers(rgi_path, buffer=glacier_buffer)

# Refine the mask (work on the inverted mask for morphological operations)
logger.info(
    f"Applying morphological operations with structure_size={morph_structure_size}, iterations={morph_iterations}"
)
mask_creator.invert_mask()
mask_creator.apply_morphological_operations(
    structure_size=morph_structure_size, iterations=morph_iterations
)

logger.info(f"Applying median filter with filter_size={median_filter_size}")
mask_creator.apply_median_filter(filter_size=median_filter_size)

logger.info(
    f"Applying gaussian filter with sigma={gaussian_sigma}, threshold={gaussian_threshold}"
)
mask_creator.apply_gaussian_filter(sigma=gaussian_sigma, threshold=gaussian_threshold)
mask_creator.invert_mask()

# Reproject the mask to match the reference DEM
mask_creator.reproject(ref=ref_dem)

# Save the mask
logger.info(f"Saving final mask to {output_path}")
mask_creator.save_mask(output_path)

elapsed_time = time.time() - start_time
logger.info(f"Stable mask creation completed in {elapsed_time:.2f} seconds")
