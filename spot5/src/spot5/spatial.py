import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from sklearn.cluster import DBSCAN

logger = logging.getLogger("spot5")


def calculate_polygon_similarity(
    poly1: Polygon, poly2: Polygon, threshold: float = 0.9
) -> bool:
    """Calculate similarity between two polygons using intersection over union.

    Args:
        poly1 (Polygon): The first polygon.
        poly2 (Polygon): The second polygon.
        threshold (float, optional): IoU threshold above which the polygons are considered similar. Defaults to 0.9.

    Returns:
        bool: True if IoU is greater than threshold, else False.
    """
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union
    return iou > threshold


def count_footprints_overlay(
    footprints: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame | BaseGeometry,
    predicate: str = "intersects",
    output_path: str | None = None,
    inplace: bool = False,
) -> gpd.GeoDataFrame:
    """
    Counts the number of footprints overlaying/intersecting each geometry in the target.

    Parameters:
        footprints (GeoDataFrame): GeoDataFrame containing footprint geometries.
        target (GeoDataFrame or shapely Geometry): Target geometries to be evaluated.
        predicate (str): Spatial predicate for the query (default "intersects"). Options include "intersects", "within", etc.
        output_path (str, optional): If provided, the resulting GeoDataFrame is saved to this path.
        inplace (bool): If True, updates the input target GeoDataFrame directly.

    Returns:
        GeoDataFrame: The target GeoDataFrame with an added 'footprints_count' column.
    """
    logger.debug("Starting count_footprints_overlay function.")

    # Ensure target is a GeoDataFrame
    if not isinstance(target, gpd.GeoDataFrame):
        logger.debug(
            "Target is not a GeoDataFrame. Checking if it is a valid geometry."
        )
        if isinstance(target, BaseGeometry):
            target = gpd.GeoDataFrame(geometry=[target])
            logger.debug("Converted single geometry to GeoDataFrame.")
        else:
            logger.error("Invalid target type provided.")
            raise ValueError(
                "Input target must be a GeoDataFrame or a valid shapely Geometry."
            )

    logger.debug("Creating STRtree for footprint geometries.")
    # Create an STRtree of footprint geometries
    footprints_tree = STRtree(footprints.geometry)

    # Compute counts for each geometry in target using the spatial predicate
    logger.debug(
        f"Computing overlay counts for each target geometry using predicate '{predicate}'."
    )
    counts = []
    for idx, geom in enumerate(target.geometry):
        count = len(footprints_tree.query(geom, predicate=predicate))
        counts.append(count)
        logger.debug(f"Geometry index {idx}: count = {count}")

    # Determine whether to update in-place or create a copy
    if inplace:
        logger.debug("Updating input target GeoDataFrame in place.")
        target["footprints_count"] = counts
        result = target
    else:
        logger.debug("Creating a copy of the target GeoDataFrame.")
        result = target.copy()
        result["footprints_count"] = counts

    # Logging summary statistics
    total_geometries = len(result)
    total_counts = sum(counts)
    logger.info(
        f"Processed {total_geometries} geometries with a total footprint overlay count of {total_counts}."
    )

    # Save to disk if output_path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_file(output_path, driver="GeoJSON")
        logger.info(f"Saved output GeoDataFrame to {output_path}")

    logger.debug("Completed count_footprints_overlay function.")
    return result


def create_clusters(
    gdf: pd.DataFrame,
    threshold: float = 0.8,
    min_samples: int = 2,
) -> np.ndarray:
    """Group similar footprints by clustering based on polygon similarity using DBSCAN.

    Args:
        gdf (pd.DataFrame): A GeoDataFrame containing footprint geometries.
        threshold (float, optional): IoU similarity threshold. Defaults to 0.8.
        min_samples (int, optional): Minimum number of samples for a cluster in DBSCAN. Defaults to 2.

    Returns:
        np.ndarray: An array containing the cluster labels for each footprint.
    """
    n: int = len(gdf)
    similarity_matrix: np.ndarray = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            similarity: bool = calculate_polygon_similarity(
                gdf.iloc[i].geometry, gdf.iloc[j].geometry, threshold
            )
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    clustering = DBSCAN(eps=0.5, min_samples=min_samples, metric="precomputed").fit(
        1 - similarity_matrix
    )
    clusters = clustering.labels_

    return clusters


def group_similar_footprints(
    gdf: pd.DataFrame,
    threshold: float = 0.8,
    min_samples: int = 2,
    inplace: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group similar footprints by clustering based on polygon similarity using DBSCAN.

    Args:
        gdf (pd.DataFrame): A GeoDataFrame containing footprint geometries.
        threshold (float, optional): IoU similarity threshold. Defaults to 0.8.
        min_samples (int, optional): Minimum number of samples for a cluster in DBSCAN. Defaults to 2.
        inplace (bool, optional): If True, updates the input GeoDataFrame directly. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the updated GeoDataFrame and the cluster results.

    """
    clusters = create_clusters(gdf, threshold, min_samples)

    if not inplace:
        gdf = gdf.copy()
    gdf["cluster"] = clusters

    cluster_results = gdf.groupby("cluster").agg(
        {
            "orbit": lambda x: list(set(x)),
            "row": lambda x: list(set(x)),
            "shift_value": lambda x: list(set(x)),
        }
    )

    footprint_ids = {}
    for i, _ in cluster_results.iterrows():
        cluster_ids = list(gdf[clusters == i].index)
        footprint_ids[i] = cluster_ids

    cluster_results["footprint_ids"] = footprint_ids

    return gdf, cluster_results
