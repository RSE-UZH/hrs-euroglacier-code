import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj.crs import CRS
from shapely.geometry import Polygon

logger = logging.getLogger("spot5")


def extract_cloudcover(scene_dir: Path) -> float:
    """Extract cloud cover percentage from SPOT HRS metadata.

    Args:
        scene_dir (Path): Directory containing the scene metadata.

    Returns:
        float: Cloud cover percentage.
    """
    p_meta = Path(scene_dir) / "SCENE01" / "INFO.XML"

    tree = ET.parse(p_meta)
    root = tree.getroot()

    # Find freeCloudPercent and convert to float
    free_cloud = float(root.find(".//freeCloudPercent").text)
    return 100 - free_cloud


def extract_metadata(metadata_file: Path) -> pd.Series:
    tree = ET.parse(metadata_file)
    root = tree.getroot()

    dataset_name = root.find(".//DATASET_NAME").text
    date = root.find(".//IMAGING_DATE").text
    time = root.find(".//IMAGING_TIME").text
    grid_reference = root.find(".//GRID_REFERENCE").text
    shift_value = root.find(".//SHIFT_VALUE").text
    incidence_angle = root.find(".//INCIDENCE_ANGLE").text
    viewing_angle = root.find(".//VIEWING_ANGLE").text
    sun_azimuth = root.find(".//SUN_AZIMUTH").text
    sun_elevation = root.find(".//SUN_ELEVATION").text

    return pd.Series(
        {
            "dataset_name": [dataset_name],
            "date": [date],
            "time": [time],
            "grid_reference": [grid_reference],
            "shift_value": [shift_value],
            "incidence_angle": [incidence_angle],
            "viewing_angle": [viewing_angle],
            "sun_azimuth": [sun_azimuth],
            "sun_elevation": [sun_elevation],
        }
    )


def extract_scene_id(metadata_file: Path) -> str:
    """Extract the source scene ID (SOURCE_ID) from a METADATA.DIM file.

    Kept separate from `extract_metadata` on purpose: adding it there would change the
    contents of SceneMetadata.metadata / .footprint, which are stored in the datastore pickle.
    """
    return ET.parse(metadata_file).getroot().find(".//SOURCE_ID").text


def extract_footprint(metadata_file: Path) -> tuple[Polygon, CRS]:
    """Extract scene extent as GeoDataFrame polygon.

    Args:
        scene_dir (Path): Directory containing the scene metadata.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the scene extent polygon.
    """

    tree = ET.parse(metadata_file)
    root = tree.getroot()

    # Extract vertices from XML metadata
    vertices = root.findall(".//Vertex")
    coords = []
    for vertex in vertices:
        lon = float(vertex.find("FRAME_LON").text)
        lat = float(vertex.find("FRAME_LAT").text)
        row = int(vertex.find("FRAME_ROW").text)
        col = int(vertex.find("FRAME_COL").text)
        coords.append([lon, lat, row, col])
    vertices_array = np.array(coords)

    # Verify expected row/col pattern
    expected_rows = [1, 1, 12000, 12000]
    expected_cols = [1, 12000, 12000, 1]
    if not (
        np.array_equal(vertices_array[:, 2], expected_rows)
        and np.array_equal(vertices_array[:, 3], expected_cols)
    ):
        raise ValueError("Unusual row/col order found!")

    # Extract CRS
    crs = root.find(".//HORIZONTAL_CS_CODE").text
    crs = CRS.from_string(crs)

    # Create polygon from lon/lat coordinates
    poly = Polygon(vertices_array[:, :2])

    return poly, crs


def parse_strip_id(scene_name: str) -> tuple[int, int, int]:
    """
    Parse the Spot5 scene name to extract strip id (orbital path, row, shift value), e.g. '188-270-0'. Convert '188-270-0' to an integer tuple, suitable for sorting.
    Spot5 naming schema is the following:
        <scene_rank>-<scene_count>_S<platform>_<grid_reference>-<shift_value>_<YYYY>-<MM>-<DD>-<HH>-<MM>-<SS>_<instrument>_<sensor_code>_<transmission_mode>_<station_code>
    where
        <grid_reference> is the orbital path and row, such as 189-270. Abramov is always found in one of the following: paths 187 to 190, row 269 to 271.
        <shift_value> is 0...9: “Shift along the track value for scenes located between 2 grid nodes for the current source scene. This value is sensor dependant, Spot uses a figure between 0 and 9 which is given in tenth of a scene.”

    Example: 002-005_S5_188-270-0_2005-07-08-06-13-21_HRG-1_A_MX_KK
    """
    match = re.search(r"_S\d_([0-9]{3}-[0-9]{3}-[0-9])_", scene_name)
    if match:
        pr = match.group(1)  # something like '188-270-0'
        path_str, row_str, shift_str = pr.split("-")
        return (int(path_str), int(row_str), int(shift_str))
    return (999999, 999999, 999999)  # fallback: large number to sort to end
