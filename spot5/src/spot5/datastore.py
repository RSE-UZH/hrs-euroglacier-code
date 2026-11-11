import json
import logging
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import seaborn as sns
import shapely
from joblib import Parallel, delayed, dump, load
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, date2num
from rasterio.control import GroundControlPoint
from rasterio.plot import show
from rasterio.transform import from_gcps
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from spot5.spatial import create_clusters
from spot5.spot5_utils import (
    extract_cloudcover,
    extract_footprint,
    extract_metadata,
    extract_scene_id,
    parse_strip_id,
)
from spot5.zip import unzip_scenes

logger = logging.getLogger("spot5")


def _repr_formatter(name, value):
    if isinstance(value, Path):
        return value.name
    elif isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(value, gpd.GeoDataFrame):
        return f"Bounds: {value.bounds.values}"
    elif isinstance(value, shapely.geometry.Point):
        return f"Point: {value.coords[:]}"
    else:
        return value


# TODO: not used yet
class SensorType(Enum):
    HRS1 = "HRS-1"
    HRS2 = "HRS-2"
    HRG = "HRG-1"


class SortBy(Enum):
    TIMESTAMP = "timestamp"
    CLOUD_COVER = "cloud_cover"
    CENTROID = "centroid"
    STRIP = "strip"


@dataclass(kw_only=True)
class SceneMetadata:
    path: Path = field(repr=False)
    save_footprint: bool = field(default=False, repr=False)
    name: str = field(init=False, repr=True)
    timestamp: datetime | None = field(init=False, default=None, repr=True)
    cloud_cover: float | None = field(init=False, default=None, repr=True)
    metadata: pd.Series | None = field(init=False, default=None, repr=False)
    footprint: gpd.GeoDataFrame | None = field(init=False, default=None, repr=False)
    centroid: shapely.geometry.Point | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self):
        self.name = self.path.name
        self.get_metadata()

    def get_metadata(self):
        try:
            self.timestamp = datetime.strptime(
                self.path.name[21:40], "%Y-%m-%d-%H-%M-%S"
            )
        except Exception as e:
            logger.error(f"Error parsing timestamp for {self.path}: {str(e)}")

        metadata_path = self.path / "SCENE01" / "METADATA.DIM"

        try:
            self.metadata = extract_metadata(metadata_path)
        except Exception as e:
            logger.error(f"Error extracting metadata for {self.path}: {str(e)}")

        try:
            poly, crs = extract_footprint(metadata_path)
            self.footprint = gpd.GeoDataFrame(
                {**self.metadata.to_dict(), "geometry": [poly]}, crs=crs
            )
            self.centroid = self.footprint.union_all().centroid
            if self.save_footprint and self.footprint is not None:
                footprint_path = self.path / "footprint.geojson"
                self.footprint.to_file(footprint_path, driver="GeoJSON")
        except Exception as e:
            logger.error(f"Error extracting footprint for {self.path}: {str(e)}")

        try:
            self.cloud_cover = extract_cloudcover(self.path)
        except Exception as e:
            logger.error(f"Error extracting cloud cover for {self.path}: {str(e)}")
            self.cloud_cover = None

    def __repr__(self):
        """Just like the default __repr__ but supports reformatting values."""
        fields = (
            f"{name}={_repr_formatter(name, value)}"
            for field in self.__dataclass_fields__.values()
            if field.repr
            for name, value in ((field.name, self.__getattribute__(field.name)),)
        )
        return f"{self.__class__.__name__}({', '.join(fields)})"

    @classmethod
    def from_dict(data: dict):
        return SceneMetadata(
            path=Path(data["path"]),
            timestamp=datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S"),
            cloud_cover=data.get("cloud_cover"),
            metadata=pd.Series(data["metadata"]),
            footprint=gpd.GeoDataFrame(data["footprint"]),
            centroid=shapely.geometry.Point(data["centroid"]),
        )

    def validate(
        self,
        min_scene_width: float = 0.5,  # degrees
        max_scene_width: float = 10,  # degrees
        reference_polygon: gpd.GeoDataFrame | None = None,
        min_intersection_area_km2: float | None = None,
    ):
        return validate_scene(
            self,
            min_scene_width,
            max_scene_width,
            reference_polygon,
            min_intersection_area_km2,
        )


@dataclass(kw_only=True)
class SpotPair:
    id: int = field(init=False, default=None)
    name: str = field(init=False, default=None)
    timestamp: datetime = field(repr=True)
    hrs1: SceneMetadata = field(repr=True)
    hrs2: SceneMetadata = field(repr=True)

    def __repr__(self):
        """Just like the default __repr__ but supports reformatting values."""
        fields = (
            f"{name}={_repr_formatter(name, value)}"
            for field in self.__dataclass_fields__.values()
            if field.repr
            for name, value in ((field.name, self.__getattribute__(field.name)),)
        )
        return f"{self.__class__.__name__}({', '.join(fields)})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp,
            "hrs1": self.hrs1.__dict__,
            "hrs2": self.hrs2.__dict__,
        }


class SpotDatastore:
    def __init__(self):
        self._last_id: int = -1
        self._pairs: dict[int, SpotPair] = {}
        self.data_dir: Path = None
        self.footprints: gpd.GeoDataFrame = None

    def __len__(self):
        return len(self._pairs)

    def __iter__(self):
        return iter(self._pairs.values())

    def __getitem__(self, key: int):
        return self._pairs[key]

    def __setitem__(self, key: int, value: SpotPair):
        self._pairs[key] = value

    def __delitem__(self, key: int):
        del self._pairs[key]

    def __contains__(self, key: int):
        return key in self._pairs

    def __repr__(self):
        return f"SpotDatastore with {len(self)} pairs"

    def __str__(self):
        return self.__repr__()

    @property
    def pairs(self):
        return list(self._pairs.values())

    def parse_scenes_from_dir(
        self,
        data_dir: Path,
        *,
        sort_by: str = "timestamp",
        recursive: bool = False,
        reference_polygon: Path = None,
        min_intersection_area_km2: float = None, # Minimum intersection area between each scene and the polygons of interest
        min_overlap_area_km2: float = 0, # Minimum overlap between the HRS-1 and HRS-2 scenes to be suitable for a pair (0: any overlap)
        compute_stats: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Repository directory not found: {self.data_dir}")

        # Parse HRS scenes and match HRS-1 with HRS-2 data
        logger.info("Parsing SPOT5 HRS scenes...")
        pair_list = parse_unsorted_scenes(
            image_dir=self.data_dir,
            recursive=recursive,
            reference_polygon=reference_polygon,
            min_intersection_area_km2=min_intersection_area_km2,
            min_overlap_area_km2=min_overlap_area_km2,
        )

        # Sort pairs based on the specified criterion
        pair_sorted = sort_pairs(pair_list, sort_by)

        # Add pairs to the datastore
        for pair in pair_sorted:
            self.add_pair(pair)
        logger.info(f"Found {len(self._pairs)} pairs")

        # Parse footprints and create clusters
        logger.info("Parsing footprints and creating clusters...")
        self.parse_footprints()
        logger.info("Footprints parsed and clusters created")

        # Save pairs with metadata to json file
        self.to_json(self.data_dir / "spot5_hrs_pairs.json")
        logger.info(
            f"Pairs saved to JSON file {self.data_dir / 'spot5_hrs_pairs.json'}"
        )

        # Compute statistics for the dataset
        if compute_stats:
            self.compute_stats(output_dir=self.data_dir)

    def load_sorted_pairs_from_dir(
        self,
        pairs_dir: Path,
        *,
        pair_name_pattern: str = None,
        sort_by: str = "timestamp",
        compute_stats: bool = False,
    ):
        """
        Load pre-sorted pairs from a directory structure where each subdirectory contains a pair.

        Args:
            pairs_dir (Path): Directory containing subdirectories for each pair.
            pair_name_pattern (str, optional): Regular expression pattern to extract pair ID from directory name.
                Default is to use the ifolder name.
            sort_by (str, optional): Criterion for sorting loaded pairs. Defaults to "timestamp".

        Returns:
            SpotDatastore: The datastore object with loaded pairs.
        """

        def _process_pair_dir(pair_dir, pair_name_pattern):
            """Helper function to process a single pair directory."""
            # Extract pair ID from directory name using the pattern
            pair_id_match = re.search(pair_name_pattern, pair_dir.name)
            pair_id = int(pair_id_match.group(1)) if pair_id_match else None

            # Find HRS-1 and HRS-2 scenes in this directory
            hrs1_dirs = list(pair_dir.glob("*HRS-1*"))
            hrs2_dirs = list(pair_dir.glob("*HRS-2*"))

            if not hrs1_dirs or not hrs2_dirs:
                logger.warning(
                    f"Missing HRS scene in {pair_dir}. Found HRS-1: {bool(hrs1_dirs)}, HRS-2: {bool(hrs2_dirs)}"
                )
                return None

            if len(hrs1_dirs) > 1 or len(hrs2_dirs) > 1:
                logger.warning(
                    f"Multiple HRS scenes found in {pair_dir}. Using first found."
                )

            # Create SceneMetadata objects
            try:
                hrs1_scene = SceneMetadata(path=hrs1_dirs[0])
                hrs2_scene = SceneMetadata(path=hrs2_dirs[0])

                # Create SpotPair object
                # Use the HRS-1 timestamp as the pair timestamp
                spot_pair = SpotPair(
                    timestamp=hrs1_scene.timestamp, hrs1=hrs1_scene, hrs2=hrs2_scene
                )

                # Return the pair with its ID and name
                return {
                    "pair": spot_pair,
                    "pair_id": pair_id,
                    "dir_name": pair_dir.name,
                }
            except Exception as e:
                logger.error(f"Error processing pair {pair_dir}: {str(e)}")
                return None

        self.data_dir = Path(pairs_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Pairs directory not found: {self.data_dir}")

        logger.info(f"Loading sorted pairs from {self.data_dir}...")

        # Get all subdirectories in the pairs directory
        pair_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not pair_dirs:
            raise ValueError(f"No pair directories found in {self.data_dir}")

        # Define the regex pattern for extracting pair ID if not provided
        if pair_name_pattern is None:
            pair_name_pattern = r"^(\d+)"  # Default pattern: integer at start

        with logging_redirect_tqdm([logger]):
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(_process_pair_dir)(pair_dir, pair_name_pattern)
                for pair_dir in tqdm(pair_dirs, desc="Loading pairs")
            )

        # Store the pairs in the datastore
        unsorted_pairs = [res["pair"] for res in results if res is not None]
        sorted_pairs = sort_pairs(unsorted_pairs, sort_by)
        self._pairs = {}
        for pair in sorted_pairs:
            self.add_pair(pair)

        # Parse footprints
        logger.info("Parsing footprints...")
        self.parse_footprints()

        # Save pairs to JSON file
        json_path = self.data_dir / "spot5_hrs_pairs.json"
        self.to_json(json_path)
        logger.info(f"Pairs saved to JSON file {json_path}")

        # Compute statistics for the dataset
        if compute_stats:
            self.compute_stats(output_dir=self.data_dir)

    def sort_by(self, sort_by: str, *, keep_ids: bool = False):
        """Sort pairs by a specific key"""
        sorted_pairs = sort_pairs(self.pairs, sort_by)
        self._pairs = {pair.id: pair for pair in sorted_pairs}
        if not keep_ids:
            self.reset_ids()

    def reset_ids(self):
        """Reset IDs of all pairs. Assign new IDs and names to all pairs"""
        old_pairs = self._pairs.copy()
        self._pairs = {}
        self._last_id = -1
        for pair in old_pairs.values():
            pair.id = None
            pair.name = None
            self.add_pair(pair)

        # Keep the footprints table in sync with the new ids/names (row order and cluster_id are unchanged)
        if self.footprints is not None:
            pairs = [old_pairs[i] for i in self.footprints["pair_id"]]
            self.footprints["pair_id"] = [p.id for p in pairs]
            self.footprints["name"] = [p.name for p in pairs]

    def add_pair(self, pair: SpotPair):
        if pair.id is not None and pair.id in self._pairs:
            raise ValueError(f"Pair with ID {pair.id} already exists")

        # Assign new ID and name to the pair
        self._last_id += 1
        pair.id = self._last_id
        pair.name = get_pair_name(pair, pair.id)

        # Add pair to the datastore
        self._pairs[pair.id] = pair

    def remove_pair(self, key: int):
        self._pairs.pop(key, None)

    def get_pair_by_id(self, key: int):
        return self._pairs.get(key)

    def get_pair_names(self):
        return list(p.name for p in self._pairs.values())

    def get_pair_timestamps(self):
        return list(p.timestamp for p in self._pairs.values())

    def get_pair_by_name(self, pair_name: str):
        return self._pairs.get(pair_name)

    def get_pairs_by_timestamp(self, timestamp: datetime):
        return list(p for p in self._pairs.values() if p.timestamp == timestamp)

    def save_pairs_to_dir(
        self, output_dir: Path, *, overwrite: bool = False, workers: int = -1
    ):
        """Copy all sorted pairs to a new directory"""

        def _copy_pair(
            pair: SpotPair, output_dir: Path, overwrite: bool = False
        ) -> int:
            """Helper function to copy a single pair."""
            idx = pair.id
            pair_dir = output_dir / pair.name
            if not overwrite and pair_dir.exists():
                logger.debug(f"Pair {idx} already exists, skipping...")
                return 1
            pair_dir.mkdir(exist_ok=True)
            try:
                hrs1_path = pair.hrs1.path
                hrs2_path = pair.hrs2.path
                shutil.copytree(
                    hrs1_path, pair_dir / hrs1_path.name, dirs_exist_ok=True
                )
                shutil.copytree(
                    hrs2_path, pair_dir / hrs2_path.name, dirs_exist_ok=True
                )
                return 0
            except Exception as e:
                logger.error(f"Failed to copy pair {idx}: {e}")
                return -1

        logger.info(f"Copying {len(self)} pairs to {output_dir}...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        with logging_redirect_tqdm([logger]):
            results = Parallel(n_jobs=workers, backend="threading")(
                delayed(_copy_pair)(p, output_dir, overwrite) for p in tqdm(self.pairs)
            )
        successes = [i for i, res in enumerate(results) if res == 0]
        failures = [i for i, res in enumerate(results) if res == -1]
        skipped = [i for i, res in enumerate(results) if res == 1]
        if failures:
            logger.error(f"Failed to copy {len(failures)} pairs: {failures}")
        if skipped:
            logger.warning(f"Skipped {len(skipped)} pairs: {skipped}")
        logger.info(f"Successfully copied {len(successes)} pairs")

        self.save_footprints_to(output_dir / "footprints.geojson")

    def parse_footprints(
        self, grouping_threshold: float = 0.7, grouping_min_samples: int = 2
    ):
        data = [
            {
                "pair_id": pair.id,
                "name": pair.name,
                "timestamp": pair.timestamp,
                "hrs1_dataset_name": str(pair.hrs1.metadata["dataset_name"][0]),
                "hrs1_date": str(pair.hrs1.metadata["date"][0]),
                "hrs1_grid_reference": str(pair.hrs1.metadata["grid_reference"][0]),
                "hrs1_orbit": pair.hrs1.footprint["grid_reference"].values[0][:3],
                "hrs1_row": pair.hrs1.footprint["grid_reference"].values[0][3:7],
                "hrs1_shift_value": pair.hrs1.metadata["shift_value"][0],
                "hrs1_cloud_cover": pair.hrs1.cloud_cover,
                "hrs1_incidence_angle": pair.hrs1.metadata["incidence_angle"][0],
                "geometry": pair.hrs1.footprint.geometry.iloc[0]
                if pair.hrs1.footprint is not None and not pair.hrs1.footprint.empty
                else None,
            }
            for pair in self._pairs.values()
        ]

        # Filter out entries with None geometries
        valid_data = [d for d in data if d["geometry"] is not None]

        if not valid_data:
            logger.warning("No valid footprints found in the dataset")
            # Initialize an empty GeoDataFrame with the correct columns
            self.footprints = gpd.GeoDataFrame(
                columns=[
                    "pair_id",
                    "name",
                    "timestamp",
                    "hrs1_dataset_name",
                    "hrs1_date",
                    "hrs1_grid_reference",
                    "hrs1_orbit",
                    "hrs1_row",
                    "hrs1_shift_value",
                    "hrs1_cloud_cover",
                    "hrs1_incidence_angle",
                    "geometry",
                ]
            )
            return

        # Create a GeoDataFrame from the valid data
        crs = next(
            (
                pair.hrs1.footprint.crs
                for pair in self._pairs.values()
                if pair.hrs1.footprint is not None
            ),
            None,
        )
        self.footprints = gpd.GeoDataFrame(valid_data, crs=crs, geometry="geometry")

        # Create clusters based on the grouping threshold
        self.cluster_footprints(
            grouping_threshold=grouping_threshold,
            grouping_min_samples=grouping_min_samples,
        )

    def cluster_footprints(
        self,
        *,
        grouping_threshold: float = 0.7,
        grouping_min_samples: int = 2,
    ) -> None:
        """Cluster footprints based on their geometry"""
        if self.footprints is None or self.footprints.empty:
            logger.warning("No footprints available for clustering")
            return None

        # Create clusters based on the grouping threshold
        clusters = create_clusters(
            self.footprints,
            threshold=grouping_threshold,
            min_samples=grouping_min_samples,
        )
        self.footprints["cluster_id"] = clusters
        logger.debug(f"Clusters: {Counter(clusters)}")

    def filter_by_geodataframe(
        self,
        geodataframe: gpd.GeoDataFrame,
        *,
        target_crs: str = None,
        keep_ids: bool = True,
        inplace: bool = False,
    ):
        """Filter footprints by a GeoDataFrame of polygons

        Args:
            geodataframe: GeoDataFrame containing polygons to filter by
            target_crs: Optional target CRS to use for the overlay computation. If None, the best UTM zone is estimated. Default is None.
            keep_ids: Whether to keep original pair IDs. If False, new IDs will be assigned. Default is True.
            inplace: Whether to modify the datastore in-place. If False, a new SpotDatastore is returned. Default is False.

        Returns:
            If inplace=False, returns a new SpotDatastore with filtered data
            If inplace=True, returns None and modifies the current datastore
        """
        if not isinstance(geodataframe, gpd.GeoDataFrame):
            raise ValueError("geodataframe must be a GeoDataFrame")

        if self.footprints is None or self.footprints.empty:
            logger.warning("No footprints available for filtering")
            return None

        # Save original CRS to convert back at the end
        original_crs = self.footprints.crs

        # Determine the target projected CRS for overlay operations
        if target_crs is not None:
            # Use the specified target CRS
            projected_crs = target_crs
        elif self.footprints.crs.is_projected:
            # Already in a projected CRS, use that
            projected_crs = self.footprints.crs
        else:
            # Estimate a suitable UTM CRS from the footprints
            projected_crs = self.footprints.estimate_utm_crs()
            logger.debug(f"Estimated UTM CRS for overlay: {projected_crs}")

        # Project both datasets to the target CRS
        footprints_projected = self.footprints.to_crs(projected_crs)
        geodataframe_projected = geodataframe.to_crs(projected_crs)

        # Perform the intersection in the projected CRS
        filtered_footprints = footprints_projected[
            footprints_projected.intersects(geodataframe_projected.unary_union)
        ]

        if len(filtered_footprints) == 0:
            logger.warning(
                f"No footprints intersect with the geodataframe. Check that the CRS of the geodataframe matches the footprints (footprints CRS: {self.footprints.crs})"
            )
            return None

        # Filter the original footprints based on the filtered ones
        new_pairs = []
        for id in self._pairs:
            if id in filtered_footprints["pair_id"].values:
                new_pairs.append(self._pairs[id])

        # Convert the filtered footprints back to the original CRS
        filtered_footprints = filtered_footprints.to_crs(original_crs)

        if inplace:
            # Update the existing datastore
            if keep_ids:
                self._pairs = {pair.id: pair for pair in new_pairs}
                self._last_id = len(new_pairs) - 1
                self.footprints = filtered_footprints
            else:
                self._last_id = -1
                self._pairs = {}
                for pair in new_pairs:
                    pair.id = None
                    pair.name = None
                    self.add_pair(pair)

            return None

        else:
            # Create a new SpotDatastore with the filtered footprints
            filt_ds = SpotDatastore()
            filt_ds.data_dir = self.data_dir
            if keep_ids:
                filt_ds._pairs = {pair.id: pair for pair in new_pairs}
                filt_ds._last_id = len(new_pairs) - 1
                filt_ds.footprints = filtered_footprints

            else:
                for seq_id, pair in enumerate(new_pairs):
                    # Update the id in the footprints geodataframe
                    filtered_footprints.loc[
                        filtered_footprints["pair_id"] == pair.id, "pair_id"
                    ] = seq_id

                    # Add the pair to the new datastore
                    filt_ds.add_pair(pair)
                filt_ds.footprints = filtered_footprints

            # Save pairs with metadata to json file
            path = self.data_dir / "spot5_hrs_pairs.json"
            filt_ds.to_json(path)
            logger.info(f"Filtered pairs saved to {path}")
            return filt_ds

    def _build_geojson_footprints(self) -> gpd.GeoDataFrame:
        """Build the GeoDataFrame written to footprints.geojson.

        One row per pair (rows and cluster_id taken from self.footprints, which is left untouched).
        The geometry is the intersection of the HRS-1 and HRS-2 footprints, since pairs do not
        always overlap fully. HRS-1 and HRS-2 fields are generated by the same code, so they are
        identical in type and order; the sun angles are HRS-1 only. cluster_id is still computed
        on the full HRS-1 footprints.
        """
        fp = self.footprints
        pairs = [self._pairs[i] for i in fp["pair_id"]]

        overlaps = [
            p.hrs1.footprint.geometry.iloc[0].intersection(
                p.hrs2.footprint.geometry.iloc[0]
            )
            for p in pairs
        ]

        def _area_km2(geom) -> float:
            # Same as in find_corresponding_pairs: area in the UTM zone estimated for the overlap
            g = gpd.GeoSeries([geom], crs=fp.crs)
            return g.to_crs(g.estimate_utm_crs()).area.iloc[0] / 1e6

        cols = {
            "pair_id": [f"{p.id:03d}" for p in pairs],  # same as in the pair folder names
            "name": [p.name for p in pairs],
            "overlap_area_km2": [round(_area_km2(g), 2) for g in overlaps],
            "cluster_id": fp["cluster_id"].values,
        }
        for key in ("hrs1", "hrs2"):
            scenes = [getattr(p, key) for p in pairs]
            meta = [s.metadata for s in scenes]
            grid = [m["grid_reference"][0] for m in meta]

            cols[f"{key}_scene_id"] = [
                extract_scene_id(s.path / "SCENE01" / "METADATA.DIM") for s in scenes
            ]
            # Name of the scene folder as copied into the pair folder (see save_pairs_to_dir)
            cols[f"{key}_folder_name"] = [s.name for s in scenes]
            cols[f"{key}_dataset_name"] = [m["dataset_name"][0] for m in meta]
            cols[f"{key}_datetime"] = [  # UTC
                f"{m['date'][0]}T{m['time'][0]}Z" for m in meta
            ]
            cols[f"{key}_grid_reference"] = grid
            cols[f"{key}_orbit"] = [g[:3] for g in grid]
            cols[f"{key}_row"] = [g[3:7] for g in grid]
            cols[f"{key}_shift_value"] = [m["shift_value"][0] for m in meta]
            cols[f"{key}_cloud_cover"] = [s.cloud_cover for s in scenes]
            cols[f"{key}_incidence_angle"] = [m["incidence_angle"][0] for m in meta]
            if key == "hrs1":  # Single set of sun angles, taken from HRS-1
                cols["hrs1_sun_azimuth"] = [float(m["sun_azimuth"][0]) for m in meta]
                cols["hrs1_sun_elevation"] = [float(m["sun_elevation"][0]) for m in meta]

        return gpd.GeoDataFrame(cols, geometry=overlaps, crs=fp.crs)

    def save_footprints_to(self, path: Path, *, single_files: bool = False):
        """Save the pair footprints (HRS-1/HRS-2 overlap) to a GeoJSON file, or the raw HRS-1 footprints to one file per pair"""

        # Always rebuild, so that footprints match the current pair ids/names
        self.parse_footprints()

        path = Path(path)
        if not single_files:
            path.parent.mkdir(exist_ok=True, parents=True)
            if path.suffix != ".geojson":
                path = path.with_suffix(".geojson")
            self._build_geojson_footprints().to_file(path, driver="GeoJSON")
            logger.info(f"Footprints saved to {path}")

        else:
            path.mkdir(exist_ok=True, parents=True)
            with logging_redirect_tqdm([logger]):
                for pair in tqdm(self.pairs, desc="Footprint"):
                    pair.hrs1.footprint.to_file(
                        path / f"{pair.name}_footprint.geojson", driver="GeoJSON"
                    )
            logger.info(f"Footprints saved to {path}")

    def to_json(self, path: Path):
        """Save pairs information to JSON file"""
        write_pairs_json(self._pairs, path)

    def compute_stats(self, output_dir: Path = None):
        """Compute statistics for the dataset"""
        if output_dir is None:
            output_dir = self.data_dir

        make_pair_timestamps_plot(
            self.pairs, output_path=output_dir / "spot5_hrs_scenes.png"
        )

    def generate_previews(
        self, preview_dir: Path, downsample_factor: float = 0.1, n_jobs: int = -1
    ):
        """Generate previews for all pairs"""
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(exist_ok=True)

        logger.info(f"Generating previews for {len(self.pairs)} pairs...")
        with logging_redirect_tqdm([logger]):
            Parallel(n_jobs=n_jobs)(
                delayed(create_preview)(
                    pair=pair,
                    preview_path=preview_dir / f"{pair.name}_preview.png",
                    downsample_factor=downsample_factor,
                )
                for pair in tqdm(self.pairs, desc="Preview")
            )

    @staticmethod
    def extract_zip(zip_dir: Path, extracted_dir: Path = None):
        """Extract ZIP files in the repository directory"""
        zip_dir = Path(zip_dir)
        if not zip_dir.exists():
            raise FileNotFoundError(f"ZIP directory not found: {zip_dir}")
        zip_files = list(zip_dir.glob("*.zip"))
        if not zip_files:
            logger.info("No ZIP files found in repository directory")
        else:
            extracted_dir = zip_dir if extracted_dir is None else Path(extracted_dir)
            unzip_scenes(zip_files, extracted_dir)

    def to_pickle(self, path: Path):
        """Save the datastore to a pickle file using joblib serialization"""
        dump(self, path)

    @classmethod
    def from_pickle(self, path: Path):
        """Load the datastore from a pickle file"""
        return load(path)


def get_scenes_from_dir(
    image_dir: Path,
    *,
    sensor_names: list[str] = None,
    recursive: bool = False,
) -> list[Path]:
    """
    Get a list of scene directories from the specified directory.

    Args:
        image_dir (Path): Directory containing scene folders.
        sensor_names (list[str]): List of sensor names to filter the directories. If None, ["HRS-1", "HRS-2"] is used. Defaults to None.
        recursive (bool): Whether to search recursively in subdirectories.

    Returns:
        list[Path]: List of scene directories.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{image_dir} is not a valid directory.")

    if sensor_names is None:
        sensor_names = ["HRS-1", "HRS-2"]

    # Find HRS-1 and HRS-2 directories
    scene_list = []
    for sensor in sensor_names:
        if recursive:
            scene_list.extend(list(image_dir.rglob(f"*{sensor}*")))
        else:
            scene_list.extend(list(image_dir.glob(f"*{sensor}*")))

        # Exclude zip files and preview directories
        if any(d.name.endswith(".zip") for d in scene_list):
            logger.warning(
                f"Found ZIP files in the directory {image_dir}. Skipping them. If you want to extract them, use the 'extract_zip' function."
            )
        scene_list = [d for d in scene_list if not d.name.endswith(".zip")]

        if not scene_list:
            raise FileNotFoundError(
                f"No {sensor} directories found in {image_dir}. Please check the input directory."
            )

    return scene_list


def parse_unsorted_scenes(
    image_dir: Path,
    *,
    sensor_names: list[str] = None,
    recursive: bool = False,
    reference_polygon: Path | None = None,
    min_intersection_area_km2: float | None = None,
    min_overlap_area_km2: float = 0,
    n_jobs: int = -1,
) -> list[SpotPair]:
    """
    Parse SPOT5 HRS scenes, decompress ZIP files if needed, read scene metadata,
    and optionally filter by intersection with a reference polygon.

    Args:
        image_dir (Path): Directory containing unsorted scene folders.
        sensor_names (list[str]): List of sensor names to filter the directories. If None, ["HRS-1", "HRS-2"] is used. Defaults to None.
        recursive (bool): Whether to search recursively in subdirectories. Defaults to False.
        reference_polygon (Optional[gpd.GeoDataFrame]): Polygon used to filter scenes. Defaults to None.
        min_intersection_area_km2 (Optional[float]): Minimum intersection area (scene <-> polygons) in km². Defaults to None.
        min_overlap_area_km2 (float): Minimum overlap area (HRS-1 scene <-> HRS-2 scene) in km². Defaults to 0 (any overlap).
        n_jobs (int): Number of parallel workers. Defaults to -1 (use all available cores).

    Returns:
        list[SpotPair]: List of matched pairs between HRS-1 and HRS-2 scenes.
    """

    def _parse_scene(scene_dir: Path) -> SceneMetadata | None:
        scene = SceneMetadata(path=scene_dir)
        if not scene.validate(
            reference_polygon=reference_polygon,
            min_intersection_area_km2=min_intersection_area_km2,
        ):
            return None
        return scene

    # Find HRS-1 and HRS-2 directories
    if sensor_names is None:
        sensor_names = ["HRS-1", "HRS-2"]

    # Validate the input directory
    if reference_polygon is not None:
        reference_polygon = gpd.read_file(reference_polygon)
        
    scene_list = get_scenes_from_dir(
        image_dir=image_dir,sensor_names=sensor_names, recursive=recursive)
    logger.info(f"Found {len(scene_list)} scenes")

    # Process HRS scenes to extract scene metadata and validate
    logger.info("Extracting HRS scene metadata and validate scenes...")
    with logging_redirect_tqdm([logger]):
        metadata = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_parse_scene)(d) for d in tqdm(scene_list, desc="Scene")
        )

    # Sort the results into HRS-1 and HRS-2 data based on the sensor name (excluding None values)
    scenes = {}
    for sensor in sensor_names:
        metadata = [m for m in metadata if m is not None]
        scenes[sensor] = [m for m in metadata if sensor in m.path.name]
    logger.info(
        f"Found {len(scenes['HRS-1'])} HRS-1 scenes and {len(scenes['HRS-2'])} HRS-2 scenes"
    )

    # Find matching pairs between HRS-1 and HRS-2 scenes
    logger.info("Finding corresponding pairs...")
    pairs = find_corresponding_pairs(scenes["HRS-1"], scenes["HRS-2"], intersect_thresh = min_overlap_area_km2*1e6)
    if len(pairs) == 0:
        logger.error("No matching pairs found")
        return None, None

    return pairs


def validate_scene(
    scene: SceneMetadata,
    min_scene_width: float = 0.5,  # degrees
    max_scene_width: float = 10,  # degrees
    reference_polygon: gpd.GeoDataFrame | None = None,
    min_intersection_area_km2: float | None = None,
) -> bool:
    """
    Validate the scene footprint for a single HRS scene.

    Args:
        scene (SceneMetadata): Scene metadata to be validated.
        min_scene_width (float): Minimum valid scene width in degrees. Defaults to 0.5.
        max_scene_width (float): Maximum valid scene width in degrees. Defaults to 10.
        reference_polygon (Optional[gpd.GeoDataFrame]): Polygon used to filter this scene.
        min_intersection_area_km2 (Optional[float]): Minimum intersection area in km².

    Returns:
        bool: True if the scene is valid, False otherwise.
    """
    # Check if the scene width is within the valid range
    bbox = scene.footprint.geometry[0].bounds
    scene_width = bbox[2] - bbox[0]
    if scene_width <= min_scene_width or scene_width >= max_scene_width:
        logger.warning(
            f"Scene {scene.path.name} covers invalid area (width: {scene_width})."
        )
        return False

    # Check if the scene intersects with the reference polygon
    if reference_polygon is not None:
        if scene.footprint is None or scene.footprint.empty:
            logger.warning(f"Scene {scene.path.name} has no footprint.")
            return False

        footprint_warp = scene.footprint.to_crs(reference_polygon.crs)
        intersection = footprint_warp.overlay(reference_polygon, how="intersection")

        if len(intersection) == 0:
            logger.warning(
                f"Scene {scene.path.name} does not intersect reference polygon."
            )
            return False
        elif len(intersection) > 1:
            logger.debug(
                f"Scene {scene.path.name} has multiple intersections: {len(intersection)}"
            )

        if min_intersection_area_km2 is not None:
            utm_crs = intersection.estimate_utm_crs()
            intersection_areas_km2 = intersection.to_crs(utm_crs).area.values / 1e6
            total_intersection = (
                sum(intersection_areas_km2)
                if len(intersection_areas_km2) > 1
                else intersection_areas_km2[0]
            )

            if total_intersection < min_intersection_area_km2:
                logger.warning(
                    f"Scene {scene.path.name} intersection area is too small: {total_intersection} km^2"
                )
                return False

    return True


def get_pair_name(pair: SpotPair, pair_id: int = None) -> str:
    """Extract pair name"""

    # Extract the name from the directory name up to the string "HRS"
    str = pair.hrs1.path.name.split("_HRS")[0]

    # Remove the time from the scene name
    split = str.split("-")[:-3]
    name = "-".join(split)

    if pair_id is not None:
        name = f"{pair_id:03d}_{name}"

    return name


def sort_pairs(
    pairs: list[SpotPair],
    sort_by: str | SortBy = SortBy.TIMESTAMP,
) -> list[SpotPair]:
    """
    Sort pairs by timestamp, strip number, or lat-lon, either by passing a string or a SortBy enum.

    Args:
        pairs (list[SpotPair]): List of pairs to be sorted.
        sort_by (str | SortBy): Criterion for sorting pairs. Defaults to "timestamp".

    Returns:
        list[SpotPair]: Sorted list of pairs.
    """
    # Convert to enum if it's a string
    if isinstance(sort_by, str):
        sort_by = SortBy(sort_by)

    if sort_by == SortBy.TIMESTAMP:
        return sorted(pairs, key=lambda x: x.timestamp)
    elif sort_by == SortBy.CLOUD_COVER:
        return sorted(pairs, key=lambda x: (x.hrs1.cloud_cover, x.timestamp))
    elif sort_by == SortBy.STRIP:
        return sorted(
            pairs,
            key=lambda x: (parse_strip_id(x.hrs1.name if x.hrs1 else ""), x.timestamp),
        )
    elif sort_by == SortBy.CENTROID:

        def _centroid_key(pair: SpotPair):
            point = pair.hrs1.centroid
            if point is None:
                return ((float("inf"), float("inf")), pair.timestamp)
            return ((point.y, point.x), pair.timestamp)

        return sorted(pairs, key=_centroid_key)
    else:
        raise ValueError(f"Unsupported sort_by: {sort_by}")


def find_corresponding_pairs(
    hrs1_data: list[SceneMetadata],
    hrs2_data: list[SceneMetadata],
    time_threshold: int = 1800,  # seconds
    intersect_thresh: float = 1e9,  # m^2
) -> list[SpotPair]:
    """Find matching pairs between HRS-1 and HRS-2 scenes.

    Args:
        hrs1_data (list[SceneMetadata]): List of HRS-1 scene metadata.
        hrs2_data (list[SceneMetadata]): List of HRS-2 scene metadata.
        time_threshold (int, optional): Time threshold in seconds for matching scenes. Defaults to 1800.
        intersect_thresh (float, optional): Intersection area threshold in square meters for matching scenes. Defaults to 1e9.

    Returns:
        SpotDatastore: Dataset of matched pairs.
    """

    # TODO: Parallelize this function

    pairs = []
    with logging_redirect_tqdm([logger]):
        for hrs1 in tqdm(hrs1_data, desc="Pair"):
            logger.debug(f"Processing scene {hrs1.path.name}...")

            # Find potential matches within time threshold
            potential_matches = [
                hrs2
                for hrs2 in hrs2_data
                if abs((hrs1.timestamp - hrs2.timestamp).total_seconds())
                < time_threshold
            ]

            if not potential_matches:
                logger.warning(
                    f"Scene {hrs1.path.name} - No potential matches found. The scene is orphan. Moving to the next one..."
                )
                continue

            # Find best match based on intersection area
            logger.debug(f"{len(potential_matches)} potential matches --")

            # Calculate intersection areas
            intersection_areas = np.zeros(len(potential_matches))
            for i, hrs2 in enumerate(potential_matches):
                intersection = hrs1.footprint.overlay(
                    hrs2.footprint, how="intersection"
                )
                if len(intersection) == 1:
                    utm_crs = intersection.estimate_utm_crs()
                    intersection_areas[i] = intersection.to_crs(utm_crs).area.values[0]
                elif len(intersection) > 1:
                    logger.error(
                        f"Scene {hrs1.path.name} - Intersection is not a single polygon! Please check manually."
                    )
                    continue
                else:
                    intersection_areas[i] = 0
            areas_km2 = [round(area / 1e6) for area in intersection_areas]
            logger.debug(f"intersection areas [Km2]: {', '.join(map(str, areas_km2))}")

            # Get the potential match with the largest intersection area
            pm_sel = np.argmax(intersection_areas)
            max_intersection = intersection_areas[pm_sel]
            if max_intersection <= intersect_thresh:
                logger.warning(
                    f"Scene {hrs1.path.name} - Intersection area between scenes footprint is below the threshold. Skipping scene..."
                )
                continue

            best_match = potential_matches[pm_sel]
            pairs.append(SpotPair(timestamp=hrs1.timestamp, hrs1=hrs1, hrs2=best_match))

        return pairs


def write_pairs_json(pairs: dict | list, output_file: Path):
    """Write pairs information to JSON file.

    Args:
        pairs (SpotDatastore): Dataset of pairs to be written.
        output_file (Path): Path to the output JSON file.
    """

    class PairEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d_%H:%M:%S")
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, gpd.GeoDataFrame):
                return f"Bounds: {obj.bounds.values}"
            if isinstance(obj, shapely.geometry.Point):
                return f"Point: {obj.coords[:]}"
            if isinstance(obj, shapely.geometry.Polygon):
                return f"Polygon: {obj.bounds.values}"
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, np.ndarray):
                return obj.tolist()

            return super().default(obj)

    if isinstance(pairs, dict):
        formatted_pairs = [pair.to_dict() for pair in pairs.values()]
    elif isinstance(pairs, list):
        formatted_pairs = [pair.to_dict() for pair in pairs]

    with open(output_file, "w") as f:
        json.dump(formatted_pairs, f, cls=PairEncoder, indent=2)


def get_pairs_from_json(input_file: Path, base_dir: Path = None) -> SpotDatastore:
    """Read pairs information from JSON file and reconstruct a SpotDatastore object.

    The JSON file is expected to have a structure as written by write_pair_json.

    Args:
        input_file (Path): Path to JSON file containing pairs information.
        pair_dir (Path): Base directory to prepend to relative scene paths. If None, the current working directory is used. Defaults to None.

    Returns:
        SpotDatastore: Dataset of reconstructed SpotPair objects.
    """

    def _reconstruct_pair(pair_info: dict) -> SpotPair:
        # Parse the pair's overall metadata
        pair_timestamp = datetime.strptime(pair_info["timestamp"], "%Y-%m-%d_%H:%M:%S")

        # Reconstruct HRS1 scene
        hrs1 = pair_info["hrs1"]
        hrs1_path = base_dir / hrs1["path"]
        scene1 = SceneMetadata(path=hrs1_path)

        # Reconstruct HRS2 scene
        hrs2 = pair_info["hrs2"]
        hrs2_path = base_dir / hrs2["path"]
        scene2 = SceneMetadata(path=hrs2_path)

        return SpotPair(timestamp=pair_timestamp, hrs1=scene1, hrs2=scene2)

    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input JSON file not found: {input_file}")

    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    with open(input_file) as f:
        pairs_json = json.load(f)

    with logging_redirect_tqdm([logger]):
        pairs = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_reconstruct_pair)(pair)
            for pair in tqdm(pairs_json, desc="Reconstructing pairs")
        )
    ds = SpotDatastore()
    for pair in pairs:
        ds.add_pair(pair)

    return ds


def make_pair_timestamps_plot(
    pairs: list[SpotPair], output_path: str = "spot_hrs_scenes.png"
):
    """Create an enhanced visualization of scene acquisition timestamps.

    Args:
        pairs (SpotDatastore): Dataset of pairs to be plotted.
        output_path (str, optional): Path to save the output plot. Defaults to "spot_hrs_scenes.png".
    """

    import warnings

    # Add deprecated warning
    warnings.warn(
        "make_pair_timestamps_plot is deprecated and will be removed in a future version. Use functions from the plots module instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Get dates and counts
    dates = [p.timestamp.date() for p in pairs]
    daily_counts = Counter(dates)
    dates_numeric = date2num(dates)

    # Create figure with single plot
    fig, ax1 = plt.subplots(figsize=(10, 4))
    fig.suptitle("SPOT HRS Scene Pairs Distribution", fontsize=14, y=0.95)

    # Create secondary axis for density
    ax2 = ax1.twinx()

    # Scatter plot on primary axis
    x = list(daily_counts.keys())
    y = list(daily_counts.values())
    ax1.scatter(x, y, s=40, alpha=0.6, c=y, cmap="viridis")  # s=[v * 20 for v in y],

    # Density plot on secondary axis
    sns.kdeplot(data=dates_numeric, ax=ax2, fill=True, color="lightblue", alpha=0.05)

    # Format primary axis (scatter)
    ax1.set_ylabel("Scenes per Day", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Format secondary axis (density)
    ax2.set_ylabel("Density", color="lightblue")
    ax2.tick_params(axis="y", labelcolor="lightblue")

    # Format x-axis (shared)
    ax1.xaxis.set_major_locator(MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax1.xaxis.set_minor_locator(MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Add grid
    ax1.grid(visible=True, alpha=0.3)
    ax1.grid(visible=True, which="minor", axis="x", linestyle="--", alpha=0.1)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_preview(pair: SpotPair, preview_path: Path, downsample_factor: float = 0.1):
    """Create low-res preview for a single pair

    Args:
        pair: SpotPair containing pair information
        preview_path: Path to save the preview image
        downsample_factor: Factor to downsample image (0-1), e.g. 0.1 = 10x reduction
    """

    def scale_gcps(gcps, downsample_factor: float):
        """Scale GCP pixel coordinates by downsample factor"""
        scaled_gcps = []
        for gcp in gcps:
            scaled_gcp = GroundControlPoint(
                row=gcp.row * downsample_factor,
                col=gcp.col * downsample_factor,
                x=gcp.x,  # lon stays same
                y=gcp.y,  # lat stays same
                z=gcp.z if gcp.z is not None else None,
                id=gcp.id if hasattr(gcp, "id") else None,
                info=gcp.info if hasattr(gcp, "info") else None,
            )
            scaled_gcps.append(scaled_gcp)
        return scaled_gcps

    preview_path = Path(preview_path)
    preview_path.parent.mkdir(exist_ok=True, parents=True)

    hrs1_image = pair.hrs1.path / "SCENE01" / "IMAGERY.TIF"

    with rio.open(hrs1_image) as src:
        gcps, gcp_crs = src.gcps

        # Create transformation from GCPs
        tform = from_gcps(gcps)

        # Read the data
        data = src.read(1)  # Read first band

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plot with correct georeferencing
        show(data, transform=tform, cmap="gray", ax=ax)

        fig.savefig(preview_path, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.close(fig)

    return preview_path


if __name__ == "__main__":
    data_dir = Path("scenes")

    ds = SpotDatastore()
    ds.parse_scenes_from_dir(data_dir=data_dir, compute_stats=True)
    print(ds.pairs)
