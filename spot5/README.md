# SPOT5 HRS Data Processing

A Python toolkit for processing SPOT5 HRS (High Resolution Stereoscopic) satellite imagery data.

## Overview

This repository provides tools for:
- Parsing SPOT5 HRS scenes 
- Managing scene metadata
- Filtering scenes by geographic regions
- Generating footprints and previews
- Organizing image pairs

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RSE-UZH/spot5.git
   cd spot5
   ```
2. Create a virtual environment with `conda` or `venv`.
   You can skip this step if you already have a virtual environment where you want to install this library):
   
   ```bash
    conda create -n spot5 python=3.11
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
   
   Alternatively, you can create a virtual environment with venv:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the pacakage:
   ```bash
   pip install -e .
   ```

## Usage
The main interface for working with SPOT5 data is the SpotDatastore class. 
For detailed examples and workflows, see the Jupyter notebooks in the notebooks/ directory.

Here's a quick overview of common workflows:

Parsing Scenes
```python
from pathlib import Path
from spot5 import SpotDatastore

# Create a new datastore
ds = SpotDatastore()

# Parse scenes from a directory
ds.parse_scenes_from_dir(
    data_dir=Path("scenes"), 
    compute_stats=True, 
    reference_polygon="rgi7_alps.geojson"
)

# Save datastore to disk
ds.to_pickle(Path("pair_sorted/datastore.pkl"))
```

Loading an Existing Datastore
```python
from pathlib import Path
from spot5 import SpotDatastore

# Load a previously saved datastore
ds = SpotDatastore.from_pickle(Path("pair_sorted/datastore.pkl"))
```

Working with Footprints
```python
# Parse and save footprints
ds.parse_footprints()
ds.save_footprints_to("footprints.geojson")

# Save individual footprint files
ds.save_footprints_to("footprints_dir", single_files=True)
```


Filtering Scenes by Geographic Area
```python
import geopandas as gpd

# Load a polygon to filter by
gdf = gpd.read_file("polygon.geojson")

# Filter the datastore
ds.filter_by_geodataframe(gdf, inplace=True, target_crs="EPSG:32632")
```

Clusterize scenes based on proximity
```python
# Compute clusters of scenes
ds.compute_clusters(
    cluster_distance=5000,
    min_cluster_size=5,
    min_footprint_size=10000,
    max_footprint_size=1000000,
    target_crs="EPSG:32632",
)
```
Organizing and Previewing Data
```python
# Save pairs to a directory
ds.save_pairs_to_dir("pair_sorted")

# Generate preview images
ds.generate_previews("previews")

# Sort pairs by strip
ds.sort_by("strip")
```
