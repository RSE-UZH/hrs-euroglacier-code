#!/usr/bin/env python

# This script matches HRS scenes into pairs, preparing them for running through the dem_production pipeline.

import os
import spot5

root_dir = "PATH/TO/MAIN/PROCESSING/FOLDER"
scene_dir = root_dir + "hrs_all"
sorted_dir = root_dir + "hrs_pairs"
rgi_path = root_dir + "/PATH/TO/GLACIERS/VECTOR/FILE"


if not os.path.isdir(sorted_dir):
    os.makedirs(sorted_dir, exist_ok = True)


ds = spot5.SpotDatastore()

ds.parse_scenes_from_dir(data_dir=scene_dir, compute_stats=True, reference_polygon=rgi_path)

ds.sort_by("timestamp")

ds.to_pickle(sorted_dir + "/datastore.pkl")

ds.save_pairs_to_dir(sorted_dir)
