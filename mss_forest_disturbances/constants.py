"""Global constants and helper functions used throughout project.
"""
import os
import ee

import tensorflow as tf

LANDCOVER = "projects/sat-io/open-datasets/CA_FOREST_LC_VLCE2"
FIRE = "users/boothmanrylan/NTEMSCanada/forest_fire_1985-2020"
HARVEST = "users/boothmanrylan/NTEMSCanada/harvest_1985-2020"
ECOZONES = "users/boothmanrylan/forest_dominated_ecozones"

PROJECTION = "EPSG:3978"
SCALE = 60

MAX_ELEV = 3000  # few points in Canada are higher than this

LANDCOVER_CLASSES = {
    0: "Unclassified",
    20: "Water",
    31: "Snow/Ice",
    32: "Rock/Rubble",
    33: "Exposed/Barren land",
    40: "Bryoids",
    50: "Shrubs",
    80: "Wetland",
    81: "Wetland-treed",
    100: "Herbs",
    210: "Coniferous",
    220: "Broadleaf",
    230: "Mixedwood",
}

FOREST_CLASSES = [81, 210, 220, 230]

LANDCOVER_PALETTE = [
    "#686868",
    "#3333ff",
    "#ccffff",
    "#cccccc",
    "#996633",
    "#ffccff",
    "#ffff00",
    "#993399",
    "#9933cc",
    "#ccff33",
    "#006600",
    "#00cc00",
    "#cc9900",
]

LANDCOVER_REMAP = [
    [0, 20, 31, 32, 33, 40, 50, 80, 81, 100, 210, 220, 230],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
]

REMAPPED_LANDCOVER_CLASSES = {
    k2: LANDCOVER_CLASSES[k1] for k1, k2 in zip(*LANDCOVER_REMAP)
}

CLASSES = {
    0: "None",
    1: "NonForest",
    2: "Forest",
    3: "Water",
    4: "PreviousFire",
    5: "Fire",
    6: "PreviousHarvest",
    7: "Harvest",
    8: "Cloud",
    9: "CloudShadow",
}

CLASS_PALETTE = [
    "white",
    "black",
    "gold",
    "darkCyan",
    "darkOrange",
    "red",
    "orchid",
    "purple",
    "cornsilk",
    "dimGrey",
]

NUM_CLASSES = 10

CLASS_VIS = {"min": 1, "max": 9, "palette": CLASS_PALETTE}

JUL_1 = 172  # approximate day of year for July 1st
SEP_30 = 282  # approximate day of year for Sep. 30th
DOY_RANGE = [JUL_1, SEP_30]

FIRST_MSS_YEAR = 1972
LAST_MSS_YEAR = 1995

FIRST_LANDCOVER_YEAR = 1984
FIRST_DISTURBANCE_YEAR = 1985

_BANDS = ["nir", "red_edge", "red", "green", "tca", "ndvi"]
_HISTORICAL_BANDS = ["historical_" + x for x in _BANDS]
BANDS = _BANDS + ["dem"] + _HISTORICAL_BANDS

PROJECT = "api-project-269347469410"
BUCKET = "gs://rylan-mssforestdisturbances/"
LOCATION = "us-central1"
ASSET_PATH = os.path.join("projects", PROJECT, "assets", "rylan-mssforestdisturbances")

EXPORT_PATCH_SIZE = 512
PATCH_SIZE = 128
OVERLAP = 8

# this is necessary in order for every pixel to be represented exactly once in
# the test dataset
msg = "Model input size must perfectly divide patch export size."
assert EXPORT_PATCH_SIZE % PATCH_SIZE == 0, msg

HIGH_VOLUME_ENDPOINT = "https://earthengine-highvolume.googleapis.com"
MAX_REQUESTS = 20

DOCKER_IMAGE_DIR = f"{LOCATION}-docker.pkg.dev/{PROJECT}/dataflow-containers/"
DOCKER_IMAGE_URI = DOCKER_IMAGE_DIR + "mss_forest_disturbances.dockerfile:1.0"

MAX_DOY = 110
NUM_ECOZONES = 10

DEFAULT_PARSE_OPTIONS = {
    "size": EXPORT_PATCH_SIZE,
    "bands": BANDS,
    "label": "label",
    "num_classes": NUM_CLASSES,
    "integer_metadata": ["doy", "ecozone"],
    "float_metadata": ["lat", "lon"],
}

DEFAULT_MODEL_OPTIONS = {
    "input_shape": (PATCH_SIZE, PATCH_SIZE, len(BANDS)),
    "filters": [32, 64, 128, 256],
    "kernels": [5, 3, 3, 3],
    "dilation_rates": [1, 2, 4, 4],
    "first_downstack_inputs": len(_BANDS) + 1,
    "integer_metadata": ["doy", "ecozone"],
    "max_integer_metadata_values": [MAX_DOY, NUM_ECOZONES],
    "float_metadata": ["lat", "lon"],
}

BATCH_SIZE = 32

"""
Penalize errors between current and previous disturbances (of the same type)
less than other errors.
Penalize errors between non forest and distrubances (of any type
current/previous) less than other errors.

Based on:
https://discuss.pytorch.org/t/own-loss-function-for-multi-class-classifikation/115448/2
"""
_label_smoothing_matrix = []
for i in range(NUM_CLASSES):
    if i in [0, 2, 3, 8, 9]:
        _label_smoothing_matrix.append(tf.one_hot(i, NUM_CLASSES))
    elif i == 1:
        _label_smoothing_matrix.append(
            tf.constant([0.00, 0.80, 0.00, 0.00, 0.05, 0.05, 0.05, 0.05, 0.00, 0.00])
        )
    elif i == 4:
        _label_smoothing_matrix.append(
            tf.constant([0.00, 0.05, 0.00, 0.00, 0.85, 0.10, 0.00, 0.00, 0.00, 0.00])
        )
    elif i == 5:
        _label_smoothing_matrix.append(
            tf.constant([0.00, 0.05, 0.00, 0.00, 0.10, 0.85, 0.00, 0.00, 0.00, 0.00])
        )
    elif i == 6:
        _label_smoothing_matrix.append(
            tf.constant([0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.85, 0.10, 0.00, 0.00])
        )
    elif i == 7:
        _label_smoothing_matrix.append(
            tf.constant([0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.10, 0.85, 0.00, 0.00])
        )
LABEL_SMOOTHING_MATRIX = tf.stack(_label_smoothing_matrix)


def get_default_projection():
    return ee.Projection(PROJECTION).atScale(SCALE)
