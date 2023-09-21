"""Global constants and helper functions used throughout project.
"""

import ee

LANDCOVER = 'projects/sat-io/open-datasets/CA_FOREST_LC_VLCE2'
FIRE = 'users/boothmanrylan/NTEMSCanada/forest_fire_1985-2020'
HARVEST = 'users/boothmanrylan/NTEMSCanada/harvest_1985-2020'
ECOZONES = 'users/boothmanrylan/forest_dominated_ecozones'

PROJECTION = 'EPSG:3978'
SCALE = 60

MAX_ELEV = 3000  # few points in Canada are higher than this

LANDCOVER_CLASSES = {
    0: "Unclassified", 20: "Water", 31: "Snow/Ice", 32: "Rock/Rubble",
    33: "Exposed/Barren land", 40: "Bryoids", 50: "Shrubs", 80: "Wetland",
    81: "Wetland-treed", 100: "Herbs", 210: "Coniferous", 220: "Broadleaf",
    230: "Mixedwood"
}

FOREST_CLASSES = [81, 210, 220, 230]

LANDCOVER_PALETTE = [
    "#686868", "#3333ff", "#ccffff", "#cccccc", "#996633", "#ffccff", "#ffff00",
    "#993399", "#9933cc", "#ccff33", "#006600", "#00cc00", "#cc9900"
]

LANDCOVER_REMAP = [
    [0, 20, 31, 32, 33, 40, 50, 80, 81, 100, 210, 220, 230],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
]

REMAPPED_LANDCOVER_CLASSES = {
    k2: LANDCOVER_CLASSES[k1] for k1, k2 in zip(*LANDCOVER_REMAP)
}

CLASSES = {
    0: "None", 1: "NonForest", 2: "Forest", 3: "Water",
    4: "PreviousFire", 5: "Fire", 6: "PreviousHarvest", 7: "Harvest",
    8: "Cloud", 9: "CloudShadow"
}

CLASS_PALETTE = [
    'black', 'gold', 'darkCyan', 'darkOrange', 'red',
    'orchid', 'purple', 'cornsilk', 'dimGrey'
]

CLASS_VIS = {'min': 1, 'max': 9, 'palette': CLASS_PALETTE}

JUL_1 = 172 # approximate day of year for July 1st
SEP_30 = 282 # approximate day of year for Sep. 30th
DOY_RANGE = [JUL_1, SEP_30]

_BANDS = ['nir', 'red_edge', 'red', 'green', 'tca', 'ndvi']
_HISTORICAL_BANDS = ['historical_' + x for x in _BANDS]
BANDS = _BANDS + ['dem'] + _HISTORICAL_BANDS


def get_default_projection():
    return ee.Projection(PROJECTION).atScale(SCALE)
