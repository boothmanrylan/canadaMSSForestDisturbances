"""Methods for creating train/test/validation data.

Landcover Data comes from:
Hermosilla, T., M.A. Wulder, J.C. White, N.C. Coops, G. W. Hobart, (2018).
Disturbance-Informed Annual Land Cover Classification Maps of Canada's Forested
Ecosystems for a 29-Year Landsat Time Series. Canadian Journal of Remote
Sensing. 44(1) 67-87.

Forest fire and harvest data comes from:
Hermosilla, T., M.A. Wulder, J.C. White, N.C. Coops, G.W. Hobart, L.B. Campbell,
2016. Mass data processing of time series Landsat imagery: pixels to data
products for forest monitoring. International Journal of Digital Earth 9(11),
1035-1054

Data is available in GeoTiff format at:
https://opendata.nfis.org/mapserver/nfis-change_eng.html

Data is available in Earth Engine at:
landcover: ee.ImageCollection('projects/sat-io/open-datasets/CA_FOREST_LC_VLCE2')
forest fire: ee.Image('users/boothmanrylan/NTEMSCanada/forest_fire_1985-2020')
harvest: ee.Image('users/boothmanrylan/NTEMSCanada/harvest_1985-2020')
"""

import ee
from msslib import msslib

LANDCOVER = ee.ImageCollection('projects/sat-io/open-datasets/CA_FOREST_LC_VLCE2')
FIRE = ee.Image('users/boothmanrylan/NTEMSCanada/forest_fire_1985-2020')
HARVEST = ee.Image('users/boothmanrylan/NTEMSCanada/harvest_1985-2020')
ECOZONES = ee.FeatureCollection('users/boothmanrylan/forest_dominated_ecozones')

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
    0: "None", 1: "Non-forest", 2: "Forest", 3: "Water",
    4: "Burn", 5: "Harvest", 6: "Cloud", 7: "Cloud Shadow"
}

CLASS_PALETTE = [
    'black', 'gold', 'forestGreen', 'blue', 'red', 'saddleBrown', 'white', 'grey'
]

JUL_1 = 172 # approximate day of year for July 1st
SEP_30 = 282 # approximate day of year for Sep. 30th

_QUALITY_BAND = 'tca'
_QUALITY_KEY = 'quality'


def get_landcover(year=None, aoi=None):
    """Gets a landcover map for the given year and aoi.

    The map is the Canada Forested Landcover VLCE2 map.

    Args:
        year: integer, if given the year to get the map for, if not given
            the most recent landcover map will be used. Must be >= 1984
        aoi: ee.Geometry, if given the map will be clipped to the aoi, if not
            given the entire landcover map is returned.

    Returns:
        ee.Image
    """
    if year is None:
        landcover = LANDCOVER.sort('system:time_start', False).first()
    else:
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        landcover = LANDCOVER.filterDate(start, end).first()

    if aoi is not None:
        landcover = landcover.clip(aoi)

    return landcover


def get_treed_mask(year=None, aoi=None):
    """Gets a tree mask for the given year and aoi.

    The mask is based on the Canada Forested Landcover VLCE2 map.
    Classes 81 (wetland treed), 210 (coniferous), 220 (broadleaf), and
    230 (mixedwood) are considered to be treed, all other classes are
    considered to be non-treed.

    This method is based on Flavie Pelletier's treedMask script.

    Args:
        year: integer, if given the year to get the tree mask for, if not given
            the most recent landcover map will be used. Must be >= 1984
        aoi: ee.Geometry, if given the mask will be clipped to the aoi, if not
            given the entire landcover map is returned.

    Returns:
        ee.Image that is 1 where there are trees and 0 otherwise
    """
    if year is None:
        landcover = LANDCOVER.sort('system:time_start', False).first()
    else:
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        landcover = LANDCOVER.filterDate(start, end).first()

    if aoi is not None:
        landcover = landcover.clip(aoi)

    return landcover.eq(81).Or(landcover.gte(210))


def get_water_mask(year=None, aoi=None):
    """Gets a water mask for the given year and aoi.

    The mask is based on the Canada Forested Landcover VLCE2 map.

    Args:
        year: integer, if given the year to get the water mask for, if not given
            the most recent landcover map will be used. Must be >= 1984
        aoi: ee.Geometry, if given the mask will be clipped to the aoi, if not
            given the entire landcover map is returned.

    Returns:
        ee.Image that is 1 where there is water and 0 otherwise
    """
    if year is None:
        landcover = LANDCOVER.sort('system:time_start', False).first()
    else:
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        landcover = LANDCOVER.filterDate(start, end).first()

    if aoi is not None:
        landcover = landcover.clip(aoi)

    return landcover.eq(20)


def get_basemap(year=None, aoi=None):
    """Gets a tress/water/other map for the given year and aoi.

    The map is based on the Canada Forested Landcover VLCE2 map.

    Args:
        year: integer, if given the year to get the map for, if not given
            the most recent landcover map will be used. Must be >= 1984
        aoi: ee.Geometry, if given the mask will be clipped to the aoi, if not
            given the entire landcover map is returned.

    Returns:
        ee.Image that is 1 where there are trees, 2 where there is water,
        and 0 otherwise.
    """
    trees = get_treed_mask(year, aoi)
    water = get_water_mask(year, aoi).selfMask().add(1)
    return trees.blend(water)


def get_fire_map(year=None, aoi=None):
    """Gets a map of forest fire for the given year and aoi.

    The map is based on the Canadian Forest Service's annual forest fire maps.

    Args:
        year: integer, if given the year to get the map for, if not given
            the most recent landcover map will be used. Must be >= 1985
        aoi: ee.Geometry, if given the map will be clipped to the aoi, if not
            given the entire map is returned.

    Returns:
        ee.Image that is 1 where there was fire and 0 otherwise
    """
    if year is None:
        year = 2020

    fire = FIRE.eq(year)

    if aoi is not None:
        fire = fire.clip(aoi)

    return fire.unmask(0)


def get_harvest_map(year=None, aoi=None):
    """Gets a map of harvest for the given year and aoi.

    The map is based on the Canadian Forest Service's annual harvest maps.

    Args:
        year: integer, if given the year to get the map for, if not given
            the most recent landcover map will be used. Must be >= 1985
        aoi: ee.Geometry, if given the map will be clipped to the aoi, if not
            given the entire map is returned.

    Returns:
        ee.Image that is 1 where there was harvest and 0 otherwise
    """
    if year is None:
        year = 2020

    harvest = HARVEST.eq(year)

    if aoi is not None:
        harvest = harvest.clip(aoi)

    return harvest.unmask(0)


def get_disturbance_map(year=None, aoi=None):
    """Gets a map of forest disturbances for the given year and aoi.

    The map is based on the Canadian Forest Service's annual harvest maps.

    Args:
        year: integer, if given the year to get the map for, if not given
            the most recent landcover map will be used. Must be >= 1985
        aoi: ee.Geometry, if given the map will be clipped to the aoi, if not
            given the entire map is returned.

    Returns:
        ee.Image that is 1 where there was fire, 2 where there was harvest and
        0 otherwise.
    """
    fire = get_fire_map(year, aoi)
    harvest = get_harvest_map(year, aoi).selfMask().add(1)
    disturbances = fire.blend(harvest)

    return disturbances


def label_image(image):
    """Creates the target labels for a given MSS image.

    Pixels are labelled as forest, non-forest, burn, harvest, water, cloud,
    or shadow. Forest, non-forest, water is based on the Canada Forested
    Landcover VLCE2 dataset. Fire and harvest comes from the Canadian Forest
    Service NTEMS annual forest fire and harvest maps. Cloud and cloud shadow
    are labelled based on Justin Braaten's MSS clear-view-mask.

    If a pixel would be labelled as more than one class the following precedence
    rule is used: (water/forest/non-forest) < (burn/harvest) < (cloud/shadow).

    Args:
        image: an ee.Image should originate from msslib.getCol() and
            msslib.calcToa().

    Returns:
        an ee.Image with one integer band containing the class of each pixel.
    """

    year = image.getNumber('year')
    aoi = image.geometry()

    base = get_basemap(year, aoi).add(1)
    disturbances = get_disturbance_map(year, aoi).selfMask().add(3)
    occlusion = msslib.addMsscvm(image).select('msscvm').selfMask().add(5)

    # TODO: apply threshold to TCA within disturbed regions

    return base.blend(disturbances).blend(occlusion)

