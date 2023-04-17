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
    'black', 'gold', 'darkCyan', 'red', 'purple', 'cornsilk', 'dimGrey'
]

CLASS_VIS = {'min': 1, 'max': 7, 'palette': CLASS_PALETTE}

JUL_1 = 172 # approximate day of year for July 1st
SEP_30 = 282 # approximate day of year for Sep. 30th
DOY_RANGE = [JUL_1, SEP_30]


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

    The returned mask can be used like this:
    ```
    im = im.update_mask(get_water_mask(year))
    ```
    So that all regions covered by water in `year` are masked out in `im`.

    The mask is based on the Canada Forested Landcover VLCE2 map.

    Args:
        year: integer, if given the year to get the water mask for, if not given
            the most recent landcover map will be used. Must be >= 1984
        aoi: ee.Geometry, if given the mask will be clipped to the aoi, if not
            given the entire landcover map is returned.

    Returns:
        ee.Image that is 0 where there is water and 1 otherwise
    """
    if year is None:
        landcover = LANDCOVER.sort('system:time_start', False).first()
    else:
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        landcover = LANDCOVER.filterDate(start, end).first()

    if aoi is not None:
        landcover = landcover.clip(aoi)

    return landcover.neq(20)


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
    water = get_water_mask(year, aoi).Not().selfMask().add(1)
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
    """Gets a map of typed forest disturbances for the given year and aoi.

    The map is based on the Canadian Forest Service's annual harvest and fire
    maps.

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


def get_disturbed_regions(year=None, aoi=None):
    """ Returns a map of all disturbances for a given year and aoi.

    The result will be 1 wherever there was a disturbance and 0 otherwise.
    Disturbance type gets stripped. If you need disturbance type, use
    get_disturbance_map()

    Args:
        year: int, if given only disturbances that occurred during this year
            are returned, if not disturbances from all years are returned.
        aoi: ee.Geometry, if given the map will be clipped to the aoi, if not
            given the entire map is returned.

    Returns:
        ee.Image, map of disturbances.
    """
    if year is not None:
        fire = FIRE.eq(year).selfMask().unmask(0)
        harvest = HARVEST.eq(year).selfMask().unmask(0)
    else:
        fire = FIRE.gt(0).selfMask().unmask(0)
        harvest = HARVEST.gt(0).selfMask().unmask(0)

    disturbances = harvest.Or(fire)

    if aoi is not None:
        disturbances = disturbances.clip(aoi)

    return disturbances


def build_grid(aoi, proj, scale, chip_size, overlap_size=0):
    """ Creates a tiled grid that covers aoi.

    Each grid cell will be a square that ischip_size pixels wide in the given
    projection at the given scale and overlap with its neighbours by
    overlap_size pixels.

    Args:
        aoi: ee.Geometry to create a grid for
        proj: ee.Projection to perform the tiling in, must cover all of aoi
        scale: int, the nominalScale in meters of one pixel
        chip_size: int, the grid cell size in pixels
        overlap_size: int, optional, the amount of overlap in pixels between
            adjacent grid cells

    Returns:
        ee.FeatureCollection
    """
    def buffer_and_bound(feat):
        return ee.Feature(feat.geometry().buffer(overlap, 1).bounds(1, proj))

    patch = (chip_size - overlap_size) * scale
    overlap = overlap_size * scale

    grid = aoi.coveringGrid(proj.atScale(scale), patch)
    return grid.map(buffer_and_bound)


def get_disturbed_grid_cells(grid, year):
    """ Returns the subset of cells in grid that overlap a disturbance in year

    Args:
        grid: FeatureCollection e.g. originating from build_grid()
        year: int the year to get disturbed regions for

    Returns:
        2-tuple of ee.FeatureCollections (disturbed cells, undisturbed cells)
    """
    grid = grid.map(lambda x: x.set('year', year))

    disturbed_regions = get_disturbed_regions(year).selfMask()
    disturbed_vectors = disturbed_regions.reduceToVectors(
        geometry=grid.geometry(),
        scale=1000
    )
    large_disturbances = disturbed_vectors.filter(ee.Filter.gte('count', 10))

    intersect_filter = ee.Filter.intersects(
        leftField='.geo',
        rightField='.geo',
        maxError=100
    )

    join = ee.Join.simple()
    invert_join = ee.Join.inverted()

    disturbed = join.apply(grid, large_disturbances, intersect_filter)
    undisturbed = invert_join.apply(grid, large_disturbances, intersect_filter)
    
    return disturbed, undisturbed


def sample_grid_cells(grid, n, percent_disturbed, year):
    """ Samples n cells from grid.

    Attempts to have as close as possible to percent_disturbed of the sample to
    overlap with disturbances from year.

    Args:
        grid: ee.FeatureCollection e.g. originating form build_grid()
        n: int, the number of cells to sample
        percent_disturbed: float, the percentage of cells sampled that should
            overlap with a disturbance
        year: int, the year to base overlap with disturbances on

    Returns:
        2-tuple of ee.FeatureCollection, (disturbed sample, undisturbed sample)
    """
    disturbed_cells, undisturbed_cells = get_disturbed_grid_cells(grid, year)

    n_disturbed = disturbed_cells.size().min(n * percent_disturbed)
    n_undisturbed = ee.Number(n).subtract(n_disturbed)

    disturbed_cells = disturbed_cells.randomColumn('rand', year)
    disturbed_cells = disturbed_cells.limit(n_disturbed, 'rand')

    undisturbed_cells = undisturbed_cells.randomColumn('rand', year)
    undisturbed_cells = undisturbed_cells.limit(n_undisturbed, 'rand')

    return disturbed_cells, undisturbed_cells


def train_test_val_split(grid, year, n, ptrain, ptest, pval, pdisturbed):
    """ Samples cells from grid to make train/test/validation splits.

    Args:
        grid: ee.FeatureCollection, e.g. originating from build_grid()
        year: int, the year to base disturbances off of
        n: int, the total number of cells to sample
        ptrain: float, percentage of samples to allocate to the train split
        ptest: float, percentage of samples to allocate to the test split
        pval: float, percentage of samples to allocate to the val split
        pdisturbed: float, percentage of cells to enforce overlapping with a
            disturbance

    Returns:
        3-tuple of ee.FeatureCollections (train, test, validation)
    """
    disturbed, undisturbed = sample_grid_cells(grid, n, pdisturbed, year)

    def sample_and_merge(count1, count2, offset1, offset2):
        # grab count1 samples starting at offset1 from disturbed
        # grab count2 samples starting at offset2 from undisurbed
        # merge and shuffle
        d = ee.FeatureCollection(disturbed.toList(count1, offset1))
        u = ee.FeatureCollection(undisturbed.toList(count2, offset2))
        return d.merge(u).randomColumn('rand', 42).sort('rand')

    trnc1 = disturbed.size().multiply(ptrain).ceil()
    trnc2 = undisturbed.size().multiply(ptrain).ceil()
    tstc1 = disturbed.size().multiply(ptest).ceil()
    tstc2 = undisturbed.size().multiply(ptest).ceil()
    valc1 = disturbed.size().multiply(pval).ceil()
    valc2 = undisturbed.size().multiply(pval).ceil()

    # we don't need to shuffle before selecting b/c that is already done in
    # sample_grid_cells
    train = sample_and_merge(trnc1, trnc2, 0, 0)
    test = sample_and_merge(tstc1, tstc2, trnc1, trnc2)
    val = sample_and_merge(valc1, valc2, trnc1.add(tstc1), trnc2.add(tstc2))

    return train, test, val


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

