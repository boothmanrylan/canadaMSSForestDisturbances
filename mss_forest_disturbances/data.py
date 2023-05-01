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
    'black', 'gold', 'darkCyan', 'orangeRed', 'red',
    'orchid', 'purple', 'cornsilk', 'dimGrey'
]

CLASS_VIS = {'min': 1, 'max': 9, 'palette': CLASS_PALETTE}

JUL_1 = 172 # approximate day of year for July 1st
SEP_30 = 282 # approximate day of year for Sep. 30th
DOY_RANGE = [JUL_1, SEP_30]

TM4_T1 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2')
TM4_T2 = ee.ImageCollection('LANDSAT/LT04/C02/T2_L2')
TM5_T1 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
TM5_T2 = ee.ImageCollection('LANDSAT/LT05/C02/T2_L2')
TM = TM4_T1.merge(TM4_T2).merge(TM5_T1).merge(TM5_T2)


def get_landcover(year=1984):
    """Gets a landcover map for the given year.

    The map is the Canada Forested Landcover VLCE2 map.

    Args:
        year: integer >= 1984, the year to get the map for.

    Returns:
        ee.Image
    """
    year = ee.Number(year).max(1984)
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    return LANDCOVER.filterDate(start, end).first()


def get_treed_mask(year=1984):
    """Gets a tree mask for the given year.

    The mask is based on the Canada Forested Landcover VLCE2 map.
    Classes 81 (wetland treed), 210 (coniferous), 220 (broadleaf), and
    230 (mixedwood) are considered to be treed, all other classes are
    considered to be non-treed.

    This method is based on Flavie Pelletier's treedMask script.

    Args:
        year: integer >= 1984, the year to get the tree mask for.

    Returns:
        ee.Image that is 1 where there are trees and 0 otherwise
    """
    landcover = get_landcover(year)
    return landcover.eq(81).Or(landcover.gte(210))


def get_water_mask(year=1984):
    """Gets a water mask for the given year and aoi.

    The returned mask can be used like this:
    ```
    im = im.update_mask(get_water_mask(year))
    ```
    So that all regions covered by water in `year` are masked out in `im`.

    The mask is based on the Canada Forested Landcover VLCE2 map.

    Args:
        year: integer >= 1984, the year to get the water mask for.

    Returns:
        ee.Image that is 0 where there is water and 1 otherwise
    """
    landcover = get_landcover(year)
    return landcover.neq(20)


def get_basemap(year=1984, lookback=0):
    """Gets a tress/water/other map for the given year.

    The map is based on the Canada Forested Landcover VLCE2 map.

    Args:
        year: integer >= 1984, the year to get the map for.
        lookback: integer, the tree/no tree labels will be drawn lookback years
            before the given year

    Returns:
        ee.Image that is 1 where there are trees, 2 where there is water,
        and 0 otherwise.
    """
    trees = get_treed_mask(ee.Number(year).subtract(lookback))
    water = get_water_mask(year).Not().selfMask().add(1)
    return trees.blend(water)


def get_previous_fire_map(year=1985):
    """ Get a map of all fire up to but not including year.

    The map is based on the Canadian Forest Service's annual forest fire maps.

    Args:
        year, integer >= 1985.

    Returns:
        ee.Image that is 1 where there was fire prior to year and 0 otherwise.
    """
    return FIRE.lt(year).unmask(0)


def get_fire_map(year=1985):
    """Gets a map of forest fire for the given year.

    The map is based on the Canadian Forest Service's annual forest fire maps.

    Args:
        year: integer >= 1985, the year to get the map for.

    Returns:
        ee.Image that is 1 where there was fire and 0 otherwise
    """
    return FIRE.eq(year).unmask(0)


def get_previous_harvest_map(year=1985):
    """ Gets a map of all harvest up to but not including year.

    The map is based on the Canadian Forest Service's annual harvest maps.

    Args:
        year: integer >= 1985

    Returns:
        ee.Image that is 1 where there was harvest prior to year and 0
        otherwise.
    """
    return HARVEST.lt(year).unmask(0)


def get_harvest_map(year=1985):
    """Gets a map of harvest for the given year.

    The map is based on the Canadian Forest Service's annual harvest maps.

    Args:
        year: integer >= 1985, the year to get the map for.

    Returns:
        ee.Image that is 1 where there was harvest and 0 otherwise
    """
    return HARVEST.eq(year).unmask(0)


def get_disturbance_map(year=1985):
    """Gets a map of typed forest disturbances for the given year.

    The map is based on the Canadian Forest Service's annual harvest and fire
    maps.

    Args:
        year: integer >= 1985, the year to get the map for.

    Returns:
        ee.Image that is 1 where there was fire, 2 where there was harvest and
        0 otherwise.
    """
    previous_fire = get_previous_fire_map(year)
    fire = get_fire_map(year).selfMask().add(1)
    fire = previous_fire.blend(fire)

    previous_harvest = get_previous_harvest_map(year).selfMask().add(2)
    harvest = get_harvest_map(year).selfMask().add(3)
    harvest = previous_harvest.blend(harvest)

    return fire.blend(harvest)


def get_disturbed_regions(year=1985):
    """ Returns a map of all disturbances for a given year and aoi.

    The result will be 1 wherever there was a disturbance and 0 otherwise.
    Disturbance type gets stripped. If you need disturbance type, use
    get_disturbance_map()

    Args:
        year: integer >= 1985, the year to get disturbed regions for.

    Returns:
        ee.Image, that is 1 where a disturbance occurred and 0 otherwise.
    """
    fire = get_fire_map(year)
    harvest = get_harvest_map(year)
    return fire.Or(harvest)


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


def bitwise_extract(image, from_bit, to_bit):
    """ Helper method for extracting QA bit masks.

    Code adapted from
    https://spatialthoughts.com/2021/08/19/qa-bands-bitmask-gee/
    which was in turn adapted from https://gis.stackexchange.com/a/349401/5160

    Args:
        image: ee.Image
        from_bit: integer, the bit position to start extraction from
        to_bit: integer, the bit position to end extraction at

    Returns:
        ee.Image
    """
    mask_size = ee.Number(1).add(to_bit).subtract(from_bit)
    mask = ee.Number(1).leftShift(mask_size).subtract(1)
    return image.rightShift(from_bit).bitwiseAnd(mask)


def tm_clear_mask(image):
    """ Mask cloud and cloud shadow pixels in Landsat TM images.

    Cloud mask is based on the 6th bit of the QA_PIXEL band.
    Shadow mask is based on the 4th bit of the QA_PIXEL band.

    Args:
        image: ee.Image must be from Landsat TM

    Returns:
        ee.Image, the input image after applying the mask.
    """
    qa = image.select('QA_PIXEL')
    cloud_mask = bitwise_extract(qa, 6, 6).eq(1)
    shadow_mask = bitwise_extract(qa, 4, 4).neq(1)
    return image.updateMask(cloud_mask).updateMask(shadow_mask)


def process_tm(image):
    """ Apply scaling factors, mask clouds, and calcualte NBR.

    See here for explanation of sclaing factos
    https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT04_C02_T1_L2

    Args:
        image: ee.Image from Landsat Thematic Mapper.

    Returns:
        ee.Image
    """
    # apply scaling factors
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B6').multiply(0.00341802).add(149.0)
    image = image.addBands(optical, None, True).addBands(thermal, None, True)

    nbr = image.normalizedDifference(['SR_B4', 'SR_B7']).rename('NBR')

    return tm_clear_mask(image.addBands(nbr))


def get_threshold_image(mss_image):
    """ Given an MSS image from Landsat 4/5 return the coincident TM NBR image.

    Args:
        mss_image: an ee.Image originating from msslib.getCol must be an MSS
            image from Landsat 4 or Landset 4

    Returns:
        ee.Image, the coincident TM NBR image to the input image, if one exists,
        if no conincident image exists, returns the input image TCA
    """
    aoi = mss_image.geometry()
    day_before = mss_image.date().advance(-1, "day")
    day_after = mss_image.date().advance(1, "day")
    col = TM.filterDate(day_before, day_after).filterBounds(aoi)

    return ee.Image(ee.Algorithms.If(
        col.size(),
        process_tm(col.mosaic()).select('NBR'),
        msslib.addTc(mss_image).select('tca')
    )).rename('threshold')


def get_disturbance_masks(image, fire_threshold=2, harvest_threshold=1):
    """ Returns annual disturbance masks after thresholding the image.

    The threshold is based on the NBR of the coincident TM image if one exists,
    otherwise the threshold is based on the TCA of the given image.

    Args:
        im: ee.Image originating from msslib.getCol
        fire_threshold: int, the number of standard deviations below the
            mean NBR/TCA a pixel inside of regions labeled as fire by the
            Canadian Forest Service needs to be to be labeled as fire.
            Necessary to do this check because the image may have been
            acquired before the disturbance occurred.
        harvest_threshold: int, the number of standard deviations below the
            mean NBR/TCA a pixel inside of regions labeled as harvest by the
            Canadian Forest Service needs to be to be labeled as harvest.
            Necessary to do this check because the image may have been
            acquired before the disturbance occurred.

    Returns:
        2tuple of (ee.Image, ee.Image) containing the fire and harvest masks
        respectively
    """
    year = image.getNumber('year')
    aoi = image.geometry()

    threshold = get_threshold_image(image)

    undisturbed = threshold.updateMask(get_treed_mask(year.subtract(1)))

    mean_undisturbed = undisturbed.reduceRegion(
        geometry=aoi,
        scale=600,
        reducer=ee.Reducer.mean()
    ).getNumber('threshold')
    std_undisturbed = undisturbed.reduceRegion(
        geometry=aoi,
        scale=600,
        reducer=ee.Reducer.stdDev()
    ).getNumber('threshold')

    fire_mask = mean_undisturbed.subtract(
        std_undisturbed.multiply(fire_threshold)
    )
    harvest_mask = mean_undisturbed.subtract(
        std_undisturbed.multiply(harvest_threshold)
    )

    return threshold.lte(fire_mask), threshold.lte(harvest_mask)


def label_image(image, fire_threshold=2, harvest_threshold=1):
    """Creates the target labels for a given MSS image.

    Pixels are labelled as forest, non-forest, burn, harvest, water, cloud,
    or shadow. Forest, non-forest, water is based on the Canada Forested
    Landcover VLCE2 dataset. Fire and harvest comes from the Canadian Forest
    Service NTEMS annual forest fire and harvest maps. Cloud and cloud shadow
    are labelled based on Justin Braaten's MSS clear-view-mask.

    If a pixel would be labelled as more than one class the following precedence
    rule is used: (water/forest/non-forest) < (burn/harvest) < (cloud/shadow).

    To handle the image potentially being acquired before a disturbance
    occurred harvest and fire regions are thresholded based on the NBR value of
    the coincident Thematic Mapper image if one exists, otherwise the threshold
    is based on the TCA valu fo the given image.

    Args:
        image: an ee.Image should originate from msslib.getCol() and
            msslib.calcToa().
        fire_threshold: int, the number of standard deviations below the
            mean NBR/TCA a pixel inside of regions labeled as fire by the
            Canadian Forest Service needs to be to be labeled as fire.
            Necessary to do this check because the image may have been
            acquired before the disturbance occurred.
        harvest_threshold: int, the number of standard deviations below the
            mean NBR/TCA a pixel inside of regions labeled as harvest by the
            Canadian Forest Service needs to be to be labeled as harvest.
            Necessary to do this check because the image may have been
            acquired before the disturbance occurred.

    Returns:
        an ee.Image with one integer band containing the class of each pixel.
    """
    year = image.getNumber('year')
    aoi = image.geometry()

    fire_mask, harvest_mask = get_disturbance_masks(
        image,
        fire_threshold,
        harvest_threshold
    )

    base = get_basemap(year, lookback=1).add(1)

    previous_fire = get_previous_fire_map(year).selfMask().add(3)
    fire = get_fire_map(year).selfMask().add(4)
    fire = previous_fire.blend(fire.updateMask(fire_mask))

    previous_harvest = get_previous_harvest_map(year).selfMask().add(5)
    harvest = get_harvest_map(year).selfMask().add(6)
    harvest = previous_harvest.blend(harvest.updateMask(harvest_mask))

    occlusion = msslib.addMsscvm(image, 20).select('msscvm').selfMask().add(7)

    return base.blend(fire).blend(harvest).blend(occlusion).clip(aoi)


def get_label(aoi, year):
    """ Create target labelling for the given year and aoi.

    Pixels are labelled as forest, non-forest, burn, harvest, or water. Forest,
    non-forest, water is based on the Canada Forested Landcover VLCE2 dataset.
    Fire and harvest comes from the Canadian Forest Service NTEMS annual forest
    fire and harvest maps.

    If a pixel would be labelled as more than one class the following precedence
    rule is used: (water/forest/non-forest) < (burn/harvest)

    Args:
        aoi: ee.Geomatry, the region to get the label for
        year: int, the year to base the disturbances off of

    Returns:
        ee.Image with one integer band contianing the class of each pixel.
    """
    base = get_basemap(year).add(1)
    disturbances = get_disturbance_map(year).selfMask().add(3)
    return base.blend(disturbances)


def get_data_for_cell(cell, lookback=3, lookahead=3):
    """ Return the inputs and target labels for the given cell.

    Returns an image collection containing all the MSS images from lookback
    years prior to the year of cell, an image collection containing all the MSS
    images from the year of cell, an image collection containing the labels for
    each MSS image acquired within the year of the cell, an image collection
    containing all the MSS images from lookahead years after the year of the
    cell, and an image containing the true label of cell during the year of the
    cell.

    Args:
        cell: ee.Geometry, the aoi to get the inputs and target labels for
            (e.g. this function can be mapped across on of the outputs of
            train_test_val_split)
        lookback: int, the number of years of data to include in the lookback
            collection
        lookahead: int, the number of years of data to include in the lookahead
            collection

    Returns:
        dictionary: keys are current_col, lookback_col, lookahead_col,
        label_col, and true_label
    """
    year = cell.getNumber('year')
    cell = cell.geometry()

    def clip(im):
        return im.clip(cell)

    output = {}
    output['current_col'] = msslib.getCol(
        aoi=cell,
        yearRange=[year, year],
        doyRange=DOY_RANGE,
        maxCloudCover=100
    ).map(msslib.calcToa)

    output['label_col'] = output['current_col'].map(label_image).map(clip)
    output['current_col'] = output['current_col'].map(clip)

    output['lookback_col'] = msslib.getCol(
        aoi=cell,
        yearRange=[year.subtract(lookback - 1), year.subtract(1)],
        doyRange=DOY_RANGE,
        maxCloudCover=100
    ).map(msslib.calcToa).map(clip)

    output['lookahead_col'] = msslib.getCol(
        aoi=cell,
        yearRange=[year.add(1), year.add(lookahead + 1)],
        doyRange=DOY_RANGE,
        maxCloudCover=100,
    ).map(msslib.calcToa).map(clip)

    output['true_label'] = get_label(cell, year)

    return output
