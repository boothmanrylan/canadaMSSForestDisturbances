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
PROJECTION = LANDCOVER.first().projection().atScale(60)

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
    'black', 'gold', 'darkCyan', 'darkOrange', 'red',
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


def reduce_resolution(im):
    """ Explicitly reduce the resolution of a 1/0 image from 30m Landsat data.

    Adapted from:
    https://developers.google.com/earth-engine/guides/resample#reduce_resolution

    Args:
        im: ee.Image

    Returns:
        ee.Image
    """
    return im.reduceResolution(
        reducer=ee.Reducer.max(),  # if any pixel is 1 make the output 1
        maxPixels=1024,
    ).reproject(crs=PROJECTION)


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
    return reduce_resolution(landcover.eq(81).Or(landcover.gte(210)))


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
    return reduce_resolution(landcover.neq(20))


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
    year = ee.Image.constant(year)
    return reduce_resolution(FIRE.lt(year).unmask(0))


def get_fire_map(year=1985):
    """Gets a map of forest fire for the given year.

    The map is based on the Canadian Forest Service's annual forest fire maps.

    Args:
        year: integer >= 1985, the year to get the map for.

    Returns:
        ee.Image that is 1 where there was fire and 0 otherwise
    """
    year = ee.Image.constant(year)
    return reduce_resolution(FIRE.eq(year).unmask(0))


def get_previous_harvest_map(year=1985):
    """ Gets a map of all harvest up to but not including year.

    The map is based on the Canadian Forest Service's annual harvest maps.

    Args:
        year: integer >= 1985

    Returns:
        ee.Image that is 1 where there was harvest prior to year and 0
        otherwise.
    """
    year = ee.Image.constant(year)
    return reduce_resolution(HARVEST.lt(year).unmask(0))


def get_harvest_map(year=1985):
    """Gets a map of harvest for the given year.

    The map is based on the Canadian Forest Service's annual harvest maps.

    Args:
        year: integer >= 1985, the year to get the map for.

    Returns:
        ee.Image that is 1 where there was harvest and 0 otherwise
    """
    year = ee.Image.constant(year)
    return reduce_resolution(HARVEST.eq(year).unmask(0))


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


def build_grid(aoi, chip_size, overlap_size=0):
    """ Creates a tiled grid that covers aoi.

    Each grid cell will be a square that ischip_size pixels wide in the given
    projection at the given scale and overlap with its neighbours by
    overlap_size pixels.

    Args:
        aoi: ee.Geometry to create a grid for
        chip_size: int, the grid cell size in pixels
        overlap_size: int, optional, the amount of overlap in pixels between
            adjacent grid cells

    Returns:
        ee.FeatureCollection
    """
    def buffer_and_bound(feat, overlap):
        return ee.Feature(feat
            .geometry(0.1, PROJECTION)
            .buffer(overlap_size, ee.ErrorMargin(0.1, 'projected'), PROJECTION)
            .bounds(0.1, PROJECTION)
        )

    scale = PROJECTION.nominalScale()
    patch = scale.multiply(chip_size - overlap_size)
    overlap = scale.multiply(overlap_size)

    grid = aoi.coveringGrid(PROJECTION, patch)
    return grid.map(lambda x: buffer_and_bound(x, overlap)).filterBounds(aoi)


def build_land_covering_grid(aoi, chip_size, overlap_size=0):
    """ Creates a tiled grid that covers aoi, excluding water dominant tiles.

    See build_grid()

    Args:
        aoi: ee.Geometry
        chip_size: int
        overlap_size: int

    Returns:
        ee.FeatureCollection
    """
    landcover = ee.ImageCollection(ee.List.sequence(1984, 1995).map(
        lambda y: get_water_mask(y).And(reduce_resolution(get_landcover(y).gt(0)))
    )).reduce(ee.Reducer.sum()).gt(0).selfMask().rename("landcover")

    grid = build_grid(aoi, chip_size, overlap_size)

    grid = grid.map(
        lambda x: x.set(
            "landcover",
            landcover.sample(
                x.geometry(),
                scale=60,
                numPixels=100,
                dropNulls=False
            ).aggregate_array("landcover").size(),
            "ecozone",
            ECOZONES.filterBounds(x.geometry()).first().get('ECOZONE_ID')
        )
    )

    return grid.filter(ee.Filter.gte("landcover", 30))


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


def train_test_val_split(grid, n, ptrain, ptest, pval):
    """ Samples cells from grid to make train/test/validation splits.

    Samples n cells from the grid.

    Args:
        grid: ee.FeatureCollection, e.g. originating from build_grid()
        n: int, the total number of cells to sample
        ptrain: float, percentage of samples to allocate to the train split
        ptest: float, percentage of samples to allocate to the test split
        pval: float, percentage of samples to allocate to the val split

    Returns:
        3-tuple of ee.FeatureCollections (train, test, validation)
    """
    grid = grid.randomColumn('rand', 42)
    grid = grid.sort('rand')

    n = ee.Number(n)
    train_count = n.multiply(ptrain).ceil()
    test_count = n.multiply(ptest).ceil()
    val_count = n.multiply(pval).ceil()

    train = ee.FeatureCollection(
        grid.toList(train_count, 0)
    )
    test = ee.FeatureCollection(
        grid.toList(test_count, train_count)
    )
    val = ee.FeatureCollection(
        grid.toList(val_count, train_count.add(test_count))
    )
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


def mss_clear_mask(image):
    """ Cheap cloud and cloud shadow mask for Landsat MSS images.

    Unless efficiency is absolutely necessary msslib.applyMsscvm is a better
    option than this method.

    Args:
        image: ee.Image originating from msslib.getCol

    Returns:
        ee.Image, the input image after appyting the mask.
    """
    qa = image.select('QA_PIXEL')
    mask = bitwise_extract(qa, 3, 3).eq(0)
    return image.updateMask(mask)


def add_mss_clear_mask(image):
    """ Adds the QA_PIXEL mask as a band to the input image.

    Args:
        image: ee.Image originating from msslib.getCol

    Returns:
        ee.Image, the input image with an additional band called qa_mask
    """
    qa = image.select('QA_PIXEL')
    mask = bitwise_extract(qa, 3, 3).eq(0).rename('qa_mask')
    return image.addBands(mask)


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


def get_disturbance_masks(image, fire_stds=2, harvest_stds=1, mss_adjust=-1):
    """ Returns annual disturbance masks after thresholding the image.

    The threshold is based on the NBR of the coincident TM image if one exists,
    otherwise the threshold is based on the TCA of the given image.

    Args:
        im: ee.Image originating from msslib.getCol
        fire_stds: int, the number of standard deviations below the
            mean NBR/TCA a pixel inside of regions labeled as fire by the
            Canadian Forest Service needs to be to be labeled as fire.
            Necessary to do this check because the image may have been
            acquired before the disturbance occurred.
        harvest_stds: int, the number of standard deviations below the
            mean NBR/TCA a pixel inside of regions labeled as harvest by the
            Canadian Forest Service needs to be to be labeled as harvest.
            Necessary to do this check because the image may have been
            acquired before the disturbance occurred.
        mss_adjust: int, how far to adjust fire_stds and harvest_stds when
            falling back to the MSS image.

    Returns:
        3tuple of ee.Images containing the fire mask, the harvest mask, and the
        band index that the thresholding was performed on.
    """
    year = image.getNumber('year')
    aoi = image.geometry()

    # attempt to find coincident TM image
    day_before = image.date().advance(-1, "day")
    day_after = image.date().advance(1, "day")
    col = TM.filterDate(day_before, day_after).filterBounds(aoi)

    intersection_area = col.geometry().intersection(aoi, 1000).area(1000)
    overlap_percentage = intersection_area.divide(aoi.area(1000))

    # only use NBR if it covers at least 99% of the image otherwise use TCA
    use_tm = overlap_percentage.gte(0.99)
    band_index = ee.Image(ee.Algorithms.If(
        use_tm,
        process_tm(col.mosaic()).select('NBR'),
        mss_clear_mask(msslib.addTc(image)).select('tca')
    )).rename('threshold')

    # lower the threshold requirements if falling back to the MSS image
    fire_stds = ee.Number(ee.Algorithms.If(
        use_tm,
        fire_stds,
        max(fire_stds + mss_adjust, 0)
    ))
    harvest_stds = ee.Number(ee.Algorithms.If(
        use_tm,
        harvest_stds,
        max(harvest_stds + mss_adjust, 0)
    ))

    # set the threshold based on the mean undisturbed band index
    undisturbed_index = band_index.updateMask(get_treed_mask(year.subtract(1)))
    mean_undisturbed_index = undisturbed_index.reduceRegion(
        geometry=aoi,
        scale=600,
        reducer=ee.Reducer.mean()
    ).getNumber('threshold')

    # it is possible that all of aoi gets masked and the reducer returns None
    mean_undisturbed_index = ee.Number(ee.Algorithms.If(
        mean_undisturbed_index,
        mean_undisturbed_index,
        1000
    ))

    std_undisturbed_index = undisturbed_index.reduceRegion(
        geometry=aoi,
        scale=600,
        reducer=ee.Reducer.stdDev()
    ).getNumber('threshold')

    # it is possible that all of aoi gets masked and the reducer returns None
    std_undisturbed_index = ee.Number(ee.Algorithms.If(
        std_undisturbed_index,
        std_undisturbed_index,
        1
    ))

    fire_threshold = mean_undisturbed_index.subtract(
        fire_stds.multiply(std_undisturbed_index)
    )
    harvest_threshold = mean_undisturbed_index.subtract(
        harvest_stds.multiply(std_undisturbed_index)
    )

    fire_mask = band_index.lte(fire_threshold)
    harvest_mask = band_index.lte(harvest_threshold)

    return fire_mask, harvest_mask, band_index


def label_image(image, fire_threshold=2, harvest_threshold=1, mss_adjust=-1):
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
        mss_adjust: int, how far to adjust fire_stds and harvest_stds when
            falling back to the MSS image.

    Returns:
        an ee.Image with one integer band containing the class of each pixel.
    """
    year = image.getNumber('year')
    aoi = image.geometry()

    fire_mask, harvest_mask, _ = get_disturbance_masks(
        image,
        fire_threshold,
        harvest_threshold,
        mss_adjust
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


def add_label(image, *args, **kwargs):
    """ Helper function to add result of label_image as a band to the input.

    See label_image for a more complete description.

    Args:
        image: ee.Image
        args: other arguments to pass to label_image
        kwargs: other named arguments to pass to label_image

    Returns:
        ee.Image, the input plus an additional band called "label" that is the
        result of calling label_image with the parameters passed to this
        function.
    """
    label = label_image(image, *args, **kwargs).rename("label")
    return image.addBands(label)


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
    return base.blend(disturbances).clip(aoi)


def normalize_tca(image):
    """ Normalized tca values to ~[0, 1].

    Tasseled Cap Angle is defined in msslib as:
        atan(greeness / brightness) * (180 / pi)

    It therefore ranges from 0 to 90

    Args:
        ee.Image

    Returns:
        ee.Image
    """
    tca = ee.Image(image).select(['tca'])
    tca = tca.divide(90)
    return image.addBands(tca, ['tca'], True)


def get_data_for_cell(cell, first_year=1982, last_year=1995):
    """ Return the inputs and target labels for the given cell.

    Returns all images of cell between first_year and last_year (inclusive),
    along with the labels of each individual image and the label of each year.

    Args:
        cell: ee.Geometry, the aoi to get the inputs and target labels for
            (e.g. this function can be mapped across one of the outputs of
            train_test_val_split)
        first_year: int, the year to start getting images from (inclusive)
        last_year: int, the last year to get images from (inclusive)

    Returns:
        2-tuple of dict mapping int -> ee.ImageCollection/ee.Image. In each
        dict, the keys are the years, the values are all the images of the cell
        during that year. The first dict contains the MSS images converted to
        TOA with the TCA and the pixel labels added, the second dict contains
        the "final" label for each year.
    """
    def get_col(year):
        col = msslib.getCol(
            aoi=cell,
            yearRange=[year, year],
            doyRange=DOY_RANGE,
            maxCloudCover=100
        )
        col = col.map(msslib.calcToa).map(msslib.addTc)
        # col = col.map(lambda im: im.clip(cell))
        col = col.map(add_label)
        col = col.map(add_mss_clear_mask)

        bands = ['nir', 'red_edge', 'red', 'green', 'tca', 'qa_mask']
        col = col.map(lambda im: im.select(bands))

        col = col.map(normalize_tca)
        col = col.map(lambda im: im.reproject(PROJECTION))
        return col

    years = ee.List.sequence(first_year, last_year)
    keys = years.map(lambda x: ee.Number(x).int().format())
    images = years.map(get_col)
    labels = years.map(lambda y: get_label(cell, y))

    images = ee.Dictionary.fromLists(keys, images)
    labels = ee.Dictionary.fromLists(keys, labels)

    return images, labels


def convert_image_to_array(image, cell, patch_size):
    """ Converts an image to an array with shape (H, W, C)

    Args:
        image: ee.Image
        cell: ee.Geometry, passed to sampleRectangle
        patch_size: int, the width and height of cell in pixels

    Returns:
        ee.Array
    """
    def reshape(arr):
        arr = ee.Array(arr)
        return arr.reshape([patch_size, patch_size, 1])

    image = ee.Image(image)
    bands = image.bandNames()
    array = image.reproject(PROJECTION).sampleRectangle(cell, defaultValue=0)

    band_arrays = bands.map(lambda x: reshape(array.get(x)))

    return ee.Array.cat(band_arrays, 2)


def convert_collection_to_array(col, cell, patch_size):
    """ Converts an imagecollection to an array with shape (N, H, W, C)

    Args:
        col: ee.ImageCollection,
        cell: ee.Geometry, passed to sampleRectangle
        patch_size: int, the width and height of cell in pixels

    Returns:
        ee.Array
    """
    def _helper(col):
        col = col.toList(col.size())
        col = col.map(
            lambda x: convert_image_to_array(x, cell, patch_size)
        )
        col = col.map(
            lambda x: ee.Array(x).reshape([1, patch_size, patch_size, -1])
        )
        return ee.Array.cat(col, 0)

    col = ee.ImageCollection(col)

    return ee.Algorithms.If(
        col.size(),
        _helper(col),
        ee.Array([], ee.PixelType.float())
    )


def get_dates(col):
    """ Returns the date of each image in col as an array.

    Args:
        col: ee.ImageCollection

    Returns:
        ee.Array containing the date of each image in the input in epoch time.
    """
    def _helper(col):
        col = col.toList(col.size())
        col = col.map(lambda im: ee.Image(im).date().millis())
        return ee.Array(col)

    col = ee.ImageCollection(col)

    return ee.Algorithms.If(
        col.size(),
        _helper(col),
        ee.Array([], ee.PixelType.float())
    )


def prepare_export(cell, patch_size, first_year=1982, last_year=1995):
    """ Converts data for a grid cell into a feature to export as a TFRecord.

    See get_data_for_cell() for a better description of the returned data.

    Args:
        cell: ee.Geometry
        first_year: int
        last_year: int

    Returns:
        ee.Feature with the following properties: a digital elevation model
        called "dem", the latitude and longitude of the centroid of the cell
        called "lat", and "long" respectively, the height and width of a cell
        in pixels called "height" and "width" respectively, and a list of the
        bands in each image called "bands". And then for each year between
        first_year and last_year (inclusive) a property containing the number
        of images in that year called "num{year}", all of the images in that
        year as an array with shape (N, H, W, C) called "array{year}", an array
        of the end of year label for a year called "label{year}", and a list of
        the dates of each image in a year called "dates{year}".
    """
    cell = cell.geometry(0.1, PROJECTION)
    images, yearly_labels = get_data_for_cell(cell, first_year, last_year)

    first_image = ee.ImageCollection(images.get(images.keys().get(0))).first()
    dem = msslib.getDem(first_image)
    dem = dem.divide(3000)  # only ~70 points in Canada > 3000m
    dem = dem.reproject(PROJECTION)
    dem = convert_image_to_array(dem, cell, patch_size)
    coords = cell.centroid(1).coordinates()

    output = ee.Feature(None, {
        'dem': dem,
        'lat': coords.get(1),
        'long': coords.get(0),
        'height': patch_size,
        'width': patch_size,
        'bands': first_image.bandNames(),
    })

    # helper function to rename keys
    def _prepend(string, year):
        return ee.String(string).cat(ee.String(year))

    # set the number of images in each year as a property in the feature
    yearly_counts = images.values().map(lambda v: ee.ImageCollection(v).size())
    count_keys = images.keys().map(lambda k: _prepend("num", k))
    yearly_counts = ee.Dictionary.fromLists(count_keys, yearly_counts)
    output = output.set(yearly_counts)

    # set the image sequence array (N, H, W, C) for each year as a property
    yearly_arrays = images.values().map(
        lambda v: convert_collection_to_array(v, cell, patch_size)
    )
    array_keys = images.keys().map(lambda k: _prepend("array", k))
    yearly_arrays = ee.Dictionary.fromLists(array_keys, yearly_arrays)
    output = output.set(yearly_arrays)

    # set the list of dates (epoch time) of each image in each year
    yearly_dates = images.values().map(get_dates)
    dates_keys = images.keys().map(lambda k: _prepend("dates", k))
    yearly_dates = ee.Dictionary.fromLists(dates_keys, yearly_dates)
    output = output.set(yearly_dates)

    # set the annual/end of year label for each year as a property
    label_keys = yearly_labels.keys().map(lambda k: _prepend("label", k))
    yearly_labels = ee.Dictionary.fromLists(label_keys, yearly_labels.values())
    output = output.set(yearly_labels)

    return output


def sample_image(image, points_per_class=2, num_classes=9):
    """ Given an image return a stratified sample of points from it.

    Args:
        image: ee.Image to sample points from.
        points_per_class: integer, how many points to sample from each class in
            the input image.
        num_classes: integer, how many classes are in the image. Classe labels
            should run from 0 to num_classes - 1.

    Returns:
        ee.FeatureCollection, one feature per point, each feature as one
            property per band containing the band value at the point.
    """
    def _sample(im):
        return im.sample(
            region=im.geometry(),
            scale=60,
            numPixels=1000 * points_per_class,
            dropNulls=True
        ).limit(points_per_class)

    label = image.select('label')
    classes = ee.List.sequence(0, num_classes - 1).map(
        lambda x: ee.Image.constant(x)
    )
    class_images = ee.ImageCollection(classes.map(
        lambda x: image.updateMask(label.eq(x)))
    )
    class_samples = class_images.map(_sample)
    samples = class_samples.flatten()
    samples = samples.map(
        lambda x: x.set('image', image.get('LANDSAT_SCENE_ID'))
    )
    return samples


def sample_points(cell):
    """ Given a grid cell create a dataset of labeled points to train a RF.

    Samples points from each image that overlaps with the grid during the year
    the grid was sampled from. Returns the flattened collection.

    Args:
        cell: ee.Geometry e.g., originating from train_test_val_split()

    Returns:
        ee.FeatureCollection containing one feature per point that was samples
        where each feature has one property per band.
    """
    year = cell.getNumber('year')
    cell = cell.geometry()

    col = msslib.getCol(
        aoi=cell,
        yearRange=[year, year],
        doyRange=DOY_RANGE,
        maxCloudCover=100
    ).map(msslib.calcToa)

    col = col.map(add_label).map(lambda im: im.clip(cell))

    samples = col.map(sample_image).flatten()
    samples = samples.map(
        lambda x: x.set('year', year)
    )
    return samples
