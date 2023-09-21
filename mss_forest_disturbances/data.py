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
import os

import ee
from msslib import msslib

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
    ).reproject(crs=get_default_projection())


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
    return ee.ImageCollection(LANDCOVER).filterDate(start, end).first()


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
    return reduce_resolution(ee.Image(FIRE).lt(year).unmask(0))


def get_fire_map(year=1985):
    """Gets a map of forest fire for the given year.

    The map is based on the Canadian Forest Service's annual forest fire maps.

    Args:
        year: integer >= 1985, the year to get the map for.

    Returns:
        ee.Image that is 1 where there was fire and 0 otherwise
    """
    year = ee.Image.constant(year)
    return reduce_resolution(ee.Image(FIRE).eq(year).unmask(0))


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
    return reduce_resolution(ee.Image(HARVEST).lt(year).unmask(0))


def get_harvest_map(year=1985):
    """Gets a map of harvest for the given year.

    The map is based on the Canadian Forest Service's annual harvest maps.

    Args:
        year: integer >= 1985, the year to get the map for.

    Returns:
        ee.Image that is 1 where there was harvest and 0 otherwise
    """
    year = ee.Image.constant(year)
    return reduce_resolution(ee.Image(HARVEST).eq(year).unmask(0))


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

    return harvest.blend(fire)


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

    Each grid cell will be a square that is chip_size pixels wide in the
    default projection and overlaps with its neighbours by overlap_size pixels.

    Args:
        aoi: ee.Geometry to create a grid for
        chip_size: int, the grid cell size in pixels
        overlap_size: int, optional, the amount of overlap in pixels between
            adjacent grid cells

    Returns:
        ee.FeatureCollection
    """
    projection = get_default_projection()
    error = ee.ErrorMargin(0.1, "projected")

    def buffer_and_bound(feat):
        geom = feat.geometry(0.1, projection)
        buffered = geom.buffer(overlap_size, error, projection)
        bounded = buffered.bounds(0.1, projection)
        return ee.Feature(bounded)

    # we want the final patch_size to be equal to the input chip size
    # therefore if overlap_size is given we want to start with a smaller
    # initial size and buffer out to the desired size
    chip_size -= overlap_size

    scale = projection.nominalScale()
    patch = scale.multiply(chip_size)

    grid = aoi.coveringGrid(projection, patch)
    return grid.map(buffer_and_bound).filterBounds(aoi)


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
                numPixels=2500,
                dropNulls=False
            ).aggregate_array("landcover").size(),
            "ecozone",
            ee.FeatureCollection(ECOZONES).filterBounds(x.geometry()).first().get('ECOZONE_ID')
        )
    )

    return grid.filter(ee.Filter.gte("landcover", 1000))


def add_disturbance_counts(grid, year):
    """ Estimates total harvest/fire in each cell of grid during year.

    Args:
        grid: ee.FeatureCollection originating from build_grid()
        year: int, the year to get disturbances from

    Returns:
        ee.FeatureCollection, the input grid after filtering out cells with
        no/few disturbances.
        Features will have the added properties "fire" and
        "harvest" containing the sample count of pixels with fire or harvest in
        them respectively, "year" the year the disturbance counts were based
        off of and "ecozone", the ECOZONE_ID the cell overlaps with.
    """
    fire = get_fire_map(year).selfMask().rename("fire")
    harvest = get_harvest_map(year).selfMask().rename("harvest")
    grid = grid.map(
        lambda x: x.set(
            "harvest", harvest.sample(
                x.geometry(),
                scale=60,
                numPixels=2500,
                dropNulls=False,
            ).aggregate_array("harvest").size(),
            "fire", fire.sample(
                x.geometry(),
                scale=60,
                numPixels=2500,
                dropNulls=False,
            ).aggregate_array("fire").size(),
            "ecozone",
            ee.FeatureCollection(ECOZONES).filterBounds(x.geometry()).first().get('ECOZONE_ID'),
            "year", year
        )
    )

    return grid


def _get_top_property(grid, prop, count):
    """ Return count size subset of grid after sorting (descending) by prop.

    Helper function for get_top_fire and get_top_harvest

    Args:
        grid: ee.FeatureCollection originating from add_disturbance_counts
        prop: string, name of property to sort by
        count: integer, size of subset to return

    Returns:
        ee.FeatureCollection
    """
    grid = grid.filter(ee.Filter.gt(prop, 0))
    output_grid = grid.limit(count, prop, False)
    output_grid = output_grid.map(
        lambda elem: elem.set("disturbance_type", prop)
    )
    return output_grid


def get_top_fire(grid, count):
    """ Return the count cells in grid with the most fire.

    Args:
        grid: ee.FeatureCollection originating from add_disturbance_counts
        count: integer, the number of cells to select.

    Returns:
        ee.FeatureCollection, subset of grid.
    """
    return _get_top_property(grid, "fire", count)


def get_top_harvest(grid, count):
    """" Return the count cells in grid with the most harvest.

    Args:
        grid: ee.FeatureCollection originating from add_disturbance_counts
        count: integer, the number of cells to select.

    Returns:
        ee.FeatureCollection, subset of grid.
    """
    return _get_top_property(grid, "harvest", count)


def get_random_undisturbed(grid, count):
    """ Return a random count cells from grid that are undisturbed.

    Args:
        grid: ee.FeatureCollection originating from add_disturbance_counts
        count: integer, the number of cells to select.

    Returns:
        ee.FeatureCollection, subset of grid.
    """
    grid = grid.filter(ee.Filter.And(
        ee.Filter.eq("fire", 0),
        ee.Filter.eq("harvest", 0)
    ))
    grid = grid.randomColumn("random", 42)
    grid = grid.limit(count, "random")
    grid = grid.map(
        lambda elem: elem.set("disturbance_type", "undisturbed")
    )
    return grid


def train_test_val_split(grid, trainp, testp, valp):
    """ Randomly splits grid into 3 groups.

    trianp, testp, and valp must sum to 1.0

    Args:
        grid: ee.FeatureCollection
        trainp: float between 0 and 1, % of grid to put in first group
        testp: float between 0 and 1, % of grid to put in second group
        valp: float between 0 and 1, % of grid to put in thrid group

    Returns:
        List of three ee.FeatureCollections
    """
    assert trainp + testp + valp == 1.0
    train_count = grid.size().multiply(trainp).int()
    test_count = grid.size().multiply(testp).int()
    val_count = grid.size().multiply(valp).int()

    grid = grid.randomColumn("train_test_val", 111).sort("train_test_val")

    train_grid = ee.FeatureCollection(
        grid.toList(train_count, 0)
    )
    test_grid = ee.FeatureCollection(
        grid.toList(test_count, train_count)
    )
    val_grid = ee.FeatureCollection(
        grid.toList(val_count, train_count.add(test_count))
    )
    return train_grid, test_grid, val_grid


def set_image_overlap(cell):
    """ Helper function for sample_cells to drop cells with no images.

    Args:
        cell: ee.Feature originating from add_disturbance_counts

    Returns:
        ee.Feature, the input feature with new property num_overlapping_images
    """
    year = cell.getNumber("year")

    images = msslib.getCol(
        aoi=cell.geometry().centroid(1),
        yearRange=[year, year],
        doyRange=DOY_RANGE,
        maxCloudCover=100
    )

    return cell.set("num_overlapping_images", images.size())


def sample_cells(
    grid,
    fire_count,
    harvest_count,
    undisturbed_count,
    trainp,
    testp,
    valp
):
    """ Splits grid by disturbance type and then randomly into train/test/val.

    Args:
        grid: ee.FeatureCollection originating from add_disturbance_counts
        fire_count: integer, how many of the most fire dominated cells to select
        harvest_count, integer, how many the most harvest dominated cells to
            select
        undisturbed_count, integer, how many undisturbed cells to select at
            random
        trainp: float, percentage of selected cells to allocate to the train
            group
        testp: float, percentage of selected cells to allocate to the test
            group
        valp: float, percentage of seleted cells to allocate to the val group

    Returns:
        List of three ee.FeatureCollections
    """
    # oversample then limit later to account for removal of duplicates and
    # cells with no overlapping images
    oversample = 2
    fire_set = get_top_fire(grid, int(oversample * fire_count))
    harvest_set = get_top_harvest(grid, int(oversample * harvest_count))
    undisturbed_set = get_random_undisturbed(
        grid, int(oversample * undisturbed_count)
    )

    # merge sets to drop duplicates
    grouped_set = ee.FeatureCollection(
        [fire_set, harvest_set, undisturbed_set]
    ).flatten().distinct('id')

    # drop cells with no overlapping images
    grouped_set = grouped_set.map(set_image_overlap)
    grouped_set = grouped_set.filter(ee.Filter.gt("num_overlapping_images", 0))

    # split groups by dominant disturbance type again
    fire_set = grouped_set.filter(ee.Filter.eq("disturbance_type", "fire"))
    fire_set = fire_set.limit(fire_count, "fire", False)

    harvest_set = grouped_set.filter(
        ee.Filter.eq("disturbance_type", "harvest")
    )
    harvest_set = harvest_set.limit(harvest_count, "harvest", False)

    undisturbed_set = grouped_set.filter(
        ee.Filter.eq("disturbance_type", "undisturbed")
    )
    undisturbed_set = undisturbed_set.limit(undisturbed_count)

    # apply train/test/val split to each disturbance type group individually
    splits = [trainp, testp, valp]
    fire_sets = train_test_val_split(fire_set, *splits)
    harvest_sets = train_test_val_split(harvest_set, *splits)
    undisturbed_sets = train_test_val_split(undisturbed_set, *splits)

    # group train/test/val splits from each disturbance type
    outputs = [
        ee.FeatureCollection([
            fire_sets[x], harvest_sets[x], undisturbed_sets[x]
        ]).flatten()
        for x in [0, 1, 2]
    ]

    # shuffle each group so that disturbance types are intermingled
    outputs = [
        x.randomColumn("shuffle", 1001).sort("shuffle")
        for x in outputs
    ]

    return outputs


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


def add_qa_mask(image):
    """ Adds the QA_PIXEL mask as a band to the input image.

    Args:
        image: ee.Image originating from msslib.getCol

    Returns:
        ee.Image, the input image with an additional band called qa_mask
    """
    qa = image.select('QA_PIXEL')
    mask = bitwise_extract(qa, 3, 3).eq(0).rename('qa_mask')
    return image.addBands(mask)


def get_tm():
    """ Returns an image collection containing all TM images.

    Args:
        None

    Returns:
        ee.ImageCollection
    """
    TM4_T1 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2')
    TM4_T2 = ee.ImageCollection('LANDSAT/LT04/C02/T2_L2')
    TM5_T1 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    TM5_T2 = ee.ImageCollection('LANDSAT/LT05/C02/T2_L2')
    return TM4_T1.merge(TM4_T2).merge(TM5_T1).merge(TM5_T2)


def process_tm(image):
    """ Apply scaling factors, mask clouds, and calcualte NBR.

    See here for explanation of sclaing factos
    https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT04_C02_T1_L2

    Args:
        image: ee.Image from Landsat Thematic Mapper.

    Returns:
        ee.Image
    """
    image = ee.Image(image)

    # apply scaling factors
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B6').multiply(0.00341802).add(149.0)
    image = image.addBands(optical, None, True).addBands(thermal, None, True)

    nbr = image.normalizedDifference(['SR_B4', 'SR_B7']).rename('NBR')

    return tm_clear_mask(image.addBands(nbr))


def get_coincident_tm(image):
    year = image.getNumber('year')
    aoi = image.geometry()

    # attempt to find coincident TM image
    day_before = image.date().advance(-1, "day")
    day_after = image.date().advance(1, "day")
    col = get_tm().filterDate(day_before, day_after).filterBounds(aoi)

    intersection_area = col.geometry().intersection(aoi, 1000).area(1000)
    overlap_percentage = intersection_area.divide(aoi.area(1000))

    use_tm = overlap_percentage.gte(0.99)
    return ee.Algorithms.If(
        use_tm,
        col.mosaic(),
        None
    )


def get_disturbance_mask(image, median_tca, median_nbr):
    """ Returns annual disturbance masks after thresholding the image.

    The threshold is based on the NBR of the coincident TM image if one exists,
    otherwise the threshold is based on the TCA of the given image.

    Args:
        image: ee.Image originating from msslib.getCol().
        median_tca: ee.Image
        median_nbr: ee.Image

    Returns:
        ee.Image
    """
    coincident_tm = get_coincident_tm(image)

    return ee.Image(ee.Algorithms.If(
        coincident_tm,
        process_tm(coincident_tm).select('NBR').lte(median_nbr),
        mss_clear_mask(msslib.addTc(image)).select('tca').lte(median_tca)
    ))


def make_disturbance_map(year, lookback=5):
    # TODO: what is this?
    # get treed not-treed from the year before the target
    # base will be 1 for no trees, 2 for trees, and 3 for water
    base = get_basemap(year, lookback=1)

    dated_prior_fire = ee.Image(FIRE).updateMask(ee.Image(FIRE).lt(year))
    years_since_fire = dated_prior_fire.subtract(year).multiply(-1)

    dated_prior_harvest = ee.Image(HARVEST).updateMask(ee.Image(HARVEST).lt(year))
    years_since_harvest = dated_prior_harvest.subtract(year).multiply(-1)

    base_offset = 3
    harvest_offset = year.subtract(1984)
    fire = years_since_fire.add(base_offset)
    harvest = years_since_harvest.add(harvest_offset).add(base_offset)

    return base.blend(harvest).blend(fire)


def preprocess(image):
    """ Applies preprocessing to MSS image to prepare for cnn.

    Args:
        image: ee.Image originating from msslib.getCol

    Returns:
        ee.Image
    """
    image = msslib.calcToa(image)
    image = msslib.addTc(image)
    image = normalize_tca(image)
    image = msslib.addNdvi(image)
    image = image.reproject(get_default_projection())
    image = image.addBands(get_dem())
    return image


def label_image(image, fire_lookback=3, harvest_lookback=10):
    """Creates the target labels for a given MSS image.

    Pixels are labelled as forest, non-forest, burn, harvest, water, cloud,
    or shadow. Forest, non-forest, water is based on the Canada Forested
    Landcover VLCE2 dataset. Fire and harvest comes from the Canadian Forest
    Service NTEMS annual forest fire and harvest maps. Cloud and cloud shadow
    are labelled based on Justin Braaten's MSS clear-view-mask.

    If a pixel would be labelled as more than one class the following precedence
    rule is used: (water/forest/non-forest) < (burn/harvest) < (cloud/shadow).

    To handle the image potentially being acquired before a disturbance
    occurred, a threshold is calculate based on the median value of a pixel
    across the collection. The threshold is based on NBR if a coincident TM
    image exists, otherwise it is based on the TCA of the given image.

    Class Labels:
        0: None (for masked pixels)
        1: Non-forest
        2: Forest
        3: Water
        4: Previous Fire
        6: Fire
        6: Previous Harvest
        7: Harvest
        8: Cloud
        9: Cloud Shadow

    Args:
        image: an ee.Image should originate from msslib.getCol() and
            msslib.calcToa().
        fire_lookback: int, how many prior years to include in the previous
            fire class.
        harvest_lookback: int, how many prior years to include in the previous
            harvest class.

    Returns:
        an ee.Image with one integer band containing the class of each pixel.
    """
    year = image.getNumber('year')
    aoi = image.geometry()

    fire_lookback = year.subtract(fire_lookback)
    harvest_lookback = year.subtract(harvest_lookback)
    base = get_basemap(year, lookback=1)
    base = base.add(1)  # to allow for 0 to equal masked pixels

    prior_fire = ee.Image(FIRE).lt(year).And(ee.Image(FIRE).gte(fire_lookback))
    prior_fire = prior_fire.selfMask()
    prior_harvest = ee.Image(HARVEST).lt(year).And(ee.Image(HARVEST).gte(harvest_lookback))
    prior_harvest = prior_harvest.selfMask()

    base_offset = 3  # non-forest, forest, water
    prior_fire = prior_fire.add(base_offset)

    # add an additional 2 to account for previous fire and fire
    prior_harvest = prior_harvest.add(base_offset).add(2)

    base = base.blend(prior_harvest).blend(prior_fire)

    # get the median TCA for the image region
    tca_median = msslib.getCol(
        aoi=image.geometry(),
        yearRange=[1972, 1995],
        doyRange=DOY_RANGE,
        maxCloudCover=20
    ).map(preprocess).select('tca').median()

    # get the median Thematic Mapper NBR for the image region
    tm_col = get_tm().filterBounds(image.geometry())
    tm_col = tm_col.filter(ee.Filter.lte('CLOUD_COVER', 20))
    tm_col = tm_col.filter(ee.Filter.calendarRange(*DOY_RANGE, "day_of_year"))
    nbr_median = tm_col.map(process_tm).select("NBR").median()

    # compute the mask for disturbances within the current year to account for
    # disturbances that have not yet occurred in the year at the time the image
    # was acquired
    disturbance_mask = get_disturbance_mask(image, tca_median, nbr_median)

    fire = get_fire_map(year).updateMask(disturbance_mask).selfMask()
    fire = fire.add(4)
    harvest = get_harvest_map(year).updateMask(disturbance_mask).selfMask()
    harvest = harvest.add(6)

    # compute the cloud and cloud shadow regions
    occlusion = msslib.addMsscvm(image, 20).select('msscvm').selfMask()
    occlusion = occlusion.add(7)

    label = base.blend(harvest).blend(fire).blend(occlusion)
    return label.rename("label")


def add_label(image, median_tca, median_nbr, *args, **kwargs):
    """ Helper function to add result of label_image as a band to the input.

    See label_image for a more complete description.

    Args:
        image: ee.Image
        median_tca: ee.Image
        median_nbr: ee.Image
        args: other arguments to pass to label_image
        kwargs: other named arguments to pass to label_image

    Returns:
        ee.Image, the input plus an additional band called "label" that is the
        result of calling label_image with the parameters passed to this
        function.
    """
    label = label_image(image, median_tca, median_nbr, *args, **kwargs)
    label = label.rename("label")
    return image.addBands(label)


def get_lookback_median(image, lookback=3, max_cloud_cover=20):
    """ Gets a median image from lookback prior years for given image.

    Args:
        image: ee.Image,
        lookback: int, how many years to base the median on
        max_cloud_cover: int, between 0 and 100, max cloud cover of image to
            allow in the collection the median is calculated from.

    Return:
        ee.Image
    """
    year = image.date().get("year")
    col = msslib.getCol(
        aoi=image.geometry(),
        yearRange=[year.subtract(lookback), year.subtract(1)],
        doyRange=DOY_RANGE,
        maxCloudCover=max_cloud_cover,
    ).map(preprocess).map(mss_clear_mask)
    return col.median().regexpRename("(.*)", "historical_$1", False)


def prepare_image_for_export(image):
    """ Preprocess image, adds historical bands, and calculates image label.

    Args:
        image: ee.Image originating from msslib.getCol

    Returns:
        ee.Image, ee.Image: the prepared image and label respectively
    """
    image = preprocess(image)
    historical_bands = get_lookback_median(image)
    label = label_image(image)

    image = image.addBands(historical_bands)
    types = ee.Dictionary.fromLists(
        image.bandNames(),
        ee.List.repeat("float", image.bandNames().size())
    )
    image = image.cast(types)
    image = image.select(BANDS)

    default_proj = get_default_projection()
    return image.reproject(default_proj), label.reproject(default_proj)


def prepare_metadata_for_export(image, cell):
    image = ee.Image(image)
    cell = ee.Feature(cell)

    doy = image.date().getRelative("day", "year")
    remapped_doy = doy.subtract(DOY_RANGE[0] - 1)  # shift by 1 for leap years

    ecozone = cell.get('ecozone')
    all_ecozone_ids = ee.FeatureCollection(ECOZONES).aggregate_array('ECOZONE_ID').distinct()
    remapped_ecozone = all_ecozone_ids.indexOf(ecozone)

    image_centroid = image.geometry().centroid(1)
    coords = image_centroid.coordinates()

    metadata = {
        "doy": remapped_doy.float(),
        "ecozone": remapped_ecozone.float(),
        "lon": coords.get(0),
        "lat": coords.get(1),
    }
    return metadata


def get_label(year):
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
    return base.blend(disturbances).rename("annual_label")


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


def get_dem():
    """ Gets a global digital elevation model.

    The DEM is returned in the default projection of this project.

    Adapted from msslib

    Args:
        None

    Returns:
        ee.Image
    """
    aw3d30 = ee.Image('JAXA/ALOS/AW3D30/V2_2').select('AVE_DSM').rename('dem')
    gmted = ee.Image('USGS/GMTED2010').rename('dem')
    dem = ee.ImageCollection([gmted, aw3d30]).mosaic()
    return dem.divide(MAX_ELEV).reproject(get_default_projection())


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
