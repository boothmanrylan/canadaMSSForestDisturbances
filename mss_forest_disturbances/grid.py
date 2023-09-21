"""Methods for creating a covering grid of Canada

Covering grids are used to make train/test/val splits.
"""
import os

import ee
from msslib import msslib

import constants
import preprocessing


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
    projection = constants.get_default_projection()
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


def set_ecozone(feat):
    """ Sets the ecozone property of the given feature.

    If the feature overlaps multiply ecozones ties are broken using 'first'

    Args:
        feat: ee.Feature

    Returnes:
        ee.Feature, the input feature with new property 'ecozone' containing
        the integer ecozone id of the feature.
    """
    geom = feat.goemetry()
    ecozones = ee.FeatureCollection(constants.ECOZONES)
    overlapping_ecozones = ecozones.filterBounds(geom)
    first_overlapping_ecozone = overlapping_ecozones.first()
    ecozone_id = first_overlapping_ecozone.get('ECOZONE_ID')
    return feat.set('ecozone', ecozone_id)


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
    def get_landcover(year):
        # TODO: why does this landcover need to apply a watermask?
        water_mask = preprocessing.get_water_mask(year)
        land_cover = preprocessing.reduce_resolution(
            preprocessing.get_landcover(year).gt(0)
        )
        return water_mask.And(land_cover)

    years = ee.List.sequence(
        constants.FIRST_LANDCOVER_YEAR,
        constants.LAST_MSS_YEAR
    )
    annual_landcover = ee.ImageCollection(years.map(get_landcover))
    landcover = annual_landcover.reduce(ee.Reducer.sum()).gt(0).selfMask()
    landcover = landcover.rename("landcover")

    grid = build_grid(aoi, chip_size, overlap_size)

    grid = grid.map(
        lambda x: x.set(
            "landcover",
            landcover.sample(
                x.geometry(),
                scale=constants.SCALE,
                numPixels=2500,
                dropNulls=False
            ).aggregate_array("landcover").size(),
        )
    )
    grid = grid.map(set_ecozone)

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
    fire = preprocessing.get_fire_map(year).selfMask().rename("fire")
    harvest = preprocessing.get_harvest_map(year).selfMask().rename("harvest")
    grid = grid.map(
        lambda x: x.set(
            "harvest", harvest.sample(
                x.geometry(),
                scale=constants.SCALE,
                numPixels=2500,
                dropNulls=False,
            ).aggregate_array("harvest").size(),
            "fire", fire.sample(
                x.geometry(),
                scale=constants.SCALE,
                numPixels=2500,
                dropNulls=False,
            ).aggregate_array("fire").size(),
            "year", year
        )
    )
    grid = grid.map(set_ecozone)

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
        doyRange=constants.DOY_RANGE,
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
