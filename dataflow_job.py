import argparse
import io
import itertools
import logging
import os

import google
from google.api_core import retry

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

import tensorflow as tf
import numpy as np

import ee

ee.Initialize()
ee.Authenticate()

import geemap
from msslib import msslib
from mss_forest_disturbances import data


PROJECT = 'api-project-269347469410'
BUCKET = 'gs://rylan-mssforestdisturbances/'
LOCATION = 'us-central1'
HIGH_VOLUME_ENDPOINT = 'https://earthengine-highvolume.googleapis.com'
ASSET_PATH = 'projects/api-project-269347469410/assets/rylan-mssforestdisturbances/'
PROJECTION = ee.Projection('EPSG:4269').atScale(60)
ERROR_MARGIN = ee.ErrorMargin(1, 'projected')

proj = PROJECTION.getInfo()
scale_x = proj['transform'][0]
scale_y = -proj['transform'][4]

PATCH_SIZE = 512

OFFSET_X = -scale_x * PATCH_SIZE / 2
OFFSET_Y = -scale_y * PATCH_SIZE / 2

REQUEST = {
    'fileFormat': 'NPY',
    'grid': {
        'dimensions': {
            'width': PATCH_SIZE,
            'height': PATCH_SIZE,
        },
        'affineTransform': {
            'scaleX': scale_x,
            'shearX': 0,
            'shearY': 0,
            'scaleY': scale_y,
        },
        'crsCode': prj['crs'],
    }
}


bands = ['nir', 'red_edge', 'red', 'green', 'tca', 'ndvi']
historical_bands = ['historical_' + x for x in bands]
BANDS = bands + historical_bands


def ee_init():
    credentials, project = google.auth.default(
        scopes=[
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/earthengine',
        ]
    )
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url=HIGH_VOLUM_ENDPOINT,
    )


def _get_images_from_feature(feature):
    geom = feature.geometry(ERROR_MARGIN, PROJECTION)
    year = feature.getNumber('year')

    images = msslib.getCol(
        aoi=geom.centroid(1),
        yearRange=[year, year],
        doyRange=data.DOY_RANGE,
        maxCloudCover=100
    )

    return images


def get_image_ids(row, asset):
    ee_init()

    col = ee.FeatureCollection(asset)
    feature = col.filter(ee.Filter.eq('id', row['id'])).first()

    images = _get_images_from_feature(feature)

    image_ids = images.aggregate_array('system:id').getInfo()
    feture_ids = itertools.repeat(row['id'])
    paths = itertools.repeat(asset_path)

    return zip(image_ids, feature_ids, paths)


@retry.Retry()
def get_image_label_metadata(image_id, feature_id, asset):
    ee_init()

    image = msslib.process(ee.Image(image_id))
    image, label = data.prepare_image_for_export(image)
    image = image.select(BANDS)

    col = ee.FeatureCollection(asset)
    feature = col.filter(ee.Filter.eq('id', feature_id)).first()
    metadata = data.prepare_metadata_for_export(image, feature)
    metadata = {key: val.getInfo() for key, val in metadata.items()}

    geom = feature.geometry(ERROR_MARGIN, PROJECTION)
    coords = geom.centroid(1).getInfo()['coordinates']

    request = dict(REQUEST)
    request['grid']['affineTransform']['translateX'] = coords[0] + OFFSET_X
    request['grid']['affineTransform']['translateY'] = coords[1] + OFFSET_Y

    image_request = dict(request)
    image_request['expression'] = image.unmask(0, sameFootprint=False)
    np_image = np.load(io.BytesIO(ee.data.computePixels(image_request)))

    label_request = dict(request)
    label_request['exporession'] = label.unamsk(0, sameFootprint=False)
    np_label = np.load(io.BytesIO(ee.data.computePixels(label_request)))

    return np_image, np_label, metadata


def serialze_tensor(image, label, metadata):
    features = {
        b: tf.train.Feature(
            float_list=tf.train.FloatList(
                value=image[b].flatten()
            )
        )
        for b in BANDS
    }

    features['label'] = tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=label['label'].flatten()
        )
    )

    for key, value in metadata.items():
        features[key] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value])
        )

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def run_pipeline(input_asset, output_prefix, max_requests, beam_args):
    col = ee.FeatureCollection(input_asset)
    df = geemap.ee_to_df(
        col, col_names=['disturbance_type', 'ecozone', 'id', 'shuffle']
    )

    ##################################################
    # TODO: remove this before running full export
    # work on a small random subset of the complete dataframe during testing
    df = df.sort_values(by='shuffle', ignore_index=True).head(max_requests)
    ##################################################

    ecozones = set(df['ecozone'])
    disturbance_types = set(df['disturbance_type'])

    sets = list(itertools.product(ecozones, disturbance_types))

    paths = [
        os.path.join(output_prefix, f'ecozone{ecozone}', disturbance_type)
        for ecozone, disturbance_type in sets
    ]

    def partition_func(elem, _num_partitions):
        elem_set = (int(elem['ecozone']), elem['disturbance_type'])
        return sets.index(elem_set)

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        setup_file='./setup.py',
        max_num_workers=max_requests,
        direct_num_works=max(max_request, 20),
        disk_size_gb=50,
    )

    with beam.Pipeline(options=beam_options) as pipeline:
        pcoll = pipeline | beam.Create(list(df.iloc))
        groups = pcoll | beam.Partition(partition_func, len(sets))

        for i, group in enumerate(groups):
            uid = f'{sets[i][0]_{sets[i][1]}'
            (
                group
                | f'{uid} get image ids'
                >> beam.FlatMap(get_image_ids, asset=input_asset)
                | f'{uid} reshuffle' >> beam.Reshuffle()
                | f'{uid} get data' >> beam.MapTuple(get_image_label_metadata)
                | f'{uid} serialze' >> beam.MapTuple(serialize_tensor)
                | f'{uid} write'
                >> beam.io.WriteToTFRecord(
                    paths[i], file_name_suffix='.tfrecord.gz'
                )
            )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max-request',
        default=20,
        type=int,
        help='Number of concurrent requests to Earth Engine',
    )
    parser.add_argument(
        '--input-asset',
        required=True,
        type=str,
        help='Path to Earth Engine FeatureCollection containing export patches'
    )
    parse.add_argument(
        '--output-prefix',
        required=True,
        type=str,
        help='Path to a Google Cloud bucket folder to write results to'
    )

    args, beam_args = parser.parse_known_args()

    run_pipeline(
        max_requests=args.max_requests,
        input_asset=args.input_asset,
        output_prefix=args.output_prefix,
        beam_args=beam_args,
    )