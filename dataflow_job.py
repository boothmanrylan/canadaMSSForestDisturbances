"""Launches a Beam/Dataflow job to generate training/testing data.
"""

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

import geemap
from msslib import msslib
from mss_forest_disturbances import constants, preprocessing


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
        opt_url='https://earthengine-highvolume.googleapis.com',
    )


def build_request(cell, size):
    # get top left corner of cell in *unscaled* projection
    proj = ee.Projection(constants.PROJECTION)
    coords = cell.geometry(1, proj).getInfo()['coordinates'][0][3]
    request = {
        'fileFormat': 'NPY',
        'grid': {
            'dimensions': {
                'width': size,
                'height': size,
            },
            'affineTransform': {
                'scaleX': constants.SCALE,
                'shearX': 0,
                'translateX': coords[0],
                'shearY': 0,
                'scaleY': -constants.SCALE,
                'translateY': coords[1],
            },
            'crsCode': proj.getInfo()['crs'],
        }
    }

    return request


def _get_images_from_feature(feature):
    geom = feature.geometry(
        ee.ErrorMargin(1, 'projected'),
        constants.get_default_projection()
    )

    year = feature.getNumber('year')

    images = msslib.getCol(
        aoi=geom.centroid(1),
        yearRange=[year, year],
        doyRange=constants.DOY_RANGE,
        maxCloudCover=100
    )

    return images


def get_image_ids(row, asset):
    ee_init()

    col = ee.FeatureCollection(asset)
    feature = col.filter(ee.Filter.eq('id', row['id'])).first()

    images = _get_images_from_feature(feature)

    image_ids = images.aggregate_array('system:id').getInfo()
    feature_ids = itertools.repeat(row['id'])
    paths = itertools.repeat(asset)

    return zip(image_ids, feature_ids, paths)


@retry.Retry()
def get_image_label_metadata(image_id, feature_id, asset):
    ee_init()

    image = msslib.process(ee.Image(image_id))
    image, label = preprocessing.prepare_image_for_export(image)

    col = ee.FeatureCollection(asset)
    cell = col.filter(ee.Filter.eq('id', feature_id)).first()
    metadata = preprocessing.prepare_metadata_for_export(image, cell)
    metadata = {key: val.getInfo() for key, val in metadata.items()}

    request = build_request(cell, constants.EXPORT_PATCH_SIZE)

    image_request = {
        'expression': image.unmask(0, sameFootprint=False,
        **request
    }
    np_image = np.load(io.BytesIO(ee.data.computePixels(image_request)))

    image_request = {
        'expression': label.unmask(0, sameFootprint=False)
        **request
    }
    np_label = np.load(io.BytesIO(ee.data.computePixels(label_request)))

    return np_image, np_label, metadata


def serialize_tensor(image, label, metadata):
    features = {
        b: tf.train.Feature(
            float_list=tf.train.FloatList(
                value=image[b].flatten()
            )
        )
        for b in constants.BANDS
    }

    features['label'] = tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=label['label'].flatten()
        )
    )

    for key, value in metadata.items():
        features[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[value])
        )

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def run_pipeline(input_asset, output_prefix, max_requests, beam_args):
    ee_init()

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
        max_num_workers=max_requests,
        direct_num_works=max(max_requests, 20),
        disk_size_gb=50,
    )

    with beam.Pipeline(options=beam_options) as pipeline:
        pcoll = pipeline | beam.Create(list(df.iloc))
        groups = pcoll | beam.Partition(partition_func, len(sets))

        for i, group in enumerate(groups):
            uid = f'{sets[i][0]}_{sets[i][1]}'
            (
                group
                | f'{uid} get image ids'
                >> beam.FlatMap(get_image_ids, asset=input_asset)
                | f'{uid} reshuffle' >> beam.Reshuffle()
                | f'{uid} get data' >> beam.MapTuple(get_image_label_metadata)
                | f'{uid} serialize' >> beam.MapTuple(serialize_tensor)
                | f'{uid} write'
                >> beam.io.WriteToTFRecord(
                    paths[i], file_name_suffix='.tfrecord.gz'
                )
            )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max-requests',
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
    parser.add_argument(
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
