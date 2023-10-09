"""Launches a Beam/Dataflow job to generate maps using a trained model.
"""

import argparse
import io
import os
import math

from google.api_core import retry

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.ml.inference.base import RunInference, KeyedModelHandler
from apache_beam.ml.inference.tensorflow_inference import TFModelHandlerTensor
from apache_beam.ml.inference.tensorflow_inference import ModelType

import tensorflow as tf
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

import ee

import rasterio
from rasterio.transform import Affine

from msslib import msslib
from mss_forest_disturbances import (
    constants,
    preprocessing,
    dataflow_utils,
    model,
    bulc,
)


class ProcessCell(beam.DoFn):
    """DoFn to preprocess data for a grid cell before being passed to a model."""

    def setup(self):
        dataflow_utils.ee_init()

    @retry.Retry()  # TODO: this might fail b/c we are decorating a class method
    def process(self, element, asset, start_year, end_year, batch_size):
        """Returns the result of calling computePixels as a TensorFlow dataset.

        Args:
            element: int, int, keyed unique identifier of an element of asset
            asset: str, path to an ee.FeatureCollection asset
            patch_size: int, height/width in pixels passed to computePixels
                request
            batch_size: int, batch size used in call to model.

        Returns:
            int, tf.data.Dataset, the key and the dataset
        """
        key, index = element

        col = ee.FeatureCollection(asset)
        feature = col.filter(ee.Filter.eq("cell_id", index)).first()

        geometry = feature.geometry(
            ee.ErrorMargin(1, "projected"), constants.get_default_projection()
        )
        images = (
            msslib.getCol(
                aoi=geometry,
                yearRange=[start_year, end_year],
                doyRange=constants.DOY_RANGE,
                maxCloudCover=100,
            )
            .map(preprocessing.prepare_image)
            .map(lambda im: im.clip(geometry))
        )

        request = dataflow_utils.build_request(feature, constants.PATCH_SIZE)

        num_images = images.size().getInfo()
        max_images_per_request_by_bands = math.floor(
            constants.COMPUTE_PIXELS_MAX_BANDS / len(constants.BANDS)
        )
        max_images_per_request_by_bytes = math.floor(
            constants.COMPUTE_PIXELS_MAX_BYTES
            / ((constants.PATCH_SIZE**2) * len(constants.BANDS) * (32 / 8))
        )
        max_images_per_request = min(
            max_images_per_request_by_bands, max_images_per_request_by_bytes
        )
        num_requests = math.ceil(num_images / max_images_per_request)

        image_list = images.toList(images.size())

        all_results = []
        for i in range(num_requests):
            start = i * max_images_per_request
            stop = min((i + 1) * max_images_per_request, num_images)
            request_images = image_list.slice(start, stop)
            request_image = ee.ImageCollection(request_images).toBands()
            request["expression"] = request_image

            curr_result = np.load(io.BytesIO(ee.data.computePixels(request)))

            curr_result = structured_to_unstructured(curr_result, dtype=np.float32)

            # undo the call to toBands() to get individual images back
            curr_result = np.split(
                curr_result, curr_result.shape[-1] // len(constants.BANDS), -1
            )
            curr_result = np.stack(curr_result, 0)
            all_results.append(curr_result)

        result = np.stack(all_results)

        result = tf.data.Dataset.from_tensor_slices(result).batch(batch_size)

        return key, result


class RunInferencePerElement(beam.DoFn):
    """DoFn that passes its input to RunInference.

    Necessary because we will have a pcollection of pcollections and want to
    call RunInfernce on the inner pcollections while keeping their outputs
    separate
    """

    def __init__(self, keyed_model_handler):
        """
        keyed_model_handler: KeyedModelHandler, passed to RunInference in
            process
        """
        self.keyed_model_handler = keyed_model_handler

    def process(self, element):
        """Returns element | RunInference

        Args:
            element: keyed pcollection

        Returns:
            keyed pcollection containing model inputs and outputs
        """
        return element | RunInference(model_handler=self.keyed_model_handler)


class StackPredictionResults(beam.DoFn):
    """DoFn to stack all inference elements of an iterable of PredictionResult."""

    @staticmethod
    def combine(element):
        """Stacks inference elements along first dimension.

        Args:
            element: iterable of NamedTuple with elements example and inference

        Returns:
            ndarray, the inference elements of each NamedTuple stacked along
            first dimension
        """
        return np.stack([x.inference for x in element], axis=0)

    def process(self, element):
        """Passes input pcollection to CombineGlobally.

        Uses StackPredictionResults.combine as the combineFn.

        Args:
            element: keyed pcollection

        Returns:
            ndarray
        """
        key, collection = element
        stacked_collection = collection | beam.CombineGlobally(
            StackPredictionResults.combine
        )
        return key, stacked_collection


class WriteToDisk(beam.DoFn):
    """DoFn to write a collection Numpy Arrays to disk as GeoTiffs."""

    def __init__(self, output_prefix, **kwargs):
        super().__init__(**kwargs)
        self.output_prefix = output_prefix

    def setup(self):
        dataflow_utils.ee_init()

    def process(self, element, asset):
        """Writes each member of element to disk as a separate GeoTiff file.

        Args:
            element: key, ndarray (images, rows, cols, bands)
            asset: str, path to earth engine FeatureCollection asset

        Returns:
            None
        """
        key, stacked_prediction_result = element

        col = ee.FeatureCollection(asset)
        feature = col.filter(ee.Filter.eq("cell_id", key))

        proj = ee.Projection(constants.PROJECTION)
        coords = feature.geometry(1, proj).getInfo()["coordinates"][0][3]

        transform = Affine(
            constants.SCALE, 0, coords[0], 0, -constants.SCALE, coords[1]
        )

        raster_properties = {
            "driver": "GTiff",
            "crs": constants.PROJECTION,
            "transform": transform,
        }

        for i, prediction_result in enumerate(stacked_prediction_result):
            output_file = os.path.join(
                self.output_prefix, f"cell{key:05}_image{i:05}.gtiff"
            )

            # rasterio expects bands to be in the first dimension
            prediction_result = np.transpose(prediction_result, (2, 0, 1))

            raster_properties["count"] = prediction_result.shape[0]
            raster_properties["height"] = prediction_result.shape[1]
            raster_properties["width"] = prediction_result.shape[2]
            raster_properties["dtype"] = prediction_result.dtype

            with rasterio.open(output_file, "w", **raster_properties) as dst:
                dst.write(prediction_result)


class ComputeAnnualMaps(beam.DoFn):
    """Convert a stack of classified images into annual maps."""

    def setup(self):
        dataflow_utils.ee_init()

    def process(element, start_year, end_year, input_asset):
        """Reduces a stack of classified images to a stack of annual maps.

        The returned maps will have one integer band to make exporting easier.

        Args:
            element: ndarray, stacked results output from model 1
            start_year: int, first year to get images for
            end_year: int, last year to get images for
            input_asset: str, path to earth engine FeatureCollection Asset

        Returns:
            ndarray (image, height, width, 2), annual harvest/fire maps
        """
        key, array = element

        processed_array = bulc.bulcp(array)

        col = ee.FeatureCollection(input_asset)
        feature = col.filter(ee.Filter.eq("cell_id", key)).first()

        geometry = feature.geometry(
            ee.ErrorMargin(1, "projected"), constants.get_default_projection()
        )
        images = msslib.getCol(
            aoi=geometry,
            yearRange=[start_year, end_year],
            doyRange=constants.DOY_RANGE,
            maxCloudCover=100,
        ).map(lambda im: im.set("year", im.date().get("year")))

        years = images.aggregate_array("year").getInfo()
        unique_years = images.aggregate_array("year").distinct().getInfo()

        last_occurrence_of_each_year = [
            len(years) - years[::-1].index(y) - 1 for y in unique_years
        ]

        # TODO: implement logic to convert stack of image classifications into
        # annual maps

        # Step 1a.
        # drop cloud/cloud shadow observations as they dont provide evidence
        # for or against a pixel being disturbed

        # Step 1b.
        # drop pixels whose max classification is below a threshold
        # sum previous/current disturbance classes before applying threshold

        # Step 2.
        # remove likely false positives: any pixels that are disturbed for the
        # entire time period and any pixels that are disturbed exactly once

        # Step 3a.
        # look for pattern forest -> burn -> previous burn -> forest/nonforest
        # assign pixel bunr in the year of the first burn observation

        # Step 3b.
        # look for pattern forst -> no observations in year -> previous burn
        # assign pixel burn in the year of no observations

        # Step 4a.
        # look for pattern forest -> harvest -> previous harvest -> forest/nonforest
        # assign pixel harvest in the year of the first harvest observation

        # Step 4b.
        # look for pattern forst -> no observations in year -> previous harvest
        # assign pixel harvest in the year of no observations

        # return key, array

        return key, processed_array[last_occurrence_of_each_year]


class TFNoBatchModelHandler(TFModelHandlerTensor):
    """Model handler class that overrides batching."""

    def batch_element_kwargs(self):
        return {"max_batch_size": 1}


def run_pipeline(
    beam_args,
    input_asset,
    output_prefix,
    model_checkpoint_path,
    max_requests=20,
    model_one_output_prefix=None,
    start_year=constants.FIRST_MSS_YEAR,
    end_year=constants.FIRST_DISTURBANCE_YEAR - 1,
    batch_size=constants.BATCH_SIZE,
):
    dataflow_utils.ee_init()

    col = ee.FeatureCollection(input_asset)
    cells = col.aggregate_array("cell_id").getInfo()

    ###################################################
    # TODO remove this before running full job
    # work on a small subset of the complete collection during testing
    # num_rows = col.size().getInfo()
    cells = cells[:20]
    ###################################################

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        max_num_workers=max_requests,
        direct_num_workers=max(max_requests, 20),
        disk_size_gb=50,
    )

    model_handler = KeyedModelHandler(
        TFNoBatchModelHandler(
            model_checkpoint_path,
            model_type=ModelType.SAVED_WEIGHTS,
            create_model_fn=model.build_model,
        )
    )

    with beam.Pipeline(options=beam_options) as pipeline:
        model_one_outputs = (
            pipeline
            | "Create" >> beam.Create(zip(cells, cells))
            | "Generate tf.data.Dataset"
            >> beam.ParDo(ProcessCell(), input_asset, start_year, end_year, batch_size)
            | "Run Model 1" >> beam.ParDo(RunInferencePerElement(model_handler))
            | "Stack Model 1 Outputs" >> beam.ParDo(StackPredictionResults())
        )

        if model_one_output_prefix is not None:
            model_one_outputs | "Save Model 1 Ouputs" >> beam.ParDo(
                WriteToDisk(model_one_output_prefix), input_asset
            )

        (
            model_one_outputs
            | "Run Model 2"
            >> beam.ParDo(ComputeAnnualMaps(), start_year, end_year, input_asset)
            | "Save Model 2 Outputs"
            >> beam.ParDo(WriteToDisk(output_prefix), input_asset)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-asset",
        required=True,
        type=str,
        help="Path to an Earth Engine FeatureCollection containing export patches",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=str,
        help="Path to a Google Cloud bucket folder to write results to.",
    )
    parser.add_argument(
        "--model-checkpoint-path",
        required=True,
        type=str,
        help="Path to a Google Cloud bucket location storing model weights.",
    )
    parser.add_argument(
        "--model-one-output-prefix",
        type=str,
        help="If given, Google Cloud location to write first model results to.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=constants.FIRST_MSS_YEAR,
        help="Start of time series to analyze.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=constants.FIRST_DISTURBANCE_YEAR - 1,
        help="End of time series to analyze (inclusive).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of image patches to process simultaneously.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=20,
        help="Max number of concurrent requests",
    )

    args, beam_args = parser.parse_known_args()

    run_pipeline(
        beam_args,
        input_asset=args.input_asset,
        output_prefix=args.output_prefix,
        model_checkpoint_path=args.model_checkpoint_path,
        model_one_output_prefix=args.model_one_output_prefix,
        start_year=args.start_year,
        end_year=args.end_year,
        batch_size=args.batch_size,
        max_requests=args.max_requests,
    )
