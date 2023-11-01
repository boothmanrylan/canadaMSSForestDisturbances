"""Launches a Beam/Dataflow job to generate maps using a trained model.
"""

from absl import app
from absl import flags

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.ml.inference.base import RunInference, KeyedModelHandler
from apache_beam.ml.inference.tensorflow_inference import (
    TFModelHandlerTensor,
    ModelType,
)

import xarray as xr
import xbatcher
import xarray_beam as xbeam

import tensorflow as tf

import ee

from msslib import msslib
from mss_forest_disturbances import (
    constants,
    preprocessing,
    dataflow_utils,
    model,
    dataset,
)

_INPUT_ASSET = flags.DEFINE_string(
    "input_asset",
    "",
    help="Path to an EE FeatureCollection defining the geometry to run over.",
)

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    "",
    help="Path to a Google Cloud bucket loadtion to write results to.",
)

_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path",
    "",
    help="Path to a Google Cloud bucket location containing model weights.",
)

_TRAINING_DATA_PATTERN = flags.DEFINE_string(
    "trianing_data_pattern",
    "",
    help="Unix style file pattern for training data, used to adapt normalization layer.",
)

_START_YEAR = flags.DEFINE_integer(
    "start_year",
    constants.FIRST_MSS_YEAR,
    help="First year to get images for (inclusive).",
)

_END_YEAR = flags.DEFINE_integer(
    "end_year",
    constants.FIRST_DISTURBANCE_YEAR - 1,
    help="Last year to get image for (inclusive).",
)

_MAX_REQUESTS = flags.DEFINE_integer(
    "max_requests", 32, help="Maximum number of concurrent requests to earth engine."
)

_CHUNKS_STR = flags.DEFINE_string(
    "chunks_string",
    "time=32,X=1024,Y=1024",
    help=("comma separated dimension=size pairs defining how to chunk the dataset"),
)


class XarrayModelWrapper(tf.keras.Model):
    """Convert xarray inputs to tensors and model outputs to xarray.Dataset."""

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def call(self, batch):
        inputs = tf.convert_to_tensor(batch.to_array().data)
        outputs = self.model(inputs, training=False)
        return xr.Dataset(
            outputs,
            coords=batch.coords,
            dims=batch.dims,
        )


class TFNoBatchModelHandler(TFModelHandlerTensor):
    """Model handler class that overrides batching."""

    def batch_element_kwargs(self):
        return {"max_batch_size": 1}


def keyed_batch_generator(elem, **batch_args):
    key, chunk = elem
    for batch in xbatcher.BatchGenerator(chunk, **batch_args):
        yield key, batch


def collapse_batches(elem):
    key, batches = elem
    collapsed_batches = xr.combine_by_coords(
        [batch.drop_duplicates(dim=...) for batch in batches]
    )
    return key, collapsed_batches


def create_model_fn(*args, **kwargs):
    m = model.build_model(*args, **kwargs)
    return XarrayModelWrapper(m)


def _parse_chunks_str(chunks_str):
    """Convert a str to a dict.

    Borrowed from xbeam examples by way of xee examples.

    e.g. converts "a=1,b=2,c=3" into this dict: {"a": 1, "b": 2, "c": 3}
    """
    chunks = {}
    parts = chunks_str.split(",")
    for part in parts:
        k, v = part.split("=")
        chunks[k] = int(v)
    return chunks


def main(argv):
    dataflow_utils.ee_init()

    input_geometry = ee.FeatureCollection(_INPUT_ASSET.value).geometry()
    collection = msslib.getCol(
        aoi=input_geometry,
        yearRange=[_START_YEAR.value, _END_YEAR.value],
        doyRange=constants.DOY_RANGE,
        maxCloudCover=100,
    ).map(preprocessing.prepare_image)

    ds = xr.open_dataset(
        collection,
        engine="ee",
        crs=constants.PROJECTION,
        scale=constants.SCALE,
        geometry=input_geometry,
    )

    template = xbeam.make_template(ds)

    beam_options = PipelineOptions(
        argv,
        save_main_session=True,
        max_num_workers=_MAX_REQUESTS.value,
        direct_num_workers=max(_MAX_REQUESTS.value, 20),
        disk_size_gb=50,
    )

    _, normalization_subset = dataset.build_dataset(
        _TRAINING_DATA_PATTERN.value, constants.DEFAULT_PARSE_OPTIONS, training=True
    )

    model_handler = KeyedModelHandler(
        TFNoBatchModelHandler(
            _CHECKPOINT_PATH.value,
            model_type=ModelType.SAVED_WEIGHTS,
            create_model_fn=lambda _: create_model_fn(
                normalization_subset=normalization_subset,
                **constants.DEFAULT_MODEL_OPTIONS
            ),
        )
    )

    batch_args = {
        "input_dims": {"X": constants.PATCH_SIZE, "Y": constants.PATCH_SIZE},
        "input_overlap": {"X": constants.OVERLAP, "Y": constants.OVERLAP},
        "batch_dims": {"time": constants.BATCH_SIZE},
        "pad_input": True,
        "drop_remainder": False,
    }

    chunk_schema = _parse_chunks_str(_CHUNKS_STR.value)

    with beam.Pipeline(options=beam_options) as pipeline:
        (
            pipeline
            | "To Chunks" >> xbeam.DatasetToChunks(ds, chunk_schema, split_vars=False)
            | "Batch" >> beam.FlatMap(keyed_batch_generator, **batch_args)
            | "RunInference" >> RunInference(model_handler)
            | "GroupByKey" >> beam.GroupByKey()
            | "Unbatch" >> beam.Map(collapse_batches)  # TODO: second model?
            | "Write" >> xbeam.ChunksToZarr(_OUTPUT_PATH.value, template, chunk_schema)
        )


# TODO: things that need to be tested:
# 1. xee with msslib addBands error
# 2. DatasetToChunks possibly throwing out partial batches
# 3. Dataset creation in XarrayModelWrapper has the right coordinates and dims
# 4. Model output has multiple bands: is each a DataArray? should they be # named?
# 5. Certain time steps in xee will be all NaN
# 6. collapse_batches
# 7. optimal chunk size
# 8. does Zarr store all the coordinate/time information?
# 9. convert Zarr to format than can be ingested by earth engine
    # see here: https://stackoverflow.com/questions/69228924/
    # see here: https://corteva.github.io/rioxarray/html/examples/convert_to_raster.html
    # see here: https://corteva.github.io/rioxarray/html/examples/dask_read_write.html


if __name__ == "__main__":
    flags.mark_flags_as_required(["input_asset", "output_path", "checkpoint_path"])
    app.run(main)
