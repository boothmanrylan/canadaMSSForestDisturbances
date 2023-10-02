"""Helpful methods for Dataflow jobs.
"""

import google
import ee

from . import constants


def ee_init():
    """Initializes and authenticates earth engine.

    Returns:
        None
    """
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url=constants.HIGH_VOLUME_ENDPOINT,
    )


def build_request(cell, size, file_format="NPY"):
    """Generates a base request for computePixels.

    The request is returned in the default projection and scale as defined in
    constants.py

    Args:
        cell: ee.Feature, containes the geometry to get the request for.
        size: int, the height/width in pixels of the request to return.
        file_format: str, the type of image file format to return.

    Returns:
        dict, the request to pass to computePixels, has key/values fileFormat
        and grid, use must supply expression later.
    """
    # get top left corner of cell in *unscaled* projection
    proj = ee.Projection(constants.PROJECTION)
    coords = cell.geometry(1, proj).getInfo()["coordinates"][0][3]
    request = {
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {
                "width": size,
                "height": size,
            },
            "affineTransform": {
                "scaleX": constants.SCALE,
                "shearX": 0,
                "translateX": coords[0],
                "shearY": 0,
                "scaleY": -constants.SCALE,
                "translateY": coords[1],
            },
            "crsCode": proj.getInfo()["crs"],
        },
    }

    return request


