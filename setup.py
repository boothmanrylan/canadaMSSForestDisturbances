""" Setup script for Dataflow
"""

from setuptools import setup

setup(
    name="mss_forest_disturbances",
    url="https://github.com/boothmanrylan/canadaMSSForestDisturbances/tree/main/",
    packages=["mss_forest_disturbances"],
    install_requires=[
        "earthengine-api==0.1.358",
        "tensorflow==2.12.0",
        "msslib",
        "geemap",
        "matplotlib",
        "numpy",
    ],
)
