FROM apache/beam_python3.10_sdk:2.50.0

COPY setup.py ./
COPY mss_forest_disturbances/ ./mss_forest_disturbances/

RUN pip install .
