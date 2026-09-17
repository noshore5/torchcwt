# Minimal image for benchmarking torch_cwt vs ptwt vs ssqueezepy.
# Nothing else -- no fcwt, no EEG_Benchmarks deps, no mamba image.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /bench

# Install this repo's torch_cwt package (editable) plus the two competitors.
COPY pyproject.toml /torchcwt/pyproject.toml
COPY src /torchcwt/src
# cupy-cuda12x is required alongside torch for ssqueezepy's SSQ_GPU=1 path
# (matches the base image's CUDA 12.1 toolkit).
RUN pip install --no-cache-dir -e /torchcwt \
    && pip install --no-cache-dir ptwt ssqueezepy cupy-cuda12x

COPY bench/make_dummy_data.py bench/cwt_benchmark.py /bench/

CMD ["python", "cwt_benchmark.py"]
