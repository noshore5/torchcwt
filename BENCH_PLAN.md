# Benchmark `torch_cwt` vs other PyTorch/GPU CWT implementations

## Context

`torch_cwt` (in `Documents/CoherIQs/torchcwt`, consumed by
`EEG_Benchmarks/utils/torch_cwt.py`) is a hand-written, torch.fft-based Morlet
CWT built as a numerical drop-in for `fcwt.cwt()`. There's no existing
comparison against other PyTorch-native CWT implementations on speed or
correctness at realistic EEG scales. This benchmark answers whether
`torch_cwt` is worth continuing to maintain vs adopting an existing library.

Keep this barebones: a minimal container with only the CWT libraries under
test, a synthetic dummy-data generator (no real EEG data needed for a pure
speed/shape benchmark), and one script. This belongs in the `torchcwt` repo
itself, not `EEG_Benchmarks` — that's the correct home for benchmarking the
module against its competitors.

## Candidate GPU-capable PyTorch CWT implementations to include

- **`torch_cwt`** (this repo) — the baseline.
- **`ptwt`** (PyTorch Wavelet Toolbox) — has `ptwt.cwt`, mirrors `pywt.cwt`'s
  `scales`/`wavelet` API, runs on any torch device including CUDA/MPS. Not
  currently installed anywhere.
- **`ssqueezepy`** — has an explicit torch/GPU backend (`ssqueezepy.TorchBackend`
  or `os.environ['SSQ_GPU']=1`) for its `cwt()`, and is a maintained,
  reasonably fast CWT implementation. Worth including as the second real
  competitor.
- **`kymatio`** — GPU-capable but implements wavelet *scattering transforms*,
  not a plain CWT; different math/output shape. Not a fair apples-to-apples
  comparison — exclude, mention only in the writeup as "not applicable."
- **`pytorch_wavelets`** (fbcotter) — GPU DWT/DTCWT, not a continuous CWT with
  a frequency axis — exclude for the same reason.
- `fcwt` — not torch/GPU (FFTW/CPU only, and doesn't build from PyPI on Apple
  Silicon) — keep only as an optional numeric-accuracy reference if easy to
  install, otherwise skip it entirely to keep this barebones.

So the concrete comparison set: **torch_cwt vs ptwt vs ssqueezepy (GPU
backend)**.

## Plan

### 1. Dummy data generator script (`torchcwt/bench/make_dummy_data.py`)
Generates synthetic multi-channel EEG-like signals only — no real data
dependency. Parametrized by n_channels, sampling_rate, duration_s, so the
benchmark can sweep realistic sizes (e.g. 23 channels x 256 Hz x
1-60 min). Pure numpy/torch, sine+noise mixes (same spirit as
`EEG_Benchmarks/scripts/continuous_cwt_scale_probe.py`'s synthetic fixture,
but reimplemented standalone in this repo — no cross-repo import).

### 2. Minimal Dockerfile (`torchcwt/bench/Dockerfile`)
Base: slim official `pytorch/pytorch` CUDA image (or plain `python:3.11-slim`
+ pinned torch wheel if CPU-only testing is enough locally). Installs only:
`torch`, this repo's `torch_cwt` package (editable install), `ptwt`,
`ssqueezepy`. Nothing else — no fcwt, no EEG_Benchmarks deps, no mamba image.

### 3. Benchmark script (`torchcwt/bench/cwt_benchmark.py`)
- Loads dummy data from `make_dummy_data.py`.
- For each backend (`torch_cwt`, `ptwt`, `ssqueezepy`-GPU) and each device
  available (cpu, then cuda/mps if present): run the CWT over a size sweep,
  time with `time.perf_counter()`, report peak memory
  (`torch.cuda.max_memory_allocated()` where applicable).
- Print one fixed-width stdout table: backend, device, shape, total_s,
  ms/channel, peak_MiB. No CSV/results-pipeline plumbing — plain print,
  consistent with "barebones."
- Optional lightweight correctness cross-check: compute all backends on the
  same synthetic chirp signal, report Pearson correlation of magnitude
  spectra between backends (aligned to the same log-frequency grid) — just
  enough to confirm they're not wildly wrong, not a rigorous parity study.

### 4. Run
Build and run the Docker image locally first (CPU) as a smoke test; if GPU
numbers matter, run the same image via the existing AWS GPU spot path
(`EEG_Benchmarks/scripts/eeg-run-spot.sh` / `eeg-run.yml -f kind=gpu`) — no
changes needed to that infra, this image is just handed to it as
`--docker-image`.

## Files touched (all in `Documents/CoherIQs/torchcwt`, new `bench/` dir)
- `bench/make_dummy_data.py` (new)
- `bench/Dockerfile` (new)
- `bench/cwt_benchmark.py` (new)

## Verification
`docker build -t torchcwt-bench bench/` then
`docker run --rm torchcwt-bench python cwt_benchmark.py` — confirm the table
prints for all three backends with sane timings and correlations.
