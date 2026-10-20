# Spectra Vision

<div align="center">
  <img src="docs/assets/logo.png" alt="Spectra logo" width="320" />

**Evidence-based review of recorded road footage.**
</div>

## What is Spectra?

Spectra analyses dashcam and road videos on your own computer and points out moments worth a closer look.
Each proposed moment comes with the evidence behind it, and a person decides what it is.

Proposals are not confirmed incidents. Spectra makes no accuracy, safety or fault claims.

## What can it do?

- Browse local folders and register videos in place, without copying or modifying them.
- Analyse a video in a separate worker process, with live progress and cancellation.
- Propose three kinds of moments:
  - apparent rapid approaches;
  - vehicles entering the camera's lane;
  - possible impacts involving the camera vehicle.
- Play the source video with recorded detection and lane overlays, and an interactive event timeline.
- Show each proposal's evidence, its uncertainty and what could not be observed.
- Let a reviewer accept or reject each proposal, adjust its times and add a note.
- Cut a clip of any proposal and export an analysis with its review work as JSON.
- Follow the system's light or dark theme, or use either one.

Everything runs on `127.0.0.1`. There are no accounts, cloud services or telemetry.

## How does it work?

1. The video is decoded by presentation timestamp.
2. A detector finds road users and a tracker follows them.
3. A segmentation model finds the road, lane markings and the camera vehicle's own hood. From these, the
   lane geometry and each vehicle's lane are estimated.
4. Event analysers read these records and propose moments, each with its evidence.
5. The web application presents the proposals for human review.

Analysis results are cached and verified by hash, so the same video and settings always give the same
proposals.

## Technologies

- **Frontend:** React 19, TypeScript, Vite, TanStack Query
- **Backend:** Python 3.12, FastAPI, SQLite
- **Video:** PyAV (FFmpeg)
- **Models:** ONNX Runtime on the CPU
  - D-FINE-N detector
  - comma10k SegNet segmentation
  - BoT-SORT tracking (`trackers`)
- **Tests:** pytest and Vitest

## Installation

Requirements:

- Python 3.12
- Node.js 22.12 or newer, to build the web application
- The two model files described below
- macOS on Apple Silicon is the tested platform. No GPU is needed.

Install the Python dependencies from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

Start Spectra:

```bash
python3 run.py
```

On first start, the launcher installs and builds the web application. Spectra then opens at
`http://127.0.0.1:8765`. Add a folder of videos on the Library page. Ctrl+C stops the app.

Useful options:
- `--port N`
- `--no-browser`
- `--library DIR`, which offers a folder from the start
- `--dev`, which runs the frontend development server with live reload

### Model files

The model weights are not part of this repository. Spectra needs two ONNX files in `models/engine/`,
exported from upstream checkpoints that you download yourself:

| File | Purpose | Upstream checkpoint |
|---|---|---|
| `dfine_n_coco_640.onnx` | Detects vehicles and other road users | `dfine_n_coco.pth` from [D-FINE](https://github.com/Peterande/D-FINE) |
| `comma10k_segnet_512x384.onnx` | Segments the road, lane markings and the camera vehicle | [commaai/comma10k-segnet](https://huggingface.co/commaai/comma10k-segnet) |

The export uses its own environment, `evaluation/.venv`, with PyTorch. Its pins were resolved on
macOS arm64 with Python 3.12. From the repository root:

```bash
# Export environment
python3.12 -m venv evaluation/.venv
evaluation/.venv/bin/python -m pip install -r evaluation/requirements-export-isolated.lock.txt

# Upstream sources and checkpoints, at the supported versions
git clone https://github.com/Peterande/D-FINE.git evaluation/.venv/src/D-FINE
git -C evaluation/.venv/src/D-FINE checkout 956d1709314c2c6a4df6f34de232054578a7449f
mkdir -p evaluation/.venv/weights
curl -L -o evaluation/.venv/weights/dfine_n_coco.pth \
  https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth
evaluation/.venv/bin/hf download commaai/comma10k-segnet \
  --revision b642c614eb0737fe7befa229f47fbd0245033038 --local-dir evaluation/weights/comma10k-segnet
grep -E 'dfine_n_coco|comma10k-segnet' evaluation/checkpoint_hashes.txt | shasum -a 256 -c -

# Export to ONNX
evaluation/.venv/bin/python -m evaluation.export_dfine_onnx models/engine/dfine_n_coco_640.onnx
evaluation/.venv/bin/python -m evaluation.export_segnet_onnx models/engine/comma10k_segnet_512x384.onnx
```

Each export prints the SHA-256 of its file. Spectra accepts a model only if this hash matches the one
in [models/engine/models.json](models/engine/models.json). Byte-identical exports have not been verified outside
the original export machine. If the hashes differ on your machine, Spectra refuses the files.

Without the models, Spectra still starts and existing reviews remain available. New analyses are
unavailable until the models are present. The [evaluation guide](evaluation/README.md#environments-and-reproducibility)
has more about the export environment.

The checkpoints come under their upstream terms, not under Spectra's licence. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Basic checks

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q -m "not slow and not model and not real_video"
cd frontend && npm test && npm run build
```

Model tests need the model files. Real-footage tests run only with an explicit `-m real_video` and need
the local evaluation datasets.

## Data and privacy

Spectra stores its database, analysis cache and evidence in `var/`, next to `run.py`. This folder is
ignored by Git and created on first start with owner-only permissions. Delete it to start over.

Videos stay where they are and are only read. Only videos your browser can play can be reviewed;
nothing is transcoded.

## Limitations

- The detection, lane and event methods have not been validated against human-labelled footage. Treat
  every proposal as a suggestion to review.
- Lane geometry assumes a locally flat road and is not metric. Spectra estimates no distances or speeds.
- Analysis uses the CPU and runs one video at a time. Memory safeguards may refuse or stop an analysis
  under heavy memory pressure.

## Project structure

```text
Spectra/
├── spectra/
│   ├── engine/          # decoding, perception, lanes, event analysers, cache, worker
│   ├── evidence/        # evidence packages, timeline, overlays, clips
│   └── api/             # local FastAPI server and SQLite state
├── frontend/            # React web application
├── evaluation/          # datasets, scoring, detector comparison, model export
├── models/engine/       # runtime model registry (weights are local)
├── tests/               # Python tests
└── run.py               # launcher
```

Further reading: [ARCHITECTURE.md](ARCHITECTURE.md), the [engine](spectra/engine/README.md),
[evidence](spectra/evidence/README.md) and [API](spectra/api/README.md) references, the
[web application](frontend/README.md) and the [evaluation guide](evaluation/README.md).

## Licence

Spectra's source code is released under the [MIT Licence](LICENSE). The licence does not cover
third-party model weights, datasets, media or dependencies, which have their own terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

<div align="center">
  <p>Developed by <a href="https://heykaan.dev">heykaan.dev</a>.</p>
</div>
