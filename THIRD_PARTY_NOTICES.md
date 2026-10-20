# Third-party notices

Spectra's own source code is MIT-licensed ([LICENSE](LICENSE)). That licence covers only Spectra's code and
documentation. It does not license any third-party model weights, datasets, media or binaries.

This repository contains no third-party model weights, datasets, video footage or compiled binaries. Users
install the dependencies and obtain the model checkpoints themselves, from the upstream sources below and
under the upstream terms. Redistribution of the model weights by this project has not been cleared.

## Models

The runtime models are ONNX exports of the checkpoints below. The export scripts are part of Spectra. The
checkpoints and exports are not. [models/engine/models.json](models/engine/models.json) pins the checkpoint
and export hashes.

### D-FINE-N (COCO), `dfine_n_coco_640.onnx`

- **Code:** [Peterande/D-FINE](https://github.com/Peterande/D-FINE), Apache-2.0, used at commit
  `956d1709314c2c6a4df6f34de232054578a7449f`. Spectra does not include D-FINE code. The export loads a
  local checkout.
- **Checkpoint:** `dfine_n_coco.pth`, from the project's official release
  `https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth` (SHA-256
  `41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c`). No separate licence for the
  checkpoint has been found. Treating it as Apache-2.0 is an assumption, not an established fact.
- **Objects365:** the D-FINE README warns that its `*_obj365` and `*_obj2coco` checkpoints may be subject
  to Objects365 terms. Spectra uses the COCO-only `*_coco` checkpoint.
- **Training data:** COCO. The COCO annotations are licensed CC BY 4.0 by the COCO Consortium. The images
  are not owned by the consortium and are subject to the Flickr Terms of Use
  ([cocodataset.org](https://cocodataset.org/#termsofuse)).
- **Unresolved:**
  - The HGNetV2 backbone was initialised from `PPHGNetV2_B0_stage1.pth`, from the same release.
    Its pretraining data and terms are undocumented.
  - Whether any training-image terms carry over to the trained weights.

### comma10k SegNet, `comma10k_segnet_512x384.onnx`

- **Checkpoint:** [commaai/comma10k-segnet](https://huggingface.co/commaai/comma10k-segnet), revision
  `b642c614eb0737fe7befa229f47fbd0245033038`. The model card declares the MIT licence, and the
  repository is not gated. It has no separate licence file. `model.safetensors` has SHA-256
  `8208672861ad1b111dc98f3a7c54196d29875b709c7353e2dd1b7614343fb3a8`.
- **Training data:** [comma10k](https://github.com/commaai/comma10k), images from the comma fleet with
  community-made labels. Its README states an MIT licence "with no academic only restrictions".
- **Export tooling** (export environment only): `segmentation-models-pytorch` (MIT) and `timm`
  (Apache-2.0).
- **Unresolved:**
  - The training recipe and encoder initialisation are not documented.
  - Neither the model nor the dataset publishes a statement about the permissions of the people who
    recorded or appear in the images.

## Python packages

`pip` installs these from PyPI. The repository does not include them. Versions are pinned in
[pyproject.toml](pyproject.toml).

| Package | Licence |
|---|---|
| av (PyAV) 19.0.1 | BSD-3-Clause; the wheel bundles FFmpeg (see below) |
| onnxruntime 1.25.1 | MIT |
| opencv-python 4.11.0.86 | Apache-2.0; the wheel bundles FFmpeg and other libraries under their own licences |
| numpy 2.2.6 | BSD-3-Clause |
| pydantic 2.13.3, fastapi 0.136.1, supervision 0.30.9 | MIT |
| uvicorn 0.46.0 | BSD-3-Clause |
| trackers 2.6.1 (BoT-SORT) | Apache-2.0 |

Their transitive dependencies are permissively licensed, apart from certifi and tqdm (MPL-2.0, with tqdm
also MIT). The `eval` extra adds scipy (BSD-3-Clause). The `dev` extra adds pytest (MIT) and httpx (BSD-3-Clause).
The model-export environment is separate: it is pinned in
[evaluation/requirements-export-isolated.lock.txt](evaluation/requirements-export-isolated.lock.txt) and
includes PyTorch (BSD-3-Clause).

### PyAV and FFmpeg

Spectra declares PyAV as a dependency. It neither includes nor redistributes the PyAV wheel or any
FFmpeg library. The application uses PyAV to:
- decode video and probe streams;
- cut review clips by copying compressed packets into a new container (no re-encoding).

It does not call a video encoder.

The inspected PyAV 19.0.1 wheel (macOS arm64) bundles FFmpeg configured with `--enable-version3` and
without `--enable-gpl`, and reports "LGPL version 3 or later". The same build enables and bundles
libx264 and libx265, which are GPL-licensed. FFmpeg's documentation says those libraries require a GPL
build ([ffmpeg.org/legal.html](https://ffmpeg.org/legal.html)). This inconsistency belongs to the upstream wheel and
is not resolved here.

Declaring a dependency on PyAV does not make Spectra's source code GPL-licensed. Distributing the
wheel or FFmpeg binaries with Spectra would require a separate assessment, whatever Spectra's own code
does. That assessment would cover the LGPL and GPL obligations of the bundled libraries.

### Codec patents

Patent rights for video formats such as H.264 and HEVC are separate from software copyright licences. No
open-source licence grants them. Spectra's source code contains no codec. Decoding is done by the
libraries each user installs. Patent licensing has not been assessed for any future distribution of
binaries.

## Web application

The production build bundles the following packages. `frontend/dist` is not part of the repository.

| Package | Licence |
|---|---|
| react, react-dom, scheduler | MIT |
| react-router, cookie, set-cookie-parser | MIT |
| @tanstack/react-query, @tanstack/query-core | MIT |
| lucide-react (icons) | ISC |

Build and test tools are used only during development. Their licences are permissive, apart from
lightningcss (MPL-2.0), a build tool whose code is not bundled.

## Evaluation material

The repository contains no evaluation footage, datasets or comparison checkpoints. It does contain the
following, which are metadata only:
- clip manifests: download URLs, licence labels and hashes;
- draft annotations: timestamps and notes;
- a licensing registry recording each source's terms ([evaluation/registry/registry.json](evaluation/registry/registry.json)).

The `evaluation/importers` code reads the DoTA and PREVENTION annotations. Imported labels are kept out of
the repository.

## Project assets

The Spectra logo (`docs/assets/logo.png`) is original work by the project author and is distributed with
the source code.
