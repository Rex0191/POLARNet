# OLAT Data Processing Pipeline

This repository provides a data processing pipeline for OLAT (One-Light-At-a-Time) portrait data. It converts raw captures into processed OLAT images, synthesizes uniform-light composites for matting, generates alpha masks through an external matting system, and finally produces relit synthetic images under colored HDR environment lighting.

The project is intended for research and dataset preparation workflows, especially for portrait relighting, matting, and synthetic data generation.

---

## Overview

The full pipeline consists of four main stages:

1. **Raw OLAT preprocessing**  
   Decode JPEG images from parquet files and apply image normalization operations.

2. **Uniform-light synthesis for segmentation / matting**  
   Reconstruct a more evenly illuminated portrait image from OLAT observations and composite it with a background.

3. **Alpha matte generation**  
   Use an external matting model to obtain foreground alpha masks.

4. **Colored relighting and final compositing**  
   Relight the portrait under HDR environment maps and blend the relit foreground with matching backgrounds.

---

## Repository Structure

```text
.
├── preprocess.py
├── pipeline.sh
├── ids.txt
├── light/
│   ├── light_157.txt
│   ├── OLAT_EnvMaps/
│   └── ...
├── segment/
│   ├── uniform_with_bg.py
│   ├── uniform_with_bg_gamma.py
│   └── batch_uniform_with_bg.py
├── color/
│   ├── synthesis_color_light.py
│   └── batch_synthesis_color_light.py
└── synthetic_light/
    ├── studio_hdr_generator.py
    ├── studio_hdr_generator_full.py
    ├── make_studio_hdrs.py
    ├── make_strong_hdr_scenes.py
    ├── *.json
    └── color_light.sh
```

---

## Pipeline Stages

### 1. Raw OLAT Preprocessing 

Script: `preprocess.py`

This stage reads raw `.parquet` files, decodes JPEG images in memory, and writes processed image files to disk.

The script performs the following operations:

- gamma correction
- contrast enhancement (`clahe` or `linear`)
- 90-degree camera-dependent rotation
- downsampling to one quarter of the original resolution

It only processes OLAT data and skips non-OLAT entries such as PBR samples.

#### Main features

- recursive search for parquet files
- optional preservation of the `info` path structure from the parquet metadata
- overwrite support
- camera-specific rotation rules

#### Example

```bash
python preprocess.py \
  -i /path/to/raw_olat \
  -o /path/to/ori_OLAT \
  -r \
  --keep-structure \
  --gamma 1.0 \
  --contrast linear \
  --alpha 1.0 \
  --beta 0 \
  --overwrite
```

---

### 1.5. Directory Structure Cleanup

Script location: `pipeline.sh` (`task_fix_structure`)

When preprocessing is executed with `--keep-structure`, some outputs may be stored under intermediate folders such as `Photos/`. This cleanup step removes the `Photos/` layer by moving its contents to the parent directory.

This normalization step makes the downstream directory layout more consistent for later processing.

---

### 2. Uniform-Light Synthesis for Matting

Core scripts:

- `segment/uniform_with_bg_gamma.py`
- `segment/batch_uniform_with_bg.py`

This stage reconstructs a uniformly lit portrait image from the OLAT stack and composites it with a background. The resulting images are used as input for foreground matting.

#### Core idea

Given:

- a set of OLAT images
- a text file describing image names and light indices (`olat_txt`)
- base light maps in `light/OLAT_EnvMaps/`
- a target environment map

The script:

1. loads all OLAT images listed in `olat_txt`
2. loads the corresponding base maps
3. computes per-light weights from the target environment map
4. linearly combines the OLAT images using these weights
5. applies gamma stretching to the result
6. writes the output as `*_composite.png`

#### Batch processing capabilities

`batch_uniform_with_bg.py` supports:

- traversing all IDs / sessions / camera folders
- skipping already processed outputs
- random sampling of environment maps
- multi-process parallel execution

#### Example

```bash
python segment/batch_uniform_with_bg.py \
  --root_dir /path/to/ori_OLAT \
  --root_out /path/to/for_segmentation \
  --olat_txt light/light_157_proc.txt \
  --base_map_dir light/OLAT_EnvMaps \
  --envmap_dir /path/to/hdrs_uniform \
  --background_dir /path/to/hdr_background \
  --num_envmaps 1 \
  --num_workers 16
```

---

### 3. Alpha Matte Generation

This repository does **not** implement the matting model itself. Instead, `pipeline.sh` calls an external matting project, currently **Matte-Anything**, to generate alpha masks.

Relevant pipeline entry:

- `task_matting` in `pipeline.sh`

Typical input / output:

- input: the uniform-light composites from Step 2
- output: alpha mask images, typically named `*_alpha.png`

Example command from the pipeline:

```bash
python batch_matte.py \
  -i /path/to/for_segmentation \
  -o /path/to/alpha_data \
  -r --keep_structure \
  --fg-caption "person" \
  --gpu-ids 1,2,3,4
```

> Step 4 depends on the alpha masks generated in this stage.

---

### 4. Colored Relighting and Final Compositing

Core scripts:

- `color/synthesis_color_light.py`
- `color/batch_synthesis_color_light.py`

This is the final synthesis stage of the pipeline.

#### Core idea

For each HDR environment map:

1. resize the environment map to the base-map resolution
2. compute two types of relighting weights:
   - **diffuse weights**, derived from environment intensity
   - **specular weights**, derived from RGB environment color
3. accumulate all OLAT images using these weights
4. load the matching background image
5. load the alpha mask generated in Step 3
6. composite the relit foreground with the background
7. save the final result as `*_composite.png`

The relighting formulation separates diffuse-like and specular-like contributions so that skin appearance remains more natural while colored highlights are preserved.

#### Batch processing capabilities

`batch_synthesis_color_light.py` supports:

- automatic inference of output directories from input directory structure
- automatic inference of alpha directories from input directory structure
- random sampling of environment maps
- multi-process parallel execution
- skipping already generated composites

#### Example

```bash
python color/batch_synthesis_color_light.py \
  --root_dir /path/to/ori_OLAT \
  --root_alpha /path/to/alpha_data \
  --root_out /path/to/synthetic_image \
  --olat_txt light/light_157_proc.txt \
  --base_map_dir light/OLAT_EnvMaps \
  --envmap_dir /path/to/hdrs_all \
  --background_dir /path/to/hdr_background \
  --num_envmaps 800 \
  --num_workers 16
```

---

## Main Pipeline Script

The end-to-end workflow is organized in `pipeline.sh`.

It defines the following tasks:

- `task_preprocess`
- `task_fix_structure`
- `task_uniform_with_bg`
- `task_matting`
- `task_color_light`

Each stage can be enabled or disabled through boolean flags in the script:

```bash
SKIP_STEP1=true
SKIP_STEP15=false
SKIP_STEP2=false
SKIP_STEP3=false
SKIP_STEP4=false
```

Run the full pipeline with:

```bash
bash pipeline.sh
```

---

## HDR Environment Map Generation Utilities

The `synthetic_light/` directory contains helper scripts for generating synthetic studio-style HDR environment maps used by the relighting pipeline.

### Included utilities

#### `studio_hdr_generator.py`
A configurable HDR generator for producing studio-style environment maps from JSON scene descriptions.

#### `studio_hdr_generator_full.py`
A more complete generator that supports richer light shapes, including:

- `gaussian`
- `rect`
- `greatcircle`
- `sector`

#### `make_studio_hdrs.py`
A fast utility for generating many random studio-style HDR maps in batch.

#### `make_strong_hdr_scenes.py`
A scene generator that creates stronger, higher-contrast, multi-directional HDR lighting setups and exports them as JSON scene descriptions.

#### Example commands

```bash
python synthetic_light/make_studio_hdrs.py --out_dir ./out_hdrs --count 400
```

```bash
python synthetic_light/make_strong_hdr_scenes.py --out ./scenes.json --count 400
python synthetic_light/studio_hdr_generator.py --out ./studio_hdr_out --scene-json ./scenes.json
```

---

## Input / Output Summary

### Step 1

- **Input:** raw `.parquet` OLAT data
- **Output:** processed JPG images under `ori_OLAT/`

### Step 2

- **Input:** `ori_OLAT/`
- **Output:** uniform-light composite images under `for_segmentation/`

### Step 3

- **Input:** `for_segmentation/`
- **Output:** alpha mask images under `alpha_data/`

### Step 4

- **Input:**
  - processed OLAT images in `ori_OLAT/`
  - alpha masks in `alpha_data/`
  - HDR environment maps
  - matching background images
- **Output:** final relit composite images under `synthetic_image/`

---

## Dependencies

Based on the current scripts, the main Python dependencies are:

- `numpy`
- `opencv-python`
- `imageio`
- `pandas`
- `pyarrow`
- `tqdm`

The complete pipeline also depends on:

- GNU `parallel`
- `conda`
- an external matting project such as **Matte-Anything**

The existing shell script suggests separate environments for different stages, for example:

- `delit` for preprocessing and relighting
- `lbm` for matting

---

## Practical Notes

1. **Keep directory structures aligned**  
   The pipeline infers output and alpha paths from relative directory structure. If `root_dir`, `root_out`, and `root_alpha` do not match consistently, downstream scripts may fail to find masks or write outputs to the intended locations.

2. **Start with a small-scale test**  
   Before running on all IDs and cameras, test with a small subset to verify path conventions, environment maps, mask naming, and output quality.

3. **Background file names must match environment map names**  
   Step 4 expects background files using the same stem as the environment maps, for example:
   - environment map: `example.hdr`
   - background image: `example.png`

4. **Alpha masks are expected as `*_alpha.png`**  
   The color relighting script searches for alpha masks following this naming convention.

5. **Existing outputs are skipped automatically**  
   Batch scripts detect existing `*_composite.png` files and skip those directories to avoid recomputation.

---

## Suggested Improvements

For a cleaner open-source release, the following improvements would be helpful:

- add a `requirements.txt` or `environment.yml`
- move hard-coded paths in `pipeline.sh` to command-line arguments or a config file
- document the expected dataset directory format more explicitly
- provide example input / output visualizations
- clarify differences between alternative script variants such as gamma / compatibility versions

---

## Acknowledgments

This repository focuses on OLAT decoding, relighting, compositing, and synthetic HDR lighting generation. The alpha matting stage relies on an external matting framework.
