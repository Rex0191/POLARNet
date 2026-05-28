# POLARNet

Official implementation of **POLARNet**, a generative portrait relighting model built upon the POLAR OLAT data representation.

POLARNet predicts direction-aware OLAT lighting responses from a portrait input and can be used to synthesize realistic portrait relighting results under novel illumination. This repository provides the code, environment configuration, and scripts for running POLARNet.

> **Note:** Model checkpoints are not included in the current release. We will upload the pretrained checkpoints in a future update.

---

## 📢 News

- ✅ Code released.
- 🚧 Pretrained checkpoints will be released soon.

---

## ✅ TODO

- [ ] Release pretrained POLARNet checkpoints.
- [ ] Add detailed inference instructions.
- [ ] Add example input and output visualization.
- [ ] Add training data preparation instructions.

---

## 📁 Repository Structure

```text
.
├── scripts / shell files         # Training and inference scripts
├── configs                       # Configuration files
├── src / project modules          # Core implementation
├── requirements.txt              # Python dependencies
└── README.md
```

Some folders and files are intentionally excluded from this repository, as explained below.

---

## 📦 Missing Files and Folders

### `hugging_face/` Not Included

The `hugging_face/` folder is **not included** in this repository.

This folder usually contains local Hugging Face caches, such as pretrained model weights, tokenizer files, or other downloaded assets. These files are machine-specific and can be automatically downloaded or manually placed by users.

👉 Please modify the following line in `olatlight.sh` to point to your own local Hugging Face cache path:

```shell
export HF_HOME=/path/to/your/hugging_face
```

If you do not have a local cache, you may set `HF_HOME` to any writable directory where Hugging Face models can be downloaded.

---

### `.ckpts` / Checkpoints Not Included

Pretrained checkpoints are **not included** in the current repository due to storage limitations.

🚧 We will upload the official POLARNet checkpoints in a future release. Once available, we will update this README with:

- checkpoint download links;
- expected directory structure;
- inference commands;
- model version information.

Before the checkpoints are released, users may still inspect the codebase, prepare the environment, and adapt the training or inference scripts as needed.

---

## 🛠️ Environment Setup

We recommend using Conda to create an isolated Python environment.

```shell
conda create -n polarnet python=3.10 -y
conda activate polarnet

pip install -r requirements.txt
pip install -e .
```

If your environment requires a specific CUDA / PyTorch version, please install the corresponding PyTorch build before installing the remaining dependencies.

---

## 🤗 Hugging Face Cache Setup

Before running training or inference scripts, please set the Hugging Face cache path:

```shell
export HF_HOME=/path/to/your/hugging_face
```

Alternatively, you can modify the corresponding line in the provided shell script:

```shell
export HF_HOME=Your_path/hugging_face
```

Make sure the path is writable and has enough disk space for pretrained model files.

---

## 📥 Checkpoint Preparation

The official checkpoints are not yet included in this release.

After the checkpoints are released, place them under the expected checkpoint directory. For example:

```text
examples/inference/ckpts/polarnet.ckpt
```

The exact checkpoint path and script arguments will be updated once the pretrained models are available.

---

## 🚀 Usage

The current repository provides the code and scripts required by POLARNet. Please check the provided shell scripts for training or inference entry points.

For example:

```shell
bash olatlight.sh
```

Before running the script, please make sure that:

1. the environment has been correctly installed;
2. `HF_HOME` points to a valid local path;
3. the required checkpoints are available;
4. dataset and input paths in the script are correctly configured.

---


## 📚 Citation

If you find this repository useful, please consider citing our work.

```bibtex
@InProceedings{Chen_2026_CVPR,
    author    = {Chen, Zhuo and Yang, Chengqun and Su, Zhuo and Lv, Zheng and Gao, Jingnan and Zhang, Xiaoyuan and Yang, Xiaokang and Yan, Yichao},
    title     = {POLAR: A Portrait OLAT Dataset and Generative Framework for Illumination-Aware Face Modeling},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {28871-28881}
}
```

The citation entry will be updated after the paper information is finalized.

---

## 📞 Contact

For questions about setup, usage, or checkpoints, please open an issue in this repository.
