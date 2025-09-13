
# AEAFFNet: Enhancing Real-Time Semantic Segmentation through Attention-Enhanced Adaptive Feature Fusion

Authors: Shan Zhao, Wenjing Fu, Fukai Zhang, Zhanqiang Huo, Yingxu Qiao

> This project implements AEAFFNet, a real-time semantic segmentation network that leverages Attention-Enhanced Adaptive Feature Fusion to achieve high accuracy and efficiency.

## Overview

<p align="left">
  <img src="figures/0.png" alt="overview-of-our-method" width="500"/>
</p>

The network balances inference speed and segmentation accuracy on the Cityscapes dataset (illustrative figure).

## Directory Structure (Example)

```
AEAFFNet
├── mmsegmentation                # mmsegmentation submodule or directory
├── figures
│   └── 0.png
├── configs
│   └── aeaffnet
│       └── aeaffnet2_in1k-pre_4xb12-80k_cityscapes-512x1024.py
├── data
│   └── cityscapes
│       ├── leftImg8bit
│       │   ├── train
│       │   ├── val
│       └── gtFine
│           ├── train
│           ├── val
├── tools
│   ├── train.py
│   └── test.py
├── weights
└── README.md
```

## Environment

Recommended Python / PyTorch / mmsegmentation versions for compatibility:

```
python==3.8.10
pytorch==1.12.1
torchvision==0.13.1
mmengine==0.7.3
mmcv==2.0.0
mmsegmentation==1.0.0
```

> For other versions, please follow mmsegmentation's installation instructions.

## Installation

Please follow mmsegmentation's official guide to install dependencies:

* Clone or prepare mmsegmentation (as a submodule or via pip/PYTHONPATH).
* Install CUDA / cuDNN and compatible PyTorch version.
* Install mmcv, mmengine, mmsegmentation (versions above).

Example:

```bash
# Install mmcv (choose the correct CUDA/PyTorch version)
pip install mmcv-full==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cuXXX/torchX.Y/index.html
pip install -r requirements.txt
```

## Dataset

This repository only uses the Cityscapes dataset for training and evaluation.

* Download the dataset from the [Cityscapes website](https://www.cityscapes-dataset.com/downloads/) and arrange it under `data/cityscapes` following mmsegmentation requirements.
* CamVid and Pascal VOC are not included.

## Configuration File

Default configuration file:

```
configs/aeaffnet/aeaffnet2_in1k-pre_4xb12-80k_cityscapes-512x1024.py
```

The configuration assumes ImageNet or other pretraining (e.g., `in1k-pre`), 4 GPUs with batch size 12 per GPU, 80k iterations, input size 512x1024. Modify the config for custom training settings.

## Training

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/train.py configs/aeaffnet/aeaffnet2_in1k-pre_4xb12-80k_cityscapes-512x1024.py --work-dir ./weights/aeaffnet
```

### Multi GPU (Example: 2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/aeaffnet/aeaffnet2_in1k-pre_4xb12-80k_cityscapes-512x1024.py 2 --work-dir ./weights/aeaffnet
```

> For PyCharm, you can run `tools/train.py` directly with the config path.

## Testing / Inference

Single GPU test example:

```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/test.py configs/aeaffnet/aeaffnet2_in1k-pre_4xb12-80k_cityscapes-512x1024.py ./weights/aeaffnet/aeaffnet_latest.pth --eval mIoU
```

Notes:

* Replace `aeaffnet_latest.pth` with your saved checkpoint.
* For evaluation on the test server, follow Cityscapes official submission instructions.

## Results (Template)

Fill in real results after training and evaluation:

|   Method   | FPS | Params (M) | GFLOPs | ImageNet Pretrain | val (mIoU) | test (mIoU) |
| :--------: | :-: | :--------: | :----: | :---------------: | :--------: | :---------: |
| AEAFFNet |  84.2 |   23       |  82.8  |         ✓         |    79.1    |    77.9     |

## Notes for Reproducing Experiments

* Measure FPS on a fixed GPU (e.g., RTX 3090) and fixed input size (1024×2048 or 512×1024). Make sure to disable gradients (`torch.no_grad()`), set `model.eval()`, and exclude data loading time.
* Results vary if using different pretraining, longer training, or alternative augmentations.

## Citation

If this work is helpful for your research, please cite:

```
@misc{aeaffnet2024,
  title={AEAFFNet: Enhancing Real-Time Semantic Segmentation through Attention-Enhanced Adaptive Feature Fusion},
  author={Shan Zhao and Wenjing Fu and Fukai Zhang and Zhanqiang Huo and Yingxu Qiao},
  year={2024},
  note={Based on mmsegmentation implementation}
}
```

## Contact (Optional)

You may add author emails, affiliations, license info, or contributor notes here.

---

*This README is a template. Additional content such as experiment logs, CI badges, ONNX export, FP16 training, or deployment instructions can be added if needed.*
