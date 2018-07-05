# Fork of: 
## [TorchCV](https://github.com/kuangliu/torchcv): a PyTorch vision library mimics ChainerCV

I would start by uploading [these weights](https://github.com/pytorch/vision/blob/1fb0ccf71620d113cb72696b2eb8317b3e252cbb/torchvision/models/vgg.py#L15) and converting them to proper format with [this](https://github.com/VisionLearningGroup/visda-2018-public/blob/1ef76001b1c1763295926ffe5ae5765546080052/detection/pytorch-ssd-mmd-coral/examples/ssd/download_weights.py) script. The starting file that has model trained on source and evaluated on target is loceted [here](https://github.com/VisionLearningGroup/visda-2018-public/blob/1bd56019d21e5ec3b1b8f559d2a15b5bd2c7fc49/detection/pytorch-ssd-mmd-coral/train_visda_target17.py).
