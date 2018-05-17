import torch

from torchcv.models import SSD300

d = torch.load('/scratch2/model_weights/vgg16-397923af.pth')
d_proc = {'.'.join(k.split('.')[1:]): v for k, v in d.items() if 'classifier' not in k}
net = SSD300(num_classes=12)
net.extractor.features.layers.load_state_dict(d_proc, strict=False, verbose=True)
torch.save(net.state_dict(), '/scratch2/model_weights/ssd300_12.pth')
