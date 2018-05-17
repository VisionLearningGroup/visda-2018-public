from __future__ import print_function

import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import tqdm

from PIL import Image, ImageDraw
from itertools import islice
from torch.autograd import Variable

from torchcv.evaluations import voc_eval
from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.ssd import SSD300, SSD512, SSDBoxCoder

from torchcv.loss import SSDLoss
from torchcv.datasets import ListDataset
from torchcv.transforms import resize, random_flip, random_paste, random_crop, \
    random_distort

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model',
                    default='/scratch2/model_weights/ssd300_12_imagenet.pth',
                    type=str, help='initialized model path')
parser.add_argument('--checkpoint',
                    default='/scratch/run/pyssd/ssd300_12_source.pth',
                    type=str, help='checkpoint path')
args = parser.parse_args()

# Model
print('==> Building model..')
num_classes = 12
net = SSD300(num_classes=num_classes)
net.load_state_dict(torch.load(args.model))
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# Dataset
print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net)
img_size = 300

cat_of_interest = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
                   'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


unnormalize = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def transform_train(img, boxes, labels):
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size, img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    enc_boxes, enc_labels = box_coder.encode(boxes, labels)
    return img, enc_boxes, enc_labels



trainset = ListDataset(
        root='/scratch2/mytmp/render_detection_visda',
        list_file='/scratch2/mytmp/render_detection_result/listdataset/visda18'
                  '-detection-train.txt',
        transform=transform_train
)

testset = ListDataset(
        root='/scratch2/mytmp/render_detection_visda',
        list_file='/scratch2/mytmp/render_detection_result/listdataset/visda18'
                  '-detection-test.txt',
        transform=transform_test
)

cocoset = ListDataset(
        root='/scratch2/data/coco/train2014',
        list_file='/scratch2/mytmp/render_detection_result/listdataset/coco-train-short.txt',
        transform=transform_test
)

batch_size = 32
trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8
)

# train_evalloader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=False, num_workers=1
# )
testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=0
)

cocoloader = torch.utils.data.DataLoader(
        cocoset, batch_size=32, shuffle=False, num_workers=0
)


def box_label_list(root_dir, list_file):
    return list(ListDataset(
            root=root_dir,
            list_file=list_file,
            transform=lambda img, boxes, labels:
                (resize(img, boxes, size=(img_size, img_size))[1], labels)
    ))


vda_test_boxes_labels = box_label_list(
        root_dir='/scratch2/mytmp/render_detection_visda',
        list_file='/scratch2/mytmp/render_detection_result/listdataset/'
                  'visda18-detection-test.txt'
)

coco_test_boxes_labels = box_label_list(
        root_dir='/scratch2/data/coco/train2014',
        list_file='/scratch2/mytmp/render_detection_result/listdataset/'
                  'coco-train-short.txt',
)


net.cuda()
net = torch.nn.DataParallel(net, device_ids=[0, 1])
cudnn.benchmark = True

criterion = SSDLoss(num_classes=num_classes)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


def draw_boxes(input_, loc, cls, path, name):
    boxes, labels, scores = box_coder.decode(loc, cls)
    img: Image = transforms.ToPILImage()(unnormalize(input_))
    draw = ImageDraw.Draw(img)
    if boxes is not None:
        for box, label in zip(boxes, labels):
            draw.rectangle(list(box), outline='red')
            draw.text((box[0], box[1]), cat_of_interest[label])
    else:
        print('no boxes found')
    os.makedirs(path, exist_ok=True)
    img.save(os.path.join(path, name+'.png'))


batch_n_max_count = 10000000


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_slice = islice(enumerate(trainloader), batch_n_max_count)
    for batch_idx, (inputs, loc_targets, cls_targets) in train_slice:
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], train_loss / (batch_idx + 1), batch_idx + 1,
                 len(trainloader)))
        

def test(name, epoch_i, dataloader, test_boxes_labels,
         samples_to_draw, image_folder, checkpoint=False):
    print('\nTest')
    net.eval()
    test_loss = 0
    pred_boxes, pred_labels, pred_scores = [], [], []
    gt_boxes, gt_labels = [], []
    for batch_idx, data_point in islice(enumerate(dataloader), batch_n_max_count):
        inputs, loc_targets, cls_targets = data_point
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        print('{name}: ', end='')
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print(f'train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], test_loss / (batch_idx + 1), batch_idx + 1,
                 len(dataloader)))

        for input_i in range(inputs.size(0)):
            if (batch_idx, input_i) in samples_to_draw:
                draw_boxes(inputs[input_i].data.cpu(), loc_preds[input_i].data.cpu(),
                           F.softmax(cls_preds[input_i].float(), dim=1).data.cpu(),
                           image_folder,
                           f'{batch_idx}-{input_i}-{epoch_i}-pred')

            box_preds, label_preds, score_preds = box_coder.decode(
                    loc_preds[input_i].data.cpu(),
                    F.softmax(cls_preds[input_i].float(), dim=1).data.cpu(),
                    score_thresh=0.01, nms_thresh=0.0
            )
            
            loc_unenc, cls_unenc = test_boxes_labels[batch_idx*batch_size+input_i]
        
            pred_boxes.append(box_preds)
            pred_labels.append(label_preds)
            pred_scores.append(score_preds)
            gt_boxes.append(loc_unenc)
            gt_labels.append(cls_unenc)

    print(voc_eval(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
                   iou_thresh=0.5, use_07_metric=True))

    # Save checkpoint
    global best_loss
    if checkpoint is True:
        test_loss /= len(dataloader)
        if test_loss < best_loss:
            print('Saving..')
            state = {
                'net': net.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch_i,
            }
            if not os.path.isdir(os.path.dirname(args.checkpoint)):
                os.mkdir(os.path.dirname(args.checkpoint))
            torch.save(state, args.checkpoint)
            best_loss = test_loss


k_draw = 5
# batch_idx_draw_train = list(zip(
#         random.choices(range(len(train_evalloader)), k=k_draw),
#         random.choices(range(batch_size), k=k_draw)
# ))
random.seed(0)
batch_idx_draw_test = list(zip(
        random.choices(range(len(testloader)), k=k_draw),
        random.choices(range(testloader.batch_size), k=k_draw)
))
batch_idx_draw_coco = list(zip(
        random.choices(range(len(cocoloader)), k=k_draw),
        random.choices(range(cocoloader.batch_size), k=k_draw)
))

for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    run_root = '/scratch/run/pyssd/'
    # test('train', epoch, train_evalloader, batch_idx_draw_train,
    #      os.path.join(run_root, 'source-only', 'train'))
    test('test', epoch, testloader, vda_test_boxes_labels, batch_idx_draw_test,
         os.path.join(run_root, 'source-only', 'test'))
    
    test('coco', epoch, cocoloader, coco_test_boxes_labels, batch_idx_draw_coco,
         os.path.join(run_root, 'source-only', 'coco'))
