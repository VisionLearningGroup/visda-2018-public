from __future__ import print_function

import datetime
import os
import random
import argparse
from itertools import islice

from PIL import Image, ImageDraw

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchcv.evaluations import voc_eval
from torchcv.models.ssd import SSD300, SSDBoxCoder
from torchcv.loss import SSDLoss
from torchcv.utils.tee_stdout import tee_stdout
from visda import prepare_data

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
net_student = SSD300(num_classes=num_classes)
net_student.load_state_dict(torch.load(args.model))

net_teacher = SSD300(num_classes=num_classes)
net_teacher.load_state_dict(net_student.state_dict())

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net_student.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# Dataset
print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net_student)
batch_size = 16
visda_data = prepare_data(
        box_coder, batch_size=batch_size, n_workers=0,
        img_size=net_student.steps[-1],
        # data_paths={
        #     'vda_root': '/scratch2/ben/png_json',
        #     'vda_list_train': '/scratch2/ben/visda18-detection-train.txt',
        #     'vda_list_test': '/scratch2/ben/visda18-detection-test.txt',
        #     'coco_root': '/scratch2/ben/coco',
        #     'coco_list_train': '/scratch2/ben/coco-train-8000.txt',
        #     'coco_list_test': '/scratch2/ben/coco-train-2240.txt'
        # }
        data_paths={
            'vda_root': '/scratch2/mytmp/render_detection_result//png_json',
            'vda_list_train': '/scratch2/mytmp/render_detection_result/listdataset/visda18-detection-train.txt',
            'vda_list_test': '/scratch2/mytmp/render_detection_result/listdataset/visda18-detection-test.txt',
            'coco_root': '/scratch2/data/coco/train2014',
            'coco_list_train': '/scratch2/mytmp/render_detection_result/listdataset/coco-train-8000.txt',
            'coco_list_test': '/scratch2/mytmp/render_detection_result/listdataset/coco-train-2240.txt'
        }
)

cat_of_interest = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
                   'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


unnormalize = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class ExponentialAverageOf(object):
    def __init__(self, target, source, alpha=0.999):
        self.params = list(target)
        self.src_params = list(source)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)


print('==> Distributing model..')

n_gpus = os.environ['CUDA_VISIBLE_DEVICES'].count(',')+1
if n_gpus > 1:
    gpu_range = range(n_gpus)
    net_teacher = torch.nn.DataParallel(net_teacher.extractor, device_ids=gpu_range).cuda()
    net_student = torch.nn.DataParallel(net_student, device_ids=gpu_range).cuda()
else:
    net_teacher = net_teacher.cuda()
    net_student = net_student.cuda()
    
for param in net_teacher.parameters():
    param.requires_grad = False

print('==> Building loss and updaters..')

criterion = SSDLoss(num_classes=num_classes)
optimizer = optim.SGD(
        net_student.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
)
exp_avg_setter = ExponentialAverageOf(
        net_teacher.parameters(), net_student.parameters(),
        alpha=1-1e-3
)


def draw_boxes(input_, loc, cls, path, name):
    boxes, labels, scores = box_coder.decode(loc, cls)
    img: Image = transforms.ToPILImage()(unnormalize(input_))
    draw = ImageDraw.Draw(img)
    if boxes is not None:
        for box, label in zip(boxes, labels):
            draw.rectangle(list(box), outline='red')
            draw.text((box[0], box[1]), cat_of_interest[label], fill='red')
    os.makedirs(path, exist_ok=True)
    img.save(os.path.join(path, name+'.png'))


batch_n_max_count = None


def bce(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def train(epoch):
    lambda_1 = 1
    lambda_2 = 100
    
    print('\nEpoch: %d' % epoch)
    net_student.train()
    train_loss = 0
    train_slice = islice(
            enumerate(zip(visda_data.source_train_loader,
                          visda_data.target_train_loader)),
            batch_n_max_count
    )
    for batch_idx, joint_data_batch in train_slice:
        (source_inputs, loc_targets, cls_targets), target_inputs = joint_data_batch
        source_inputs = Variable(source_inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        target_inputs = Variable(target_inputs.cuda())
        
        # loc_targets_avg = loc_targets.mean()
        cls_target_scat = torch.zeros(
                cls_targets.size(0), cls_targets.size(1), num_classes+1
        ).cuda()
        cls_target_scat.scatter_(2, cls_targets.data.unsqueeze(2).long(), 1)
        avg_cls_target = Variable(cls_target_scat.mean(0).mean(0), requires_grad=False)
        
        optimizer.zero_grad()
        loc_preds, cls_preds = net_student(source_inputs)
        detection_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)

        loc_preds_teacher_s, cls_preds_teacher_s = net_student(target_inputs)
        loc_preds_teacher_t, cls_preds_teacher_t = net_teacher(target_inputs)

        # [batch, pos]
        confidence_mask = (
                torch.sum(F.softmax(cls_preds_teacher_s, 2) > 0.95,
                          dim=2, keepdim=True) > 0
        ).float()
        loc_diff = torch.abs(loc_preds_teacher_s - loc_preds_teacher_t)*confidence_mask
        cls_diff = torch.abs(cls_preds_teacher_s - cls_preds_teacher_t)*confidence_mask
        st_loss = torch.pow(loc_diff, 1).mean() + torch.pow(cls_diff, 1).mean()

        cls_preds_teacher_s_w_bg = torch.cat(
                [Variable(torch.zeros(cls_preds_teacher_s.size(0),
                                      cls_preds_teacher_s.size(1), 1).cuda()),
                 cls_preds_teacher_s], 2
        )
        
        # stat_dist = lambda x, y: torch.mean(torch.pow(x-y, 2))
        stat_dist = lambda x, y: torch.mean(bce(x, y))

        cl_bal_loss = stat_dist(F.softmax(cls_preds_teacher_s_w_bg, 2).mean(1),
                                avg_cls_target.unsqueeze(0))
        
        loss = detection_loss + lambda_1 * st_loss + lambda_2 * cl_bal_loss
        loss.backward()
        optimizer.step()
        exp_avg_setter.step()
        train_loss += loss.data[0]
        
        print(f'st_loss: {st_loss.data[0]}', end=' | ')
        print(f'cl_bl_loss: {cl_bal_loss.data[0]}', end=' | ')
        print(f'train_loss: {detection_loss.data[0]:.3f} '
              f'full: {loss.data[0]:.3f}'
              f'| avg_loss: {train_loss / (batch_idx + 1):.3f} '
              f'[{batch_idx + 1}/{len(visda_data.source_train_loader)}]')


def test(name, network, epoch_i, dataloader, test_boxes_labels,
         samples_to_draw, image_folder, checkpoint=False):
    print('\nTest')
    network.eval()
    test_loss = 0
    pred_boxes, pred_labels, pred_scores = [], [], []
    gt_boxes, gt_labels = [], []
    for batch_idx, data_point in islice(enumerate(dataloader), batch_n_max_count):
        inputs, loc_targets, cls_targets = data_point
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        print(f'{name}: ', end='')
        loc_preds, cls_preds = network(inputs)
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
                'net': network.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch_i,
            }
            if not os.path.isdir(os.path.dirname(args.checkpoint)):
                os.mkdir(os.path.dirname(args.checkpoint))
            torch.save(state, args.checkpoint)
            best_loss = test_loss


k_draw = 5

random.seed(0)
batch_idx_draw_test = list(zip(
        random.choices(range(len(visda_data.testloader)), k=k_draw),
        random.choices(range(visda_data.testloader.batch_size), k=k_draw)
))
batch_idx_draw_coco = list(zip(
        random.choices(range(len(visda_data.cocoloader)), k=k_draw),
        random.choices(range(visda_data.cocoloader.batch_size), k=k_draw)
))


def start_train_test(run_name, run_root):
    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test('test', net_student, epoch,
             visda_data.testloader, visda_data.vda_test_boxes_labels,
             batch_idx_draw_test, os.path.join(run_root, run_name, 'test'))
        
        test('coco', net_student, epoch,
             visda_data.cocoloader, visda_data.coco_test_boxes_labels,
             batch_idx_draw_coco, os.path.join(run_root, run_name, 'coco'))


def main():
    run_root = '/scratch2/run/pyssd'
    run_name = 'basic-st'
    time_str = datetime.datetime.now().__format__('%d-%b-%y-%H:%M:%S')
    log_fn = os.path.join(
            '/home/grad2/usmn/projects/tfdetect/torchcv/logs',
            run_name+time_str + '.conf'
    )

    with tee_stdout(log_fn):
        start_train_test(run_name, run_root)


main()
