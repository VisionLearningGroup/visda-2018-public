import pickle
import random
from collections import namedtuple
from functools import partial

import os
import torch.utils.data
from torchvision import transforms as transforms

from torchcv.datasets import ListDataset
from torchcv.transforms import random_paste, random_crop, resize, random_flip

def transform_train(img, boxes, labels, ssd_box_coder, img_size):
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size, img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = ssd_box_coder.encode(boxes, labels)
    return img, boxes, labels


def transform_train_target(img, boxes, labels, img_size):
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size, img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    return img


def transform_test(img, boxes, labels, ssd_box_coder, img_size):
    img, boxes = resize(img, boxes, size=(img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    enc_boxes, enc_labels = ssd_box_coder.encode(boxes, labels)
    return img, enc_boxes, enc_labels


def box_label_list(root_dir, list_file, img_size):
    return list(ListDataset(
            root=root_dir,
            list_file=list_file,
            transform=lambda img, boxes, labels:
                (resize(img, boxes, size=(img_size, img_size))[1], labels)
    ))


# noinspection PyArgumentList
def prepare_data(ssd_box_coder, batch_size, n_workers, img_size, data_paths,
                 drop_last=False):
    trainset = ListDataset(
            root=data_paths['vda_root'], list_file=data_paths['vda_list_train'],
            transform=partial(
                    transform_train, ssd_box_coder=ssd_box_coder, img_size=img_size
            )
    )
    testset = ListDataset(
            root=data_paths['vda_root'], list_file=data_paths['vda_list_test'],
            transform=partial(
                    transform_test, ssd_box_coder=ssd_box_coder, img_size=img_size
            )
    )
    coco_trainset = ListDataset(
            root=data_paths['coco_root'], list_file=data_paths['coco_list_train'],
            transform=partial(transform_train_target, img_size=img_size)
    )
    coco_trainset_full = ListDataset(
            root=data_paths['coco_root'], list_file=data_paths['coco_list_train'],
            transform=partial(
                    transform_train, ssd_box_coder=ssd_box_coder, img_size=img_size
            )
    )
    coco_testset = ListDataset(
            root=data_paths['coco_root_test'], list_file=data_paths['coco_list_test'],
            transform=partial(
                    transform_test, ssd_box_coder=ssd_box_coder, img_size=img_size
            )
    )
    
    source_train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=drop_last
    )
    target_train_loader = torch.utils.data.DataLoader(
            coco_trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=drop_last
    )
    target_full_train_loader = torch.utils.data.DataLoader(
            coco_trainset_full, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=drop_last
    )
    cocoloader = torch.utils.data.DataLoader(
            coco_testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=drop_last
    )

    vda_test_boxes_labels, coco_test_boxes_labels = get_test_bboxes(
        data_paths['vda_list_test'], data_paths['vda_root'],
        data_paths['coco_list_test'], data_paths['coco_root_test'],
    )
    return namedtuple('Data', locals().keys())(**locals())


def get_test_bboxes(vda_list_test, vda_root_dir, coco_list_test, coco_root_dir,
                    persistence_dir='/home/grad2/usmn/tmp/'):
    img_size = 300

    boxes_vda_path = os.path.join(persistence_dir, 'boxes_vda.pkl')
    boxes_coco_path = os.path.join(persistence_dir, 'boxes_coco.pkl')
    args_file_path = os.path.join(persistence_dir, 'persist_args.pkl')

    argdump = {
        k: v for k, v in locals().items()
        if k in ['vda_list_test', 'vda_root', 'coco_list_test', 'coco_root']
    }

    if not (os.path.isfile(boxes_vda_path)
            and os.path.isfile(boxes_coco_path)
            and os.path.isfile(args_file_path)):
        print('not found boxes pkl, building .. ', os.path.isfile(boxes_vda_path),
              os.path.isfile(boxes_coco_path),
              os.path.isfile(args_file_path))

        coco_test_boxes_labels, vda_test_boxes_labels = build_save_boxes(
                args_file_path, boxes_coco_path, boxes_vda_path, coco_list_test,
                coco_root_dir, img_size, vda_list_test, vda_root_dir, argdump
        )

    else:
        with open(args_file_path, 'rb') as f:
            last_arg_dump = pickle.load(f)
        
        if last_arg_dump != argdump:
            print('argfile mismatch')
            coco_test_boxes_labels, vda_test_boxes_labels = build_save_boxes(
                    args_file_path, boxes_coco_path, boxes_vda_path, coco_list_test,
                    coco_root_dir, img_size, vda_list_test, vda_root_dir, argdump
            )
        else:
            print('files exist and argdumps match')
            with open(boxes_vda_path, 'rb') as f:
                vda_test_boxes_labels = pickle.load(f)

            with open(boxes_coco_path, 'rb') as f:
                coco_test_boxes_labels = pickle.load(f)

    return vda_test_boxes_labels, coco_test_boxes_labels


def build_save_boxes(args_file_path, boxes_coco_path, boxes_vda_path, coco_list_test,
                     coco_root_dir, img_size, vda_list_test, vda_root_dir, argdump):
    vda_test_boxes_labels = box_label_list(
            img_size=img_size,
            root_dir=vda_root_dir,
            list_file=vda_list_test,
    )
    coco_test_boxes_labels = box_label_list(
            img_size=img_size,
            root_dir=coco_root_dir,
            list_file=coco_list_test,
    )
    with open(boxes_vda_path, 'wb') as f:
        pickle.dump(vda_test_boxes_labels, f)
    with open(boxes_coco_path, 'wb') as f:
        pickle.dump(coco_test_boxes_labels, f)
    with open(args_file_path, 'wb') as f:
        pickle.dump(argdump, f)
    
    return coco_test_boxes_labels, vda_test_boxes_labels


if __name__ == '__main__':
    save_boxes_into_files()
