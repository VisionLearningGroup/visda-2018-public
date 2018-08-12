import sys
import pickle
import numpy as  np


def load_datalist(f):
    """
    Load ground truths in datalist format
    :param f: file to load
    :return: `(names, boxes, labels)`:
        `names` is a list of filenames
        `boxes` is a list of (N,4) NumPy arrays,
        `labels` is a list of (N,) NumPy arrays
    """
    # Load COCO GT
    per_image_names = []
    per_image_boxes = []
    per_image_labels = []
    for line in f.readlines():
        parts = [p.strip() for p in line.split()]
        per_image_names.append(parts[0])
        parts = parts[1:]
        boxes = []
        labs = []
        for i in range(0, len(parts), 5):
            x0, y0, x1, y1, cls = parts[i:i+5]
            x0 = float(x0)
            y0 = float(y0)
            x1 = float(x1)
            y1 = float(y1)
            cls = int(cls)
            boxes.append([y0, x0, y1, x1])
            labs.append(cls)
        per_image_boxes.append(np.array(boxes))
        per_image_labels.append(np.array(labs))

    return per_image_names, per_image_boxes, per_image_labels


def compare(coco_gt_path, pkl_gt_path):
    """
    Compare COCO ground truths with pickled cached ground truths

    :param coco_gt_path: path to CoCo file (e.g. coco17-val.txt)
    :param pkl_gt_path: path to pickled cached GT (e.g. val_ground_truth.pkl)
    :return: per-image scale factor as a (N,2) NumPy array
    """
    with open(coco_gt_path, 'r') as f_coco:
        coco_names, coco_boxes, coco_labels = load_datalist(f_coco)

    with open(pkl_gt_path, 'rb') as f_in:
        pkl_boxes, pkl_labels = pickle.load(f_in)

    scale_factors = []

    for cn, cb, cl, pb, pl in zip(coco_names, coco_boxes, coco_labels, pkl_boxes, pkl_labels):
        box_scale = pb / (cb+1.0e-12)
        box_scale = box_scale.reshape((-1, 2))
        box_scale = np.median(box_scale, axis=0)
        print('{}:  median scale factor ([y,x]): {}'.format(cn, box_scale))
        scale_factors.append(box_scale)

    scale_factors = np.stack(scale_factors, axis=0)

    return scale_factors


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:')
        print('   {} <path to coco17-val.txt> <path to val_ground_truth.pkl>')
    else:
        coco_path = sys.argv[1]
        pkl_gt_path = sys.argv[2]

        compare(coco_path, pkl_gt_path)
