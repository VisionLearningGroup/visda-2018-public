import sys
import pickle
import numpy as np


def load_datalist(path):
    # Load COCO GT
    coco_boxes = []
    coco_labels = []
    for line in open(path, 'r').readlines():
        parts = [p.strip() for p in line.split()]
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
        coco_boxes.append(np.array(boxes))
        coco_labels.append(np.array(labs))

    return coco_boxes, coco_labels


def convert(pickle_path, datalist_path):
    boxes, labels = load_datalist(datalist_path)

    gt = boxes, labels

    with open(pickle_path, 'wb') as f_out:
        pickle.dump(gt, f_out)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:')
        print('{} <out_pickle_path> <in_datalist_path>')
    else:
        pickle_path = sys.argv[1]
        datalist_path = sys.argv[2]
        convert(pickle_path, datalist_path)
