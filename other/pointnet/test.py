"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

# >>> update: new dataloader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import trimesh
import json
from sklearn.metrics import accuracy_score
import core
from pathlib import Path
from pointnet.data_utils.ShapeNetDataLoader import PartNormalDataset
import matplotlib
matplotlib.use('Agg')
# >>> end update

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

# seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
# for cat in seg_classes.keys():
#     for label in seg_classes[cat]:
#         seg_label_to_cat[label] = cat


def make_pointcloud(points, seg):
    colors = np.expand_dims(seg.astype(np.uint8)*255, 1).repeat(3, axis=1)
    assert points.shape[0] == 1 or len(points.shape) == 2
    return trimesh.PointCloud(
        points[0, :, :3], 
        colors=colors,
    )


def export_pointcloud(f_out, pointcloud):
    assert os.path.splitext(f_out)[-1] == ".obj"
    pointcloud.export(f_out, include_normals=True, include_color=True)


def plot_loss(logfile):
    loss_acc = []
    acc_acc = []
    print(logfile)
    with open(logfile, "r") as f:
        for line in f.readlines():
            if "Train loss is:" in line:
                loss_acc.append(float(line.split(" ")[-1]))
            if "Train accuracy is:" in line:
                acc_acc.append(float(line.split(" ")[-1]))
    loss_acc = np.array(loss_acc)
    acc_acc = np.array(acc_acc)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(loss_acc)
    ax[0].set_title("Train loss over time")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")

    ax[1].plot(acc_acc)
    ax[1].set_title("Accuracy over time")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    return fig

def plot_accuracy(
    frac_acc,
    no_frac_acc,
    frac_means, 
    frac_std,
    no_frac_means, 
    no_frac_std,
    labels,
    class_labels,
):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    def autolabel(rects, axes):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            axes.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '{:.2f}'.format(float(height)),
                    ha='center', va='bottom')

    ax[0].set_title("Fracture Mean Accuracy by Class: {:.4f}".format(frac_acc.mean()))
    x_pos = range(labels.max() + 1)
    bar = ax[0].bar(x_pos, frac_means, yerr=frac_std)
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(class_labels)
    autolabel(bar, ax[0])

    ax[1].set_title("Non-Fracture Mean Accuracy by Class: {:.4f}".format(no_frac_acc.mean()))
    x_pos = range(labels.max() + 1)
    bar = ax[1].bar(x_pos, no_frac_means, yerr=no_frac_std)
    ax[1].set_xticks(x_pos)
    ax[1].set_xticklabels(class_labels)
    autolabel(bar, ax[1])

    for cls in range(labels.max() + 1):
        ax[2].scatter(
            frac_acc[labels == cls],
            no_frac_acc[labels == cls],
        )
    ax[2].set_title("Fracture VS Non-Fracture accuracy")
    ax[2].set_xlabel("Fracture accuracy")
    ax[2].set_ylabel("Non-Fracture accuracy")
    ax[2].legend(class_labels)

    fig.tight_layout()
    return fig


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """ 
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--test_pkl',  type=str, default=None) 
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--model', type=str, help='network name')
    return parser.parse_args()


def vote(classifier, points, label, num_classes, num_part, num_votes=3):
    assert points.size()[0] == 1, "Doesn't support batching"

    vote_pool = torch.zeros(1, points.size()[1], num_part).cuda()
    points = points.transpose(2, 1)

    for _ in range(num_votes):
        seg_pred, _ = classifier(points, to_categorical(label, num_classes))
        vote_pool += seg_pred

    seg_pred = vote_pool / num_votes

    return np.argmax(seg_pred.squeeze(0).cpu().data.numpy(), axis=1) 

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # >>> update: added global model saving
    args.log_dir = args.log_dir.replace("$DATADIR", os.environ["DATADIR"])
    experiment_dir = args.log_dir
    # >>> end update
    # >>> update: added visual dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)
    # >>> end update

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    TEST_DATASET = PartNormalDataset(root=None, npoints=args.num_point, split='test', normal_channel=args.normal, data_file=args.test_pkl)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    # >>> update: classes and parts are determined by dataloader
    # num_classes = 16
    # num_part = 50
    num_classes = TEST_DATASET.num_classes
    num_part = TEST_DATASET.num_parts
    # >>> end update

    acc_save_path = experiment_dir + "/stats.npz"
    print("saving to ", acc_save_path)
    if not os.path.exists(acc_save_path):

        '''MODEL LOADING'''
        model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
        MODEL = importlib.import_module(model_name)
        # >>> update: added num_classes as a passable parameter
        classifier = MODEL.get_model(part_num=num_part, class_num=num_classes, normal_channel=args.normal).cuda()
        # >>> end update
        # >>> update: added dataparallel
        classifier = torch.nn.DataParallel(classifier)
        # >>> end update
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/' + args.model)
        classifier.load_state_dict(checkpoint['model_state_dict'])

        # >>> update: seg_classes is now an attribute of the dataloader
        seg_classes = TEST_DATASET.seg_classes
        # print(TEST_DATASET.classes)
        # >>> end update

        with torch.no_grad():
            
            classifier = classifier.eval()
            frac_acc_accumulator = []
            no_frac_acc_accumulator = []

            pred_accumulator = []
            gt_accumulator = []

            label_acc = []

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader)):
                class_id, instance_id, break_num = TEST_DATASET.data[batch_id]

                seg = vote(classifier, points, label, num_classes, num_part, num_votes=args.num_votes)
                target = target.numpy()[0, ...]

                pred_accumulator.append(seg)
                gt_accumulator.append(target)

                frac_acc_accumulator.append(
                    accuracy_score(target[target == 1], seg[target == 1])
                )
                no_frac_acc_accumulator.append(
                    accuracy_score(target[target == 0], seg[target == 0])
                )
                label_acc.append(
                    label[0][0].numpy()
                )

            frac_acc_accumulator = np.array(frac_acc_accumulator)
            no_frac_acc_accumulator = np.array(no_frac_acc_accumulator)
            pred_accumulator = np.array(pred_accumulator)
            gt_accumulator = np.array(gt_accumulator)
            label_acc = np.array(label_acc)

            print(pred_accumulator.shape)
            print(frac_acc_accumulator.shape)
            print(comps_accumulator)

            print("Mean fracture accuracy: {}".format(frac_acc_accumulator.mean()))
            print("Mean non-fracture accuracy: {}".format(no_frac_acc_accumulator.mean()))

            np.savez(
                acc_save_path, 
                frac_acc=frac_acc_accumulator, 
                no_frac_acc=no_frac_acc_accumulator, 
                labels=label_acc,
                pred_accumulator=pred_accumulator,
                gt_accumulator=gt_accumulator,
            )

    data = np.load(acc_save_path)
    frac_acc = data["frac_acc"]
    no_frac_acc = data["no_frac_acc"]
    labels = data["labels"]

    class_list = json.load(open("shapenet_classes.json", "r"))
    def get_classname(idx):
        for k, v in TEST_DATASET.classes.items():
            if v == idx:
                return class_list[k]

    frac_means = []
    frac_std = []
    no_frac_means = []
    no_frac_std = []
    class_labels = []
    for cls in range(labels.max() + 1):
        frac_means.append(frac_acc[labels == cls].mean())
        frac_std.append(frac_acc[labels == cls].std())
        no_frac_means.append(no_frac_acc[labels == cls].mean())
        no_frac_std.append(no_frac_acc[labels == cls].std())
        class_labels.append(get_classname(cls))

    plot_accuracy(
        frac_acc,
        no_frac_acc,
        frac_means, 
        frac_std,
        no_frac_means, 
        no_frac_std,
        labels,
        class_labels,
    ).savefig("plot_test_accuracy.jpg")

    plot_loss(
        experiment_dir + "/logs" + "/pointnet_part_seg.txt"
    ).savefig("plot_train_loss_accuracy.jpg")
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
