"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm

# >>> update: new dataloader
import subprocess
from pointnet.data_utils.ShapeNetDataLoader import PartNormalDataset
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


def network_backup(experiment_directory, ssh_address):
    dir = os.path.abspath(experiment_directory)
    logging.info("Backing up to server: {}".format(["rsync", "-r", dir, ssh_address]))
    process = subprocess.Popen(["rsync", "-r", dir, ssh_address])
    process.wait()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default=None, help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay') 
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--frac_weight', type=float, default=0.9, help='loss weight assigned to fracture region')
    parser.add_argument('--train_pkl',  type=str, default=None) 
    parser.add_argument('--test_pkl',  type=str, default=None) 
    parser.add_argument('--backup',  type=str, default=None)

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # update: let gpu be passed as none
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # end update

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    # update: print logging location
    print("Logging to: {}".format(log_dir))
    # end update

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # >>> update: changed where data lives
    assert 1.0 > args.frac_weight > 0.0
    if args.backup:
        args.backup = args.backup.replace("$DATADIR", os.environ["DATADIR"])
    log_string('backing network up to: {}'.format(args.backup))
    network_backup(exp_dir, args.backup)
    # >>> end update

    # >>> update: no validation set
    TRAIN_DATASET = PartNormalDataset(npoints=args.npoint, split='train', normal_channel=args.normal, data_file=args.train_pkl)
    # >>> end update
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(npoints=args.npoint, split='test', normal_channel=args.normal, data_file=args.test_pkl)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    # import trimesh
    # for p, _, seg in TRAIN_DATASET:
    #     c = seg.astype(np.uint8)*255
    #     c = np.expand_dims(c, axis=1)
    #     c = np.hstack((c, c, c))
    #     trimesh.PointCloud(p[:, :3], colors=c).export("test.obj", include_normals=True, include_color=True)
    #     break
    # exit()

    # acc = []
    # for _, _, seg in TRAIN_DATASET:
        # acc.append(seg.sum() / len(seg))
    # acc = np.array(acc)
    # print(acc)
    # print(acc.mean())
    # exit()

    # >>> update: classes and parts are determined by dataloader
    num_classes = TRAIN_DATASET.num_classes
    num_part = TRAIN_DATASET.num_parts
    # >>> end update

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    # >>> update: added num_classes as a passable parameter
    classifier = MODEL.get_model(part_num=num_part, class_num=num_classes, normal_channel=args.normal).cuda()
    # >>> end update
    # >>> update: added dataparallel
    classifier = torch.nn.DataParallel(classifier)
    # >>> end update
    # >>> update: added weight passing
    criterion = MODEL.get_loss(weight=torch.Tensor([1.0-args.frac_weight, args.frac_weight]).cuda()).cuda()
    # >>> end update
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    TEST_EVERY = 50
    SAVE_EVERY = 50

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    # update: loss logging
    train_loss = {}
    # end update

    # update: why is tqdm there
    for epoch in tqdm(range(start_epoch, args.epoch), smoothing=0.9):
    # end update

        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        train_loss[epoch] = []
        for i, (points, label, target) in enumerate(trainDataLoader):
            optimizer.zero_grad()

            points = points.data.numpy()
            # update: removed random scale and sift
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # end update
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            # update: loss logging
            train_loss[epoch].append(loss.item())
            # end update
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        # update: loss logging
        log_string('Train loss is: %.5f' % train_loss[epoch][-1])
        # end update

        # >>> update: seg_classes is now an attribute of the dataloader
        seg_classes = TEST_DATASET.seg_classes
        # >>> end update

        # update: turned off testing on every iteration
        if epoch % TEST_EVERY == 0:
        # end update

            with torch.no_grad():
                test_metrics = {}
                total_correct = 0
                total_seen = 0
                total_seen_class = [0 for _ in range(num_part)]
                total_correct_class = [0 for _ in range(num_part)]
                shape_ious = {cat: [] for cat in seg_classes.keys()}
                seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

                for cat in seg_classes.keys():
                    for label in seg_classes[cat]:
                        seg_label_to_cat[label] = cat

                classifier = classifier.eval()

                for batch_id, (points, label, target) in enumerate(testDataLoader):
                    cur_batch_size, NUM_POINT, _ = points.size()
                    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)
                    seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    correct = np.sum(cur_pred_val == target)
                    total_correct += correct
                    total_seen += (cur_batch_size * NUM_POINT)

                    for l in range(num_part):
                        total_seen_class[l] += np.sum(target == l)
                        total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                    for i in range(cur_batch_size):
                        segp = cur_pred_val[i, :]
                        segl = target[i, :]
                        cat = seg_label_to_cat[segl[0]]
                        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                        for l in seg_classes[cat]:
                            if (np.sum(segl == l) == 0) and (
                                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                                part_ious[l - seg_classes[cat][0]] = 1.0
                            else:
                                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                    np.sum((segl == l) | (segp == l)))
                        shape_ious[cat].append(np.mean(part_ious))

                all_shape_ious = []
                for cat in shape_ious.keys():
                    for iou in shape_ious[cat]:
                        all_shape_ious.append(iou)
                    shape_ious[cat] = np.mean(shape_ious[cat])
                test_metrics['accuracy'] = total_correct / float(total_seen)
                test_metrics['class_avg_accuracy'] = np.mean(
                    np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
                for cat in sorted(shape_ious.keys()):
                    log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
                test_metrics['class_avg_iou'] = np.mean(list(shape_ious.values()))
                test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
                # update: added nanmean
                test_metrics['class_nanavg_iou'] = np.nanmean(list(shape_ious.values()))
                test_metrics['inctance_nanavg_iou'] = np.nanmean(all_shape_ious)
                # end update

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        # if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
        if epoch % SAVE_EVERY == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model_{}.pth'.format(epoch)
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            log_string('Saving model....')
            torch.save(state, savepath)
            log_string('backing network up to: {}'.format(args.backup))
            network_backup(exp_dir, args.backup)

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1
    
    logger.info('Save model...')
    savepath = str(checkpoints_dir) + '/model_{}.pth'.format(epoch)
    log_string('Saving at %s' % savepath)
    state = {
        'epoch': epoch,
        'train_acc': train_instance_acc,
        'test_acc': test_metrics['accuracy'],
        'class_avg_iou': test_metrics['class_avg_iou'],
        'inctance_avg_iou': test_metrics['inctance_avg_iou'],
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    log_string('Saving model....')
    torch.save(state, savepath)
    log_string('backing network up to: {}'.format(args.backup))
    network_backup(exp_dir, args.backup)


if __name__ == '__main__':
    args = parse_args()
    main(args)
