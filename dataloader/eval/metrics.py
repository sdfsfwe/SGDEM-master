# MIT License
#
# Copyright (c) 2020 Marvin Klingner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import warnings


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):#函数的目的是将相机坐标系下的点的坐标转换为世界坐标系下的坐标，并返回转换后的坐标列表
                                               #source_to_target_transformations是一个包含多个变换矩阵的列表
    xyzs = []
    cam_to_world = np.eye(4)                   #创建一个4x4的单位矩阵，表示相机到世界坐标系的初始变换矩阵。
    xyzs.append(cam_to_world[:3, 3])           #将相机坐标系下的原点（相机位置）的坐标（前三个元素）添加到xyzs列表中。
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation) #将当前的相机到世界坐标系的变换矩阵cam_to_world与当前的变换矩阵
        xyzs.append(cam_to_world[:3, 3])  #将更新后的cam_to_world矩阵的前三行第四列（表示世界坐标系下的点的坐标）添加到xyzs列表中。
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]    #计算两个轨迹中第一个匹配帧之间的偏移量
    pred_xyz = pred_xyz_o + offset[None, :]   #将预测轨迹的原始坐标列表与计算得到的偏移量相加，得到经过偏移对齐后的预测轨迹坐标列表

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)   #通过最小二乘法优化缩放因子。计算真实轨迹坐标列表与预测轨迹坐标列表的点积之和除以预测轨迹坐标列表的平方和，以得到一个缩放因子。
                                                                    # 这个缩放因子用于尺度对齐两个轨迹。
    alignment_error = pred_xyz * scale - gtruth_xyz                 #计算对齐后的预测轨迹坐标列表乘以缩放因子后与真实轨迹坐标列表之间的对齐误差。通过将预测轨迹乘以缩放因子并减去真实轨迹，可以计算每个点的对齐误差。
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0] #算均方根误差（Root Mean Square Error，RMSE）。
    return rmse


class Evaluator(object):
    # CONF MATRIX 混淆矩阵
    #     0  1  2  (PRED)
    #  0 |TP FN FN|
    #  1 |FP TP FN|
    #  2 |FP FP TP|
    # (GT)
    # -> rows (axis=1) are FN
    # -> columns (axis=0) are FP
    @staticmethod
    def iou(conf):  # TP / (TP + FN + FP)   #计算交并比（Intersection over Union，IoU）
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            iu = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf))
        meaniu = np.nanmean(iu)
        result = {'iou': dict(zip(range(len(iu)), iu)), 'meaniou': meaniu}
        return result

    @staticmethod
    def accuracy(conf):  # TP / (TP + FN) aka 'Recall'   #计算准确率（Accuracy）
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            totalacc = np.diag(conf).sum() / (conf.sum())
            acc = np.diag(conf) / (conf.sum(axis=1))
        meanacc = np.nanmean(acc)
        result = {'totalacc': totalacc, 'meanacc': meanacc, 'acc': acc}
        return result

    @staticmethod
    def precision(conf):  # TP / (TP + FP)  计算精确率（Precision）
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            prec = np.diag(conf) / (conf.sum(axis=0))
        meanprec = np.nanmean(prec)
        result = {'meanprec': meanprec, 'prec': prec}
        return result

    @staticmethod
    def freqwacc(conf):            #计算加权频权准确率
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            iu = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf))
            freq = conf.sum(axis=1) / (conf.sum())
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        result = {'freqwacc': fwavacc}
        return result

    @staticmethod
    def depththresh(gt, pred):        #计算深度阈值
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        result = {'delta1': a1, 'delta2': a2, 'delta3': a3}
        return result

    @staticmethod
    def deptherror(gt, pred):         #计算深度误差
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        result = {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log}
        return result


class SegmentationRunningScore(object):  #用于评估分割模型的性能
    def __init__(self, n_classes=20): #默认情况下，假设有20个类别
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):    #_fast_hist方法计算真实标签和预测标签的直方图，label_true（真实标签），label_pred（预测标签）和n_class（类别数）
        mask_true = (label_true >= 0) & (label_true < n_class)
        mask_pred = (label_pred >= 0) & (label_pred < n_class)
        mask = mask_pred & mask_true
        label_true = label_true[mask].astype(np.int)
        label_pred = label_pred[mask].astype(np.int)
        hist = np.bincount(n_class * label_true + label_pred,   #函数np.bincount用于计算非负整数数组中每个整数出现的次数。它的作用是统计数组中每个整数值的频数。
                           minlength=n_class*n_class).reshape(n_class, n_class).astype(np.float)
        return hist

    def update(self, label_trues, label_preds):
        # label_preds = label_preds.exp()
        # label_preds = label_preds.argmax(1).cpu().numpy() # filter out the best projected class for each pixel
        # label_trues = label_trues.numpy() # convert to numpy array

        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes) #update confusion matrix
            #flatten()打平操作，将多维数组或矩阵转换为一维数组

    def get_scores(self, listofparams=None):
        """Returns the evaluation params specified in the list"""
        possibleparams = {
            'iou': Evaluator.iou,
            'acc': Evaluator.accuracy,
            'freqwacc': Evaluator.freqwacc,
            'prec': Evaluator.precision
        }
        if listofparams is None:
            listofparams = possibleparams

        result = {}
        for param in listofparams:
            if param in possibleparams.keys():
                result.update(possibleparams[param](self.confusion_matrix))
        return result

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class DepthRunningScore(object):
    def __init__(self):
        self.num_samples = 0
        self.depth_thresh = {'delta1': 0, 'delta2': 0, 'delta3': 0}       #三个深度阈值评估指标
        self.depth_errors = {'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0}

    def update(self, ground_truth, prediction):
        if isinstance(ground_truth, list):   #isinstance()用于检查一个对象是否是指定类型或指定类型的子类的实例
            self.num_samples += len(ground_truth)
        else:
            ground_truth = [ground_truth]
            prediction = [prediction]
            self.num_samples += 1

        for k in range(len(ground_truth)):
            gt = ground_truth[k].astype(np.float)   #astype()用于将数组的数据类型转换为指定的数据类型
            pred = prediction[k].astype(np.float)
            thresh = Evaluator.depththresh(gt, pred)
            error = Evaluator.deptherror(gt, pred)
            for i, j in zip(thresh.keys(), self.depth_thresh.keys()):
                self.depth_thresh[i] += thresh[j]
            for i, j in zip(error.keys(), self.depth_errors.keys()):
                self.depth_errors[i] += error[j]

    def get_scores(self, listofparams=None):   #用于获取评估指标的结果
        """Returns the evaluation params specified in the list"""
        possibleparams = {
            'thresh': self.depth_thresh,
            'error': self.depth_errors,
        }
        if listofparams is None:
            listofparams = possibleparams

        result = {}
        for param in listofparams:
            if param in possibleparams.keys():
                result.update(possibleparams[param])
        for i in result.keys():
            result[i] = result[i]/self.num_samples

        return result

    def reset(self):
        self.num_samples = 0
        self.depth_thresh = {'delta1': 0, 'delta2': 0, 'delta3': 0}
        self.depth_errors = {'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0}


class PoseRunningScore(object):
    def __init__(self):
        self.preds = list()
        self.gts = list()

    def update(self, ground_truth, prediction):
        if isinstance(ground_truth, list):
            self.gts += ground_truth
        else:
            self.gts += [ground_truth]

        if isinstance(prediction, list):
            self.preds += prediction
        else:
            self.preds += [prediction]

    def get_scores(self):
        """Returns the evaluation params specified in the list"""

        gt_global_poses = np.concatenate(self.gts)
        pred_poses = np.concatenate(self.preds)

        gt_global_poses = np.concatenate(
            (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
        gt_global_poses[:, 3, 3] = 1    #真实姿态数组中的最后一个位姿矩阵的最后一行补充为[0, 0, 0, 1]，以确保矩阵是齐次坐标形式。
        gt_xyzs = gt_global_poses[:, :3, 3]
        gt_local_poses = []
        for i in range(1, len(gt_global_poses)):
            gt_local_poses.append(
                np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))
        ates = []
        num_frames = gt_xyzs.shape[0]
        track_length = 5
        for i in range(0, num_frames - track_length + 1):
            local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
            gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
            ates.append(compute_ate(gt_local_xyzs, local_xyzs))

        pose_error = {'mean': np.mean(ates), 'std': np.std(ates)}
        return pose_error

    def reset(self):
        self.preds = list()
        self.gts = list()


class AverageMeter(object):    #求平均值
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
