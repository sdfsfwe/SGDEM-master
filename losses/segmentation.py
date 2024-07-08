import torch
import torch.nn.functional as functional
import numpy as np

from dataloader.eval.metrics import SegmentationRunningScore

SEG_CLASS_WEIGHTS = (
    2.8149201869965, 6.9850029945374, 3.7890393733978, 9.9428062438965,
    9.7702074050903, 9.5110931396484, 10.311357498169, 10.026463508606,
    4.6323022842407, 9.5608062744141, 7.8698215484619, 9.5168733596802,
    10.373730659485, 6.6616044044495, 10.260489463806, 10.287888526917,
    10.289801597595, 10.405355453491, 10.138095855713, 0
)

CLS_ROAD = 0
CLS_SIDEWALK = 1 
CLS_BUILDING = 2 
CLS_WALL = 3
CLS_FENCE = 4 #围栏
CLS_POLE = 5 #柱子
CLS_TRLIGHT = 6 #灯
CLS_TRSIGN = 7 #标签
CLS_VEGT = 8 #
CLS_TERR = 9 #书
CLS_SKY = 10 #蓝天
CLS_PERSON = 11 #人
CLS_RIDER = 12 #骑手
CLS_CAR = 13 #车
CLS_TRUCK = 14 #卡车
CLS_BUS = 15 #汽车
CLS_TRAIN = 16 #火车
CLS_MCYCLE = 17 #
CLS_BCYCLE = 18 #单车

class SegLosses(object):
    def __init__(self, device):
        self.weights = torch.tensor(SEG_CLASS_WEIGHTS, device=device)
        self.labels_to_replace = [0, 1, 2, 4, 10,19,255]

    def dice_loss(self, input, target):
        smooth = 1.0
        input_flat = input.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (input_flat * target_flat).sum()
        return 1 - ((2.0 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth))

    def seg_losses(self, inputs, outputs):
        preds = outputs['segmentation_logits', 0]
        targets = inputs['segmentation', 0, 0][:, 0, :, :].long()

        losses = dict()
        losses["loss_seg"] = functional.cross_entropy(
            preds, targets, self.weights, ignore_index=255
        )
    
        #print('a:',functional.softmax(preds, dim=1)[:, 1, :, :])

        '''
        segs_pred=preds
        segs_pred = segs_pred.exp().cpu()
        segs_pred = segs_pred.detach().numpy()
        segs_pred = segs_pred.argmax(1)
        mask = np.isin(segs_pred, self.labels_to_replace)
        segs_pred[mask] = 0
        segs_pred = (segs_pred != 0).astype(np.int)
        segs_pred_tensor = torch.tensor(segs_pred,device='cuda')

        targets=targets.cpu().detach().numpy()
        mask = np.isin(targets, self.labels_to_replace)
        targets[mask] = 0
        targets = (targets != 0).astype(np.int)
        targets = torch.tensor(targets,device='cuda')

        dice_loss = self.dice_loss( segs_pred_tensor, targets )
        losses["dice_loss"] = dice_loss

         # Combine losses
        #alpha = 0.8  # You can adjust this balance factor
        #beta = 0.2   # You can adjust this balance factor
        #combined_loss = alpha * losses["loss_seg_con"] + beta * losses["dice_loss"]
        #print('losses["loss_seg_con"]',losses["loss_seg_con"])
        #print('losses["dice_loss"]',losses["dice_loss"])
        #losses["loss_seg"] = combined_loss'''

        return losses


class RemappingScore(object):  # todo: delete this std->None. Soll direkt auf dem segmentationrunning score lauven
    REMAPS = {
        'none': (
            CLS_ROAD, CLS_SIDEWALK, CLS_BUILDING, CLS_WALL,
            CLS_FENCE, CLS_POLE, CLS_TRLIGHT, CLS_TRSIGN,
            CLS_VEGT, CLS_TERR, CLS_SKY, CLS_PERSON,
            CLS_RIDER, CLS_CAR, CLS_TRUCK, CLS_BUS,
            CLS_TRAIN, CLS_MCYCLE, CLS_BCYCLE
        ),
        'dada_16': (
            CLS_ROAD, CLS_SIDEWALK, CLS_BUILDING, CLS_WALL,
            CLS_FENCE, CLS_POLE, CLS_TRLIGHT, CLS_TRSIGN,
            CLS_VEGT, CLS_SKY, CLS_PERSON,
            CLS_RIDER, CLS_CAR, CLS_BUS,
            CLS_MCYCLE, CLS_BCYCLE
        ),
        'dada_13': (
            CLS_ROAD, CLS_SIDEWALK, CLS_BUILDING,
            CLS_TRLIGHT, CLS_TRSIGN,
            CLS_VEGT, CLS_SKY, CLS_PERSON,
            CLS_RIDER, CLS_CAR, CLS_BUS,
            CLS_MCYCLE, CLS_BCYCLE
        ),
        'dada_7': (
            (CLS_ROAD, CLS_SIDEWALK),
            (CLS_BUILDING, CLS_WALL, CLS_FENCE),
            (CLS_POLE, CLS_TRLIGHT, CLS_TRSIGN),
            (CLS_VEGT, CLS_TERR),
            CLS_SKY,
            (CLS_PERSON, CLS_RIDER),
            (CLS_CAR, CLS_TRUCK, CLS_BUS, CLS_TRAIN, CLS_MCYCLE, CLS_BCYCLE)
        ),
        'gio_10': (
            CLS_ROAD, CLS_BUILDING,
            CLS_POLE, CLS_TRLIGHT, CLS_TRSIGN,
            CLS_VEGT, CLS_TERR, CLS_SKY,
            CLS_CAR, CLS_TRUCK
        )
    }

    def __init__(self, remaps=('none',)):
        self.scores = dict(
            (remap_name, SegmentationRunningScore(self._remap_len(remap_name))) # scores = dict mit remap name und leerer confusion matrix
            for remap_name in remaps
        )

    def _remap_len(self, remap_name):
        return len(self.REMAPS[remap_name])

    def _remap(self, remap_name, gt, pred):
        if remap_name == 'none':
            return (gt, pred)

        # TODO: cleanup and document
        gt_new = 255 + torch.zeros_like(gt)

        n, _, h, w = pred.shape
        c = self._remap_len(remap_name)
        device = pred.device
        dtype = pred.dtype

        pred_new = torch.zeros(n, c, h, w, device=device, dtype=dtype)

        for cls_to, clss_from in enumerate(self.REMAPS[remap_name]):
            clss_from = (clss_from, ) if isinstance(clss_from, int) else clss_from

            for cls_from in clss_from:
                gt_new[gt == cls_from] = cls_to
                pred_new[:,cls_to,:,:] += pred[:,cls_from,:,:]

        return (gt_new, pred_new)

    def update(self, gt, pred):
        pred = pred.exp()

        for remap_name, score in self.items():
            gt_remap, pred_remap = self._remap(remap_name, gt, pred) # nichts passiert

            score.update(
                gt_remap.numpy(),
                pred_remap.argmax(1).cpu().numpy()
            )

    def reset(self):
        for remap_name, score in self.items():
            score.reset()

    def items(self):
        return iter((key, self[key]) for key in self)

    def __iter__(self):
        return iter(self.scores)

    def __getitem__(self, key):
        return self.scores[key]
