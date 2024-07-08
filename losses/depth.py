import torch
import torch.nn.functional as functional
from skimage import measure
import losses as trn_losses
import numpy as np
import torch.nn as nn
import cv2



class DepthLosses(object):
    def __init__(self, device, disable_automasking=False, avg_reprojection=False, disparity_smoothness=0):
        self.automasking = not disable_automasking  #disable_automasking 是一个布尔值，用于禁用自动掩码。
        self.avg_reprojection = avg_reprojection #avg_reprojection 是一个布尔值，表示是否对重投影损失进行平均。
        self.disparity_smoothness = disparity_smoothness  #disparity_smoothness 是视差平滑度的权重。
        self.scaling_direction = "up"
        self.masked_supervision = True
        self.segmask=[5,6,7,8,9,11,12,13,14,15,16,17,18]
        # noinspection PyUnresolvedReferences
        self.ssim = trn_losses.SSIM().to(device)
        self.smoothness = trn_losses.SmoothnessLoss()

    def _combined_reprojection_loss(self, pred, target):  #计算一批预测图像和目标图像之间的重投影损失
        """Computes reprojection losses between a batch of predicted and target images
        """
        #print('pred:',pred.size())
        # Calculate the per-color difference and the mean over all colors
        l1 = (pred - target).abs().mean(1, True)  #计算预测图像和目标图像之间的绝对差异，并在每个颜色通道上取平均

        ssim = self.ssim(pred, target).mean(1, True) #计算预测图像和目标图像之间的结构相似性指数（SSIM）。

        reprojection_loss = 0.85 * ssim + 0.15 * l1  #将 SSIM 和 L1 损失按照权重进行线性组合，得到最终的重投影损失。

        return reprojection_loss

    def _reprojection_losses(self, inputs, outputs, outputs_masked):  #计算一个小批量输入和输出的重投影损失和平滑损失（加入了mask ）
        """Compute the reprojection and smoothness losses for a minibatch
        """
        
        frame_ids = tuple(frozenset(k[1] for k in outputs if k[0] == 'color')) #从输出中提取帧 ID 和分辨率，这是为了获取相关的图像和深度信息。
        resolutions = tuple(frozenset(k[2] for k in outputs if k[0] == 'color'))
        losses = dict()

        color = inputs["color", 0, 0]#选择了一个参考帧，这里使用了索引为 (0, 0) 的输入图像作为参考。
        target = inputs["color", 0, 0]
        #print('x:',target.size())

        # Compute reprojection losses for the unwarped input images
        identity_reprojection_loss = tuple(          #计算未经畸变的输入图像的重投影损失，即输入图像与参考图像之间的重投影损失。
            self._combined_reprojection_loss(inputs["color", frame_id, 0], target)
            for frame_id in frame_ids
        )
        identity_reprojection_loss = torch.cat(identity_reprojection_loss, 1)

        if self.avg_reprojection:  #如果启用了 avg_reprojection，对重投影损失进行平均
            identity_reprojection_loss = identity_reprojection_loss.mean(1, keepdim=True)

        for resolution in resolutions:#对于每个分辨率，计算每个帧与参考帧之间的重投影损失
            # Compute reprojection losses (prev frame to cur and next frame to cur)
            reprojection_loss = tuple(
                self._combined_reprojection_loss(outputs["color", frame_id, resolution], target)
                for frame_id in frame_ids
            )
            reprojection_loss = torch.cat(reprojection_loss, 1)
           

            if self.avg_reprojection:
                reprojection_loss = reprojection_loss.mean(1, keepdim=True)

            if self.automasking:    #启用了 automasking，将未经畸变的输入图像和当前帧之间的重投影损失拼接起来，选择每个像素的最小损失。
                                    #这对于处理图像边界、信息缺失或某些输入图像中被遮挡的区域是有用的。
                reprojection_loss = torch.cat(
                    (identity_reprojection_loss, reprojection_loss), 1
                )

                reprojection_loss, idxs = torch.min(reprojection_loss, dim=1)

            # Segmentation moving mask to mask DC objects
            if outputs_masked is not None:   #提供了 outputs_masked，则应用移动掩码以屏蔽动态物体。
                moving_mask = outputs_masked['moving_mask']
                reprojection_loss = reprojection_loss * moving_mask

            loss = reprojection_loss.mean()   #计算损失的平均值，即所有像素的损失的均值。
            '''
            #print('x',loss)
            if(0!=outputs['segmentation_mask',0].size(0) and (outputs['segmentation_mask', 0] != 0).any()):
                reprojection_loss_extra = tuple(
                    self._combined_reprojection_loss(outputs["color", frame_id, resolution]*outputs['segmentation_mask',0], target*outputs['segmentation_mask',0])
                    for frame_id in frame_ids
                )
                num=outputs['segmentation_mask',0].sum()
                reprojection_loss_extra = torch.cat(reprojection_loss_extra, 1)
                loss += reprojection_loss_extra.sum()/ num
                #print(reprojection_loss_extra)'''

            if self.disparity_smoothness != 0:  #disparity_smoothness 不为零，则计算视差平滑损失。这是为了鼓励生成的深度图在空间上平滑，以减少深度图中的噪声。
                disp = outputs["disp", resolution]

                ref_color = functional.interpolate(
                    color, disp.shape[2:], mode='bilinear', align_corners=False
                )

                mean_disp = disp.mean((2, 3), True)
                norm_disp = disp / (mean_disp + 1e-7)

                disp_smth_loss = self.smoothness(norm_disp, ref_color)
                disp_smth_loss = self.disparity_smoothness * disp_smth_loss / (2 ** resolution)

                losses[f'disp_smth_loss/{resolution}'] = disp_smth_loss

                loss += disp_smth_loss


            losses[f'loss/{resolution}'] = loss

        losses['loss_depth_reprojection'] = sum(
            losses[f'loss/{resolution}']
            for resolution in resolutions
        ) / len(resolutions)

        return losses

    def depth_seg_losses(self, outputs, outputs_mask):
        loss,seg_num  = 0,0
        mask = outputs['segmentation_mask', 0]
        offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        disp = outputs['disp', 0]

        # for r in resolutions:
        for i in self.segmask:  # 每一个语义标签类
            maskx = mask.clone()
            maskx[maskx != i] = 0
            maskx[maskx == i] = 1
            for j in range(maskx.size(0)):  # 每一个batch
                if torch.sum(maskx[j]) > 10:
                    seg_num += 1
                    depth_mask = maskx[j] * disp[j]
                    # print(mask[j].size())
                    for dx, dy in offsets:
                        relative_depth_diff = (depth_mask - depth_mask.roll(shifts=(dy, dx), dims=(0, 1))).roll(shifts=(-dy, -dx), dims=(0, 1))

                        maskf = self.retain_contours(maskx[j].squeeze()).unsqueeze(0).to('cuda')
                        if  torch.sum(maskf)>0:
                            selected_values = relative_depth_diff[maskf == 1]
                            selected_depth = depth_mask[maskf == 1]

                            # average_value = torch.mean(selected_values)

                            result = torch.clamp(selected_values - 0.05 * selected_depth, min=0)

                            sum_result = torch.sum(result * 255) / torch.sum(maskf)

                            loss += sum_result
        if seg_num!=0:
            loss = loss * 10 / seg_num / 4
        else: loss=0
        return loss

    
    
    def retain_contours(self,mask):
        if not torch.is_tensor(mask) or mask.dim() != 2:
            raise ValueError("Input must be a 2D PyTorch tensor.")

        # 定义卷积核，用于检查上、下、左、右是否都为1
        kernel = torch.tensor([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=mask.dtype, device=mask.device)

        # 使用卷积操作来检查上、下、左、右是否都为1
        conv_mask = functional.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0).float(), padding=1)

        # 将卷积结果二值化，保留上、下、左、右都为1的部分
        new_mask = (conv_mask == 5).squeeze().byte()

        return new_mask
    
    def compute_losses(self, inputs, outputs, outputs_masked):
        losses = self._reprojection_losses(inputs, outputs, outputs_masked)
        losses['consistency loss']=self.depth_seg_losses(outputs,outputs_masked)
        losses['loss_depth'] =2*losses['loss_depth_reprojection']+ losses['consistency loss']
        #losses['loss_depth'] = losses['loss_depth_reprojection']

        return losses
