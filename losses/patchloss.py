import torch
import torch.nn.functional as functional
import torchvision
import numpy as np

import losses as trn_losses

class PatchLosses(object):
    def __init__(self, device, disable_automasking=False):
        self.automasking = not disable_automasking  #disable_automasking 是一个布尔值，用于禁用自动掩码     
        self.ssim = trn_losses.SSIM().to(device)

    def Consistency_loss(self,outputs,outputs_masked):
        patchnums=outputs['patchnums']
        num=np.sum(patchnums)
        disp1=outputs['disp',0]
        disp2=outputs['disp_a',0]
        batchnum=outputs['disp',0].size(0)
        if outputs_masked is not None:   #提供了 outputs_masked，则应用移动掩码以屏蔽动态物体。
                moving_mask = outputs_masked['moving_mask']
                disp1=disp1*moving_mask
                disp2=disp2*moving_mask
        mse_loss = functional.mse_loss(disp1, disp2)#.mean(1, True)

        ssim = self.ssim(disp1, disp2).mean(1, True)
        if num!=0:
            ssim_loss=torch.sum(ssim)/(num*batchnum*128)
        else : ssim_loss=1e-7

        loss=0.2*ssim_loss+0.8*mse_loss

        return loss


    def patchloss_computer(self,dataset, output,output_masked):
        loss=2*self.Consistency_loss(output,output_masked)
        return loss
