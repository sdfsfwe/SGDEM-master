import time
import os
import torch
import cv2
import numpy as np
import argparse
import warnings
import torch
import torch.nn as nn
from operator import getitem
warnings.simplefilter('ignore', np.RankWarning)


class Pacthdepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        self.factor = 0.5
        self.whole_size_threshold = 3000  # R_max from the paper
        self.GPU_threshold = 1600 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted 
    

    def rgb2gray(self,rgb):
        # Converts rgb to gray
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    

    def applyGridpatch(self,blsize, stride, img, box):
        # Extract a simple grid patch.
        counter1 = 0
        patch_bound_list = {}
        for k in range(blsize, img.shape[1] - blsize, stride):
            for j in range(blsize, img.shape[0] - blsize, stride):
                patch_bound_list[str(counter1)] = {}
                patchbounds = [j - blsize, k - blsize, j - blsize + 2 * blsize, k - blsize + 2 * blsize]
                patch_bound = [box[0] + patchbounds[1], box[1] + patchbounds[0], patchbounds[3] - patchbounds[1],
                            patchbounds[2] - patchbounds[0]]
                patch_bound_list[str(counter1)]['rect'] = patch_bound
                patch_bound_list[str(counter1)]['size'] = patch_bound[2]
                counter1 = counter1 + 1
        return patch_bound_list

    def getGF_fromintegral(self,integralimage, rect):
        # Computes the gradient density of a given patch from the gradient integral image.
        x1 = rect[1]
        x2 = rect[1]+rect[3]
        y1 = rect[0]
        y2 = rect[0]+rect[2]
        value = integralimage[x2, y2]-integralimage[x1, y2]-integralimage[x2, y1]+integralimage[x1, y1]
        return value

    # Generating local patches to perform the local refinement described in section 6 of the main paper.
    '''这段代码描述了一个生成补丁列表的过程，用于在图像处理中利用上下文线索。它首先计算图像的梯度，
    然后通过积分图像和自适应选择来生成合适的补丁，最后将补丁按照大小排序。这些补丁可以在后续的深度估计等任务中使用。'''
    def generatepatchs(self,img, base_size=128):
    
        # Compute the gradients as a proxy of the contextual cues.
        img_gray = self.rgb2gray(img)
        whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) +\
            np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

        threshold = (whole_grad[whole_grad > 0].mean())
        whole_grad[whole_grad < threshold] = 0

        # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
        gf = whole_grad.sum()/len(whole_grad.reshape(-1))
        grad_integral_image = cv2.integral(whole_grad)

        # Variables are selected such that the initial patch size would be the receptive field size
        #    and the stride is set to 1/3 of the receptive field size.
        blsize = int(round(base_size/2))
        stride = int(round(blsize*0.75)) 

        # Get initial Grid
        patch_bound_list = self.applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

        # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
        # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
        #print("Selecting patchs ...")
        patch_bound_list = self.adaptiveselection(grad_integral_image, patch_bound_list, gf)

        # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
        # patch
        patchset = sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], 'size'), reverse=True)
        #print(patchset)
        return patchset
    
    
    # Adaptively select patches

    def adaptiveselection(self,integral_grad, patch_bound_list, gf):
        patchlist = {}
        count = 0
        height, width = integral_grad.shape

        search_step = int(32/self.factor)

        # Go through all patches
        for c in range(len(patch_bound_list)):
            # Get patch
            bbox = patch_bound_list[str(c)]['rect']

            # Compute the amount of gradients present in the patch from the integral image.
            cgf = self.getGF_fromintegral(integral_grad, bbox)/(bbox[2]*bbox[3])

            # Check if patching is beneficial by comparing the gradient density of the patch to
            # the gradient density of the whole image
            if cgf >= gf:
                bbox_test = bbox.copy()
                patchlist[str(count)] = {}

                # Enlarge each patch until the gradient density of the patch is equal
                # to the whole image gradient density
                while True:

                    bbox_test[0] = bbox_test[0] - int(search_step/2)
                    bbox_test[1] = bbox_test[1] - int(search_step/2)

                    bbox_test[2] = bbox_test[2] + search_step
                    bbox_test[3] = bbox_test[3] + search_step

                    # Check if we are still within the image
                    if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                            or bbox_test[0] + bbox_test[2] >= width:
                        break

                    # Compare gradient density
                    cgf = self.getGF_fromintegral(integral_grad, bbox_test)/(bbox_test[2]*bbox_test[3])
                    if cgf < gf:
                        break
                    bbox = bbox_test.copy()

                # Add patch to selected patches
                patchlist[str(count)]['rect'] = bbox
                patchlist[str(count)]['size'] = bbox[2]
                count = count + 1
        
        # Return selected patches
        return patchlist
    
    def Patchselect(self,batch,batch_size):
        patchdict=dict()
        if batch_size==1:
            img=batch[0]['color_aug', 0, 0][0].cpu().detach().numpy()
            #print(img.shape)
            img = img.transpose(2, 1, 0)
            p=self.generatepatchs(img)
            patchdict['0']=p
        else :
            for i in range(batch_size):
                imgs=batch[0]['color_aug', 0, 0].cpu().detach().numpy()
                #print(imgs.shape)
                #img=(imgs[i] * 255).astype(np.uint8)
                img = imgs[i].transpose(2, 1, 0)
                p=self.generatepatchs(img)
                patchdict[str(i)]=p

        return patchdict



if __name__ == "__main__":
    patch=Pacthdepth()
    img=cv2.imread('/SATA2/wb/SGDepth-master_xx/imgs/intro.png')
    # 转换为浮点数
    img_float = img.astype(np.float32)

    # 归一化操作
    img_normalized = (img_float / 255.0) * 2.0 - 1.0
    img=cv2.resize(img,(192,640))
    print(img.shape)
    p=patch.generatepatchs(img,128)
    print(p)
    print(len(p))


