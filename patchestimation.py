from operator import getitem
from torchvision.transforms import Compose
from torchvision.transforms import transforms
import torch.nn.functional as F
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from models.sgdepth import SGDepth 

#DPT
from dpt.models import DPTDepthModel

# MIDAS
import midas.utils
from midas.models.midas_net import MidasNet
from midas.models.transforms import Resize, NormalizeImage, PrepareForNet

#AdelaiDepth
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import strip_prefix_if_present

from Patch import Pacthdepth

import time
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
import warnings
warnings.simplefilter('ignore', np.RankWarning)


device = torch.device("cuda")


class boostingpatch(nn.Module):
    def __init__(self):

        super().__init__()
        # Load merge network
     
        self.patchselect=Pacthdepth()
        self.gsmask=self.generatemask((192,640))   #初始化一个高斯掩膜用于patch合并
        self.gsmask1=self.generatemask((160,160))
        self.gsmask1=self.gsmask1[32:,32:]  

        #self.gsmask1 = torch.tensor(self.gsmask1,device='cuda').unsqueeze(dim=0)
        self.initmodel()
        self.init_dpt()

    #生成一个高斯掩码用于图像合并
    def generatemask(self,size):
        # Generates a Guassian mask
        mask = np.zeros(size, dtype=np.float32)
        sigma = int(size[0]/16)
        k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
        mask[int(0.15*size[0]):size[0] - int(0.15*size[0]), int(0.15*size[1]): size[1] - int(0.15*size[1])] = 1
        mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = mask.astype(np.float32)
        return mask
    
    #初始化一个用于估计patch的预训练的模型
    def initmodel(self):
        print("Init Model...")
        sgdepth = SGDepth
        self.sgdmodel=sgdepth(split_pos=1, num_layers=50, grad_scale_depth=0.95, grad_scale_seg=0.05,
                 weights_init='pretrained', resolutions_depth=1, num_layers_pose=18,goal='depth')
        state = self.sgdmodel.state_dict()
        to_load = torch.load('/SATA2/wb/SGDepth-master/my_models/sgdepth_eccv_test/resnet50+fsa_epoch_19/model.pth')
        #final：'/SATA2/wb/SGDepth-master/my_models/sgdepth_eccv_test/SGD_resnet50/checkpoints/epoch_20/model.pth'
        #resnet50:'/data/wb_project/SGDepth-master/my_model/teacher/kitti_sgdepth_withoutdicsloss/checkpoints/epoch_20/model.pth')
        #18：/SATA2/wb/SGDepth-master/my_models/resnet18/epoch_20/model.pth
        for (k, v) in to_load.items():
            if k not in state:
                print(f"    - WARNING: Model file contains unknown key {k} ({list(v.shape)})")
        for (k, v) in state.items():
            if k not in to_load:
                print(f"    - WARNING: Model file does not contain key {k} ({list(v.shape)})")
            else:
                state[k] = to_load[k]
        self.sgdmodel.load_state_dict(state)
        self.sgdmodel = self.sgdmodel.eval().cuda()
        self.sgdmodel=self.sgdmodel.to('cuda')
    
    #数据处理
    def image_dict(self,image):
        image_dict = {('color_aug', 0, 0): image}  # dict
        image_dict[('color', 0, 0)] = image
        image_dict['domain'] = ['cityscapes_train_seg', ]
        image_dict['purposes'] = [['segmentation', ], ['depth', ]]
        image_dict['num_classes'] = torch.tensor([20])
        image_dict['domain_idx'] = torch.tensor(0)
        batch = (image_dict,)  # batch tuple
        return batch

    #模型训练patch深度，HR、MR、LR分辨率
    def sgdprocessdepth(self,image, weightHR,weightMR,weightLR):

        HRimage=F.interpolate(image, size=(int(1.5*image.size(2)),int(1.5*image.size(3))), mode='bilinear', align_corners=False).float()
        LRimage=F.interpolate(image, size=(int(0.5*image.size(2)),int(0.5*image.size(3))), mode='bilinear', align_corners=False).float()

        batch=self.image_dict(image) 
        batch1=self.image_dict(HRimage)
        batch2=self.image_dict(LRimage)

        with torch.no_grad():
                output = self.sgdmodel(batch)
                output1 = self.sgdmodel(batch1)
                output2 = self.sgdmodel(batch2)

        disps_pred = output[0]["disp", 0]
        disp_predHR=output1[0]["disp", 0]
        disp_predHR = F.interpolate(disp_predHR, size=(disps_pred.size(2), disps_pred.size(3)), mode='bilinear', align_corners=False)
        disp_predLR=output2[0]["disp", 0]
        disp_predLR = F.interpolate(disp_predLR, size=(disps_pred.size(2), disps_pred.size(3)), mode='bilinear', align_corners=False)

        
        depth_z=disps_pred[7].squeeze().clone().cpu().detach().numpy()
        depth_map_normalized1 = ((depth_z - np.min(depth_z)) / (np.max(depth_z) - np.min(depth_z))) * 255
        depth_map_uint81 = depth_map_normalized1.astype(np.uint8)
        depth_map_color1 = cv2.applyColorMap(depth_map_uint81.astype(np.uint8), cv2.COLORMAP_JET)  
        cv2.imshow('Depth_patch jubuMR', depth_map_color1)
        cv2.waitKey(0)
        
        depth_z=disp_predHR[7].squeeze().clone().cpu().detach().numpy()
        depth_map_normalized1 = ((depth_z - np.min(depth_z)) / (np.max(depth_z) - np.min(depth_z))) * 255
        depth_map_uint81 = depth_map_normalized1.astype(np.uint8)
        depth_map_color1 = cv2.applyColorMap(depth_map_uint81.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Depth_patch jubuHR', depth_map_color1)
        cv2.waitKey(0)
        
        depth_z=disp_predLR[7].squeeze().clone().cpu().detach().numpy()
        depth_map_normalized1 = ((depth_z - np.min(depth_z)) / (np.max(depth_z) - np.min(depth_z))) * 255
        depth_map_uint81 = depth_map_normalized1.astype(np.uint8)
        depth_map_color1 = cv2.applyColorMap(depth_map_uint81.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('Depth_patch jubuLR', depth_map_color1)
        cv2.waitKey(0)
        
        disps=disps_pred*weightMR+disp_predHR*weightHR+disp_predLR*weightLR

        return disps

    #根据patch生成patch的原图和原深度
    def _process_patch_image(self,batch,patchlist,depths):

            le=batch[0]['color_aug', 0, 0].size(0)
            batch_patchs=list()
            for i in range(le):
                patches=patchlist[str(i)]
                img_cur=batch[0]['color_aug', 0, 0][i]
                patch_depth=list() #存放一帧的所有patch深度数据
                depth=depths[i]
                for patch in patches:
                    patch_idx=patch[0]
                    patch_point=(patch[1]['rect'][0],patch[1]['rect'][1])
                    patch_size=patch[1]['size']

                    img=img_cur[:,patch_point[0]:patch_point[0]+patch_size,patch_point[1]:patch_point[1]+patch_size]
                    depth_x=depth[:,patch_point[0]:patch_point[0]+patch_size,patch_point[1]:patch_point[1]+patch_size]

                    #补丁序号、[补丁起点，补丁大小]，彩色图 组合成元组
                    combined_tuple=(patch_idx,[patch_point,patch_size],img,depth_x)
                    patch_depth.append(combined_tuple)
                
                batch_patchs.append(patch_depth)

            return batch_patchs

    def normalize_depth_map(self,depth_map, mean, std):
        depth_map[depth_map == 0] = 1e-6
        min_value = torch.min(depth_map)
        max_value = torch.max(depth_map)
        normalized_depth_map = (depth_map - min_value) / (max_value - min_value) * mean + std
        return normalized_depth_map


    def edge_estimation(self,image_tensor):
        
        image=image_tensor.cpu().detach().numpy()
        image_rgb = np.transpose(image, (1, 2, 0))

        # 将 RGB 图像转换为灰度图像
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        image_gray = image_gray.astype(np.uint8)
        
        '''
        laplacian_edges = cv2.Laplacian(image_gray, cv2.CV_8U)

        # 显示边缘图像
        cv2.imshow('Edges', laplacian_edges)
        cv2.waitKey(0)
        cv2.destroyWindow('Edges')'''

        # 使用 Laplacian 算子计算边缘
        laplacian_edges = torch.tensor(cv2.Laplacian(image_gray, cv2.CV_8U),device='cuda')
        
        #计算边缘点数量
        nonzero_indices = torch.nonzero(laplacian_edges)
        edge_pixel_count = nonzero_indices.size(0)

        return laplacian_edges,edge_pixel_count

    def patchdepth_distance(self,edgepoint):
        # 获取 edgepoint 中值为 1 的像素的坐标
        edge_coordinates = torch.nonzero(edgepoint).float()

        # 初始化一个形状为 [128, 128] 的距离地图并将其移到 CUDA 设备上
        height, width = edgepoint.shape
        distance_map = torch.zeros(height, width, device='cuda')

        # 创建像素坐标网格
        y_indices, x_indices = torch.meshgrid(torch.arange(height, device='cuda'), torch.arange(width, device='cuda'))

        # 将像素坐标网格扁平化为一维张量
        pixel_coordinates = torch.stack((y_indices.flatten(), x_indices.flatten()), dim=1).float()

        # 使用广播计算所有像素与边缘点的欧几里得距离
        pixel_coordinates_expanded = pixel_coordinates[:, None, :]
        distances = torch.norm(pixel_coordinates_expanded - edge_coordinates, dim=2)

        # 对于每个像素，找到最近的边缘点的距离
        min_distances, _ = torch.min(distances, dim=1)

        # 将距离结果还原为形状为 [128, 128] 的地图
        distance_map = min_distances.view(height, width)
        distance_map_euclidean_tensor=distance_map

        return distance_map_euclidean_tensor
        

    def patchdepth_weight(self,distance_map_euclidean_tensor,distancemin=4,distancemax=10):
        #不同分辨率深度图融合权重举止
        # D<K1:[0.3,0.7,0]
        # D>K2:[0.1,0.7,0.2]
        # K1<D<K2:[0.2,0.7,0.1]
        # 创建权重矩阵
        weight_matrixHR = torch.zeros_like(distance_map_euclidean_tensor, dtype=torch.float32)
        weight_matrixMR = torch.zeros_like(distance_map_euclidean_tensor, dtype=torch.float32)
        weight_matrixLR = torch.zeros_like(distance_map_euclidean_tensor, dtype=torch.float32)

        # 计算权重矩阵
        condition1 = distance_map_euclidean_tensor < distancemin
        condition2 = distance_map_euclidean_tensor > distancemax
        condition3 = (distance_map_euclidean_tensor >= distancemin) & (distance_map_euclidean_tensor <= distancemax)

        weight_matrixHR[condition1] = 0.3
        weight_matrixHR[condition2] = 0.15
        weight_matrixHR[condition3] = 0.25

        weight_matrixMR[condition1] = 0.7
        weight_matrixMR[condition2] = 0.7
        weight_matrixMR[condition3] = 0.7

        weight_matrixLR[condition1] = 0
        weight_matrixLR[condition2] = 0.15
        weight_matrixLR[condition3] = 0.05

        return weight_matrixHR,weight_matrixMR,weight_matrixLR



    def run(self,batch,outputs):

        depth_y=outputs.clone()
        
        #batchsize：batch的图片数量
        batchsize=batch[0]['color_aug', 0, 0].size(0)

        #获取patch信息
        patchs=self.patchselect.Patchselect(batch,batchsize)
        batch_patchs=self._process_patch_image(batch,patchs,outputs) 

        patchnums=[] 

        for patch_ind in range(len(batch_patchs)):
            
            depth_zero = torch.zeros(1, 1,  depth_y.size(2),  depth_y.size(3),device='cuda')
            depth_zero1 = torch.zeros(1, 1,  depth_y.size(2),  depth_y.size(3),device='cuda')
            patchi=batch_patchs[patch_ind]

            patchnums.append(len(patchi))#patchnums：patch数量

            patch_imgs= list()
            patch_weightHR=list()
            patch_weightMR=list()
            patch_weightLR=list()

            for j in range(len(patchi)): #遍历补丁,得到patch估计不同分辨率的合成权重
                
                patch=patchi[j]   
                patch_imgs.append(patch[2])

                patch_depthy=patch[3]
                patch_rgb=patch[2]
                rect=patch[1]
                laplacian_edges,numpoints=self.edge_estimation(patch_rgb)

                distancemin,distancemax=0,0
                if numpoints>0:
                    distancemin=0.1*(rect[1]**2)/(numpoints)
                    distancemax=2*(rect[1]**2)/(numpoints)
                
                distance_map_euclidean=self.patchdepth_distance(laplacian_edges)
                weightHR,weightMR,weightLR=self.patchdepth_weight(distance_map_euclidean,distancemin,distancemax)
                patch_weightHR.append(weightHR.unsqueeze(0))
                patch_weightMR.append(weightMR.unsqueeze(0))
                patch_weightLR.append(weightLR.unsqueeze(0))

            combined_imgs = torch.cat([img.unsqueeze(0) for img in patch_imgs], dim=0)
            combined_weightHR=torch.cat([weightHR.unsqueeze(0) for weightHR in patch_weightHR], dim=0)
            combined_weightMR=torch.cat([weightMR.unsqueeze(0) for weightMR in patch_weightMR], dim=0)
            combined_weightLR=torch.cat([weightLR.unsqueeze(0) for weightLR in patch_weightLR], dim=0)

            #计算patch深度
            patch_estimation=self.sgdprocessdepth(combined_imgs, combined_weightHR,combined_weightMR,combined_weightLR)
            prediction=self.dpt_depth(combined_imgs)
            
            for j in range(len(patchi)):
                #从最后一个patch开始添加，图片越下方越靠前
                patch=patchi[-(j+1)]
                estimation=patch_estimation[-(j+1)]
                estimation2=prediction[-(j+1)]
                patch_depthy=patch[3]
                patch_rgb=patch[2]
                rect=patch[1]

                mean=torch.min(patch_depthy)
                std=torch.std(patch_depthy)
                estimation=self.normalize_depth_map(estimation,mean,std).unsqueeze(0)   
                estimation2=self.normalize_depth_map(estimation2,mean,std).unsqueeze(0)  
                     
                data1_np = estimation.squeeze().cpu().detach().numpy()
                data2_np = patch_depthy.squeeze().squeeze().cpu().detach().numpy()
                data3_np = estimation2.squeeze().cpu().detach().numpy()
                
                p_coef = np.polyfit(data1_np.reshape(-1), data2_np.reshape(-1), deg=1)
                merged = np.polyval(p_coef, data1_np.reshape(-1)).reshape(data1_np.shape)
                estimation=torch.from_numpy(merged).to(patch_depthy.device)
                
                p_coef = np.polyfit(data3_np.reshape(-1), data2_np.reshape(-1), deg=1)
                merged = np.polyval(p_coef, data3_np.reshape(-1)).reshape(data3_np.shape)
                estimation2=torch.from_numpy(merged).to(patch_depthy.device)
                

                #设计一个用于不同patch合成到depth_zero上的掩膜
                maskxx=(depth_zero != 0).float().to('cuda')
                maskx=maskxx[:,:,rect[0][0]:rect[0][0]+rect[1],rect[0][1]:rect[0][1]+rect[1]]
                maskx1=  maskx.cpu().detach().numpy()
                distance_transform = np.ones_like(maskx1)
                mask=np.zeros_like(maskx1)
                
                distance_transform[:,:,:, 128 - 1] = 0  # 右边
                distance_transform = np.logical_or(distance_transform, 1-maskx1)
                distance_transform = (distance_transform_edt(distance_transform)+1)
                  
                if rect[0][0]>40:
                    mask= maskx1*distance_transform
                    mask=torch.from_numpy(mask).to(patch_depthy.device)
                    max_mask=torch.max(mask)
                    #print(max_mask)
                    if max_mask>0:
                        mask=mask/max_mask+(1-maskx)
                    else: mask=1-maskx
                else :                   
                    mask=maskx1*self.gsmask1
                    mask=torch.from_numpy(mask).to(patch_depthy.device)
                    mask=mask+(1-maskx)
                    
                depth_zero[:,:,rect[0][0]:rect[0][0]+rect[1],rect[0][1]:rect[0][1]+rect[1]]=depth_zero[:,:,rect[0][0]:rect[0][0]+rect[1],rect[0][1]:rect[0][1]+rect[1]]*(1-mask)+mask*estimation  
                depth_zero1[:,:,rect[0][0]:rect[0][0]+rect[1],rect[0][1]:rect[0][1]+rect[1]]=depth_zero1[:,:,rect[0][0]:rect[0][0]+rect[1],rect[0][1]:rect[0][1]+rect[1]]*(1-mask)+mask*estimation2     
            
            depth_zero=depth_zero.squeeze().squeeze()
            depth_zero1=depth_zero1.squeeze().squeeze()

           
            mask_indices = (depth_zero !=0).nonzero(as_tuple=True)
            
            #从depth_zero中得到非0元素的位置
            depth_zero=depth_zero.squeeze()
            #print(depth_zero.size())
            depth_zero1=depth_zero1.squeeze()

            data1 = depth_zero[mask_indices]
            data2 = depth_y[patch_ind].squeeze()[mask_indices]
            data3=depth_zero1[mask_indices]


            data1_np = data1.cpu().detach().numpy()
            data2_np = data2.cpu().detach().numpy()
            data3_np = data3.cpu().detach().numpy()

            # 进行多项式拟合（使用NumPy函数）
            p_coef = np.polyfit(data1_np, data2_np, deg=1)
            adjusted_data1_np = np.polyval(p_coef, data1_np)
            
            
            p_coef = np.polyfit(data3_np, data2_np, deg=1)
            adjusted_data2_np = np.polyval(p_coef, data3_np)
            adjusted_data2 = torch.from_numpy(adjusted_data2_np).to(data1.device)
            depth_zero1[depth_zero!=0]=adjusted_data2.unsqueeze(0)
            
            adjusted_data1 = torch.from_numpy(adjusted_data1_np).to(data1.device)
            depth_zero[depth_zero!=0]=adjusted_data1.unsqueeze(0)
            
            depth_zero1_new = 0.6* depth_zero + 0.4* depth_zero1


            #patch合成在空白张量上的图
            '''
            depth_z=depth_zero.clone().cpu().detach().numpy()
            depth_map_normalized1 = ((depth_z - np.min(depth_z)) / (np.max(depth_z) - np.min(depth_z))) * 255
            depth_map_uint81 = depth_map_normalized1.astype(np.uint8)
            depth_map_color1 = cv2.applyColorMap(depth_map_uint81.astype(np.uint8), cv2.COLORMAP_JET)
            
            depth_z=depth_zero1.clone().cpu().detach().numpy()
            depth_map_normalized1 = ((depth_z - np.min(depth_z)) / (np.max(depth_z) - np.min(depth_z))) * 255
            depth_map_uint81 = depth_map_normalized1.astype(np.uint8)
            depth_map_color3 = cv2.applyColorMap(depth_map_uint81.astype(np.uint8), cv2.COLORMAP_JET)
            
            depth_z=depth_zero1_new.clone().cpu().detach().numpy()
            depth_map_normalized2 = ((depth_z - np.min(depth_z)) / (np.max(depth_z) - np.min(depth_z))) * 255
            depth_map_uint82 = depth_map_normalized2.astype(np.uint8)
            depth_map_color2 = cv2.applyColorMap(depth_map_uint82.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow('Depth_patch Mapf', depth_map_color1)
            cv2.imshow('Depth_patch Map_dpt', depth_map_color3)
            cv2.imshow('Depth_patch Mapf  _dpt', depth_map_color2)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
            '''
            #融合掩码mask_final
            maskf=((depth_zero!=0).float()) 

            non_zero_indices = torch.nonzero(maskf)

            if non_zero_indices.numel() > 0:
                # 获取最小和最大 x 和 y 坐标值
                min_x = torch.min(non_zero_indices[:, 1])
                max_x = torch.max(non_zero_indices[:, 1])
                min_y = torch.min(non_zero_indices[:, 0])
                max_y = torch.max(non_zero_indices[:, 0])

                dx=max_x-min_x+1
                dy=max_y-min_y+1

            gsmask = torch.tensor(self.gsmask,device='cuda').unsqueeze(dim=0)
            gsmask = F.interpolate(gsmask.unsqueeze(0), size=(dy, dx), mode='bilinear', align_corners=False).squeeze()
            # 缩放因子和最小值
            scale_factor = 0.5
            min_value = 0.01
            # 执行线性变换
            gsmask = (gsmask * scale_factor) + min_value
            
            
            maskf=maskf.unsqueeze(dim=0)
            maskf[:,min_y:max_y+1,min_x:max_x+1]=maskf[:,min_y:max_y+1,min_x:max_x+1]*gsmask
            depth_y[patch_ind]=depth_y[patch_ind]*(1-0.6*maskf)+depth_zero1_new*0.6*maskf
            '''            
            patch_depthy= depth_y[0].squeeze().cpu().detach().numpy()
            depth_map_normalized = ((patch_depthy - np.min(patch_depthy)) / (np.max(patch_depthy) - np.min(patch_depthy))) * 255
            depth_map_uint8 = depth_map_normalized.astype(np.uint8)
            depth_map_color = cv2.applyColorMap(depth_map_uint8.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow('Depth after patch',depth_map_color)
            cv2.waitKey(0)'''

        return depth_y,patchnums
    
    def init_dpt(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True   
        model_path='/SATA2/wb/ljcdp/SGDepth-master_final/weights/dpt_hybrid-midas-501f0c75.pt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dptmodel = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.dptmodel.eval()

        self.dptmodel.to(device)

    def dpt_depth(self,img):
        prediction = self.dptmodel.forward(img)#[0].squeeze())
        # 计算最小值和最大值
        min_val = prediction.min()
        max_val = prediction.max()

        # 将张量归一化到0-1范围内
        prediction = ((prediction - min_val) / (max_val - min_val)).unsqueeze(1)
        for i in range(0,prediction.size(0)):
            #print(type(prediction),prediction.size())
            depth_z=prediction[i,:,:,:].squeeze().clone().cpu().detach().numpy()
            depth_map_normalized1 = ((depth_z - np.min(depth_z)) / (np.max(depth_z) - np.min(depth_z))) * 255
            depth_map_uint81 = depth_map_normalized1.astype(np.uint8)
            depth_map_color1 = cv2.applyColorMap(depth_map_uint81.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow('Depth_patch xjubuLR', depth_map_color1)
            cv2.waitKey(0)

        return  prediction



    