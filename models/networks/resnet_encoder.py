import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
import cv2
import numpy as np

RESNETS = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}

class DilatedConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(DilatedConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ResnetEncoder(nn.Module):
    """A ResNet that handles multiple input images and outputs skip connections"""

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super().__init__()

        if num_layers not in RESNETS:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        self.encoder = RESNETS[num_layers](pretrained)

        self.encoder.conv1.weight.data = self.encoder.conv1.weight.data.repeat(
            (1, num_input_images, 1, 1)
        ) / num_input_images

        # Change attribute "in_channels" for clarity
        self.encoder.conv1.in_channels = num_input_images * 3  # Number of channels for a picture = 3

        # Remove fully connected layer
        self.encoder.fc = None

        if num_layers > 34:
            self.num_ch_enc = (64, 256,  512, 1024, 2048)
        else:
            self.num_ch_enc = (64, 64, 128, 256, 512)
        
        dilation = 2
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2

        self.dilated_conv = DilatedConvModule(
            in_channels=self.num_ch_enc[4], 
            out_channels=self.num_ch_enc[4], 
            kernel_size=kernel_size, 
            dilation=dilation,
            padding=padding
        )

        self.fusion_module = FeatureFusion()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.convx=nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
    
    def mypool(self,x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        fused_feature = torch.cat((max_pooled, avg_pooled, x), dim=1)
        fused_feature = self.convx(fused_feature)
        fused_feature = torch.relu(fused_feature)

        return fused_feature
        


    def forward(self, l_0):
        l_0 = self.encoder.conv1(l_0)
        l_0 = self.encoder.bn1(l_0)
        l_0 = self.encoder.relu(l_0)

        l_1 = self.encoder.maxpool(l_0)
        l_1 = self.encoder.layer1(l_1)

        l_2 = self.encoder.layer2(l_1)
        l_3 = self.encoder.layer3(l_2)
        l_4 = self.encoder.layer4(l_3)
        '''
        #l_4= self.dilated_conv(l_4)
        print("l_0.size:",l_0.size())
        print("l_1.size:",l_1.size())
        print("l_2.size:",l_2.size())
        print("l_3.size:",l_3.size())
        print("l_4.size:",l_4.size())
        
        
        l_0_w=self.fusion_module.multihead_self_attention_chunked1(l_0,4)
        print("l_4_w.size:",l_0_w.size())
        
        # Convert the tensor to a NumPy array
        l_0_w_np = l_0_w.cpu().detach().numpy()  # Convert to NumPy and move to CPU if necessary

        # Rescale values to 0-255 (assuming it's a single-channel image)
        l_0_w_np = ((l_0_w_np - l_0_w_np.min()) / (l_0_w_np.max() - l_0_w_np.min()) * 255).astype(np.uint8)

        # Remove the first dimension (axis 0) and select channels 0 to 2 (0-based index)
        l_0_w_np = l_0_w_np[0, 0, :, :]

        # Use OpenCV to display the image
        cv2.imshow("l_0_w", l_0_w_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
                
        #l_0=self.fusion_module.multihead_self_attention_chunked(l_0,1)
        l_1=self.fusion_module.multihead_self_attention_chunked(l_1,1)
        l_2=self.fusion_module.multihead_self_attention_chunked(l_2,1)
        l_3=self.fusion_module.multihead_self_attention_chunked(l_3,1)
        l_4=self.fusion_module.multihead_self_attention_chunked(l_4,1)

        
        return (l_0, l_1, l_2, l_3, l_4)
    

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.device = torch.device("cuda")
        self.num_heads=8
        self.softmax = nn.Softmax(dim=2)
        '''输入为：
        in_channels：resnet18:[64, 64, 128, 256, 512]
                     resnet50:[64, 256,  512, 1024, 2048]
        out_channels:             
        out_size:(h,w)  '''

    def upsample_layers(self,input_tensor,output_tensor):
        upsample_ratio = (output_tensor.shape[2] / input_tensor.shape[2], output_tensor.shape[3] / input_tensor.shape[3])

        # 使用上采样函数进行上采样
        upsampled_input = F.interpolate(input_tensor, scale_factor=upsample_ratio, mode='bicubic',align_corners=True)
        conv = nn.Conv2d(in_channels=upsampled_input.shape[1], out_channels=output_tensor.shape[1], kernel_size=1, stride=1, padding=0)
        conv=conv.to(self.device)
        output = conv( upsampled_input)
        return output
    
    
    def downsample_layers(self,input_tensor,output_tensor):
        # 定义卷积层，实现空间维度上的降采样
        conv = nn.Conv2d(in_channels=input_tensor.shape[1], out_channels=input_tensor.shape[1], kernel_size=3, stride=2, padding=1)
        conv = conv.to(self.device)
        while(input_tensor.shape[2]>output_tensor.shape[2]):
        # 将输入张量通过卷积层进行降采样
            input_tensor = conv(input_tensor)
        conv1 = nn.Conv2d(in_channels=input_tensor.shape[1], out_channels=output_tensor.shape[1], kernel_size=1, stride=1, padding=0)
        conv1=conv1.to(self.device)
        output = conv1( input_tensor)
        return output
    
        
    def computer(self,inputs,target_level):
        #self.attention = MultiHeadAttention(in_channels=inputs[target_level].size()[1]*len(inputs)).to(self.device)
        target=[]
        for pos in range(len(inputs)):
            if pos==target_level:
                feature=self.multihead_self_attention_chunked(inputs[pos],4)
                target.append(feature)
            elif pos<target_level:
                feature=self.downsample_layers(inputs[pos],inputs[target_level])
                feature=self.multihead_self_attention_chunked(feature,4)
                target.append(feature)
            else:
                feature=self.upsample_layers(inputs[pos],inputs[target_level])
                feature=self.multihead_self_attention_chunked(feature,4)
                target.append(feature)
        feature_pyramid = torch.cat(target, dim=1)

        #feature_pyramid_attention = self.multihead_self_attention(feature_pyramid)

        conv1 = nn.Conv2d(in_channels=feature_pyramid.shape[1], out_channels=inputs[target_level].shape[1], kernel_size=1, stride=1, padding=0)
        conv1=conv1.to(self.device)
        output = conv1(feature_pyramid)


        return output
    
    def multihead_self_attention_chunked1(self, input_tensor, chunk_size):
        # 计算注意力权重
        b, c, h, w = input_tensor.size()
    
        # Create a single MultiheadAttention instance
        attention = nn.MultiheadAttention(c, self.num_heads).to(self.device)
    
        target = []
        for i in range(b):
            input = input_tensor[i]
        
            # Calculate the number of chunks needed
            num_chunks = h  // chunk_size
        
            # Process the input in chunks
            chunked_outputs = []
            for i in range(0, h, num_chunks):  # 只在高度维度划分
                chunked_input = input[ :, i:i+num_chunks, :]
        
                reshaped_chunk = chunked_input.permute(1, 2, 0).contiguous().view(num_chunks,  w, c)
        
                output_chunk, _ = attention(reshaped_chunk, reshaped_chunk, reshaped_chunk)
        
                restored_output = output_chunk.permute(2, 0, 1).contiguous().view(c, num_chunks, w)
                chunked_outputs.append(restored_output)
        
            # Concatenate the chunked outputs to form the final output
            reshaped_output = torch.cat(chunked_outputs, dim=1)
        
            # Reshape the output tensor back to [h, w, c]
            final_output = reshaped_output.view(h, w, c).permute(2, 0, 1)
            target.append(final_output)

        # Stack the reshaped outputs along the batch dimension
        stacked_outputs = torch.stack(target, dim=0)
        return stacked_outputs
    
    def multihead_self_attention_chunked(self, x, chunk_size):

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        #print( attention.size())
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = out + x
        return out

    
