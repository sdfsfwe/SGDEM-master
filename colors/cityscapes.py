import torch
from dataloader.definitions.labels_file import labels_cityscape_seg

# Extract the Cityscapes color scheme
TRID_TO_LABEL = labels_cityscape_seg.gettrainid2label() #用来获取Cityscapes数据集中训练ID到标签的映射。

COLOR_SCHEME_CITYSCAPES = torch.tensor(          #它创建了一个张量(tensor)，表示Cityscapes数据集的颜色方案
    tuple(
        TRID_TO_LABEL[tid].color if (tid in TRID_TO_LABEL) else (0, 0, 0)   #其中每个元素都是通过从TRID_TO_LABEL中获取相应训练ID对应的颜色，如果训练ID不存在，则使用(0, 0, 0)作为默认颜色。
        for tid in range(256)
    )
).float() / 255  #将颜色值归一化到[0, 1]的范围
