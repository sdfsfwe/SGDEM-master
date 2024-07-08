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



from collections import namedtuple
import numpy as np


Label = namedtuple( 'Label' , [  #定义了一个名为Label的命名元组（namedtuple），其中包含了标签的各种属性和信息

    'name'        , # 标签的标识符，用于唯一命名一个类别
    'id'          , # 与该标签相关联的整数ID，用于表示标签在真值图像中的标注

    'trainId'     , # 用于训练的ID。可以根据方法的需要自由修改这些ID。
                    # 使用预处理文件夹中提供的工具，使用训练ID创建真值图像。
                    # 但是，确保使用上述常规ID验证或提交结果给我们的评估服务器！对于trainId，可能有多个标签具有相同的ID。
                    # 然后，这些标签在真值图像中被映射到同一类别。对于逆映射，我们使用在下面的列表中首次定义的标签。
                    # 例如，在训练中将所有无效类别映射到相同的ID，对于某些方法可能是有意义的。最大值为255！
    'category'    , # 该标签所属的类别名称
    'categoryId'  , # 该类别的ID，用于在类别级别创建真值图像
    'hasInstances', # 该标签是否区分单个实例
    'ignoreInEval', # 是否忽略具有此类别作为真值标签的像素在评估中
    'color'       , # 标签的颜色
    ] )


class ClassDefinitions(object): #该类包含了用于处理分割掩模的类定义及其相关操作的过程
    """This class contains the classdefintions for the segmentation masks and the
    procedures to work with them"""

    def __init__(self, classlabels):
        self.labels = classlabels   #它接受一个classlabels参数，该参数是一个包含类标签的列表
        for i, label in zip(range(len(self.labels)), self.labels):
            if isinstance(label.color, int):
                self.labels[i] = label._replace(color=tuple([int(label.color/(256.0**2)) % 256,
                                    int(label.color/256.0) % 256,
                                    int(label.color) % 256]))   #通过循环遍历所有的标签，对其中的颜色属性进行处理，确保颜色以元组的形式存储。

    def getlabels(self):  #getlabels(self): 这个方法返回类定义中的所有标签
        return self.labels

    def getname2label(self):  #getname2label(self): 这个方法返回一个字典，将类名称映射到相应的标签对象
        name2label = {label.name: label for label in self.labels}
        return name2label

    def getid2label(self):  #getid2label(self): 这个方法返回一个字典，将类ID映射到相应的标签对象
        id2label = {label.id: label for label in self.labels}
        return id2label

    def gettrainid2label(self): #gettrainid2label(self): 这个方法返回一个字典，将训练ID映射到相应的标签对象。标签按照逆序排列
        trainid2label = {label.trainId: label for label in reversed(self.labels)}
        return trainid2label

    def getcategory2label(self): #getcategory2label(self): 这个方法返回一个字典，将类别名称映射到相应的标签对象列表。对于具有相同类别的标签，它们被分组在同一个列表中
        category2labels = {}
        for label in self.labels:
            category = label.category
            if category in category2labels:
                category2labels[category].append(label)
            else:
                category2labels[category] = [label]

    def assureSingleInstanceName(self,name): #assureSingleInstanceName(self, name): 这个方法用于确保给定的名称是表示单个实例的标签名称
        # if the name is known, it is not a group
        name2label = self.getname2label()
        if name in name2label: #如果给定的名称已经是一个已知的标签名称，则直接返回
            return name
        # test if the name actually denotes a group
        if not name.endswith("group"):
            return None
        # remove group
        name = name[:-len("group")] #如果名称以"group"结尾，则尝试将其截断，并检查截断后的名称是否存在并且表示具有实例的标签
        # test if the new name exists
        if not name in name2label:
            return None
        # test if the new name denotes a label that actually has instances
        if not name2label[name].hasInstances:
            return None
        # all good then
        return name


labels_cityscape_seg = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])


dataset_labels = {
    'cityscapes': labels_cityscape_seg,
}
