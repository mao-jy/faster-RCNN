from detection.datasets.utils import *


class ImageTransform(object):
    """预处理图片
        1. 尺寸放缩
        2. 归一化
        3. 翻转
        4. 填充
    """
    def __init__(self,
                 scale=(800, 1333),
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 pad_mode='fixed'):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.pad_mode = pad_mode

        self.impad_size = max(scale) if pad_mode == 'fixed' else 64

    def __call__(self, img, flip=False):

        img, scale_factor = imrescale(img, self.scale)          # 等比缩放
        img_shape = img.shape
        img = imnormalize(img, self.mean, self.std)             # 归一化
          
        if flip:
            img = img_flip(img)                                 # 左右翻转

        if self.pad_mode == 'fixed':    # 在图片右下角进行零填充，使得img.shape[0] = img.shape[1] = pad_size
            img = impad_to_square(img, self.impad_size)
        else:                           # 在图片右下角进行零填充，使得图片长、宽能被ipad_size整除
            img = impad_to_multiple(img, self.impad_size)
        
        return img, img_shape, scale_factor


class BboxTransform(object):
    """边界框预处理
        1. 根据图片缩放后的尺寸进行边界框缩放
        2. 根据图片是否翻转进行决定边界框是否翻转
    """
    def __init__(self):
        pass
    
    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
 
        bboxes = bboxes * scale_factor                                     # 按比例放缩bbox

        if flip:
            bboxes = bbox_flip(bboxes, img_shape)                          # 将bbox进行左右翻转
            
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[0] - 1)    # 将bbox超出图片边界的部分裁去
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[1] - 1)
            
        return bboxes
