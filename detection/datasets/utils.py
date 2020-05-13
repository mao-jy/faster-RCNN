import cv2
import numpy as np

###########################################
#
# Utility Functions for 
# Image Preprocessing and Data Augmentation
#
###########################################


def img_flip(img):
    """图片左右翻转
    
    Args
    ---
        img: [height, width, channel]
    
    Returns
    ---
        [height, width, channel]
    """
    return np.fliplr(img)


def bbox_flip(bboxes, img_shape):
    """将边界框进行左右翻转
    
    Args
    ---
        bboxes: [..., 4]
        img_shape: [height, width]
    
    Returns
    ---
        flipped: [..., 4] 左右翻转后的边界框
    """
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 1] = w - bboxes[..., 3] - 1
    flipped[..., 3] = w - bboxes[..., 1] - 1

    return flipped


def impad_to_square(img, pad_size):
    """在单张图片右下角进行零填充，使得img.shape[0] = img.shape[1] = pad_size
    
    Args
    ---
        img: [height, width, channels] 待填充图片
        pad_size: int
    
    Returns
    ---
        pad: [pad_size, pad_size, channels] 填充后的图片
    """
    shape = (pad_size, pad_size, img.shape[-1])
    
    pad = np.zeros(shape, dtype=img.dtype)
    pad[:img.shape[0], :img.shape[1], ...] = img

    return pad


def impad_to_multiple(img, divisor):
    """在单张图片右下角进行零填充，使得图片长、宽能被divisor整除
    
    Args
    ---
        img: [height, width, channels] 待填充图片
        divisor: Int
    
    Returns
    ---
        pad: 填充好的图片
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    shape = (pad_h, pad_w, img.shape[-1])
    
    pad = np.zeros(shape, dtype=img.dtype)
    pad[:img.shape[0], :img.shape[1], ...] = img

    return pad


def imrescale(img, scale):
    """进行等比缩放，保证图片长边小于等于scale中较大值，图片短边小于等于scale中较小值，且缩放比例尽可能小
    
    Args
    ---
        img: [height, width, channels] 待缩放图片
        scale: [2]
    
    Returns
    ---
        rescaled_img: [height, width, channels] 缩放后的图片
        scale_factor: 缩放因子，缩放后尺寸 = int(缩放前尺寸*scale_factor + 0.5)
    """
    h, w = img.shape[:2]
    
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))
    
    new_size = (int(w * float(scale_factor) + 0.5),
                int(h * float(scale_factor) + 0.5))

    rescaled_img = cv2.resize(
        img, new_size, interpolation=cv2.INTER_LINEAR)
    
    return rescaled_img, scale_factor


def imnormalize(img, mean, std):
    """图片归一化
    
    Args
    ---
        img: [height, width, channel]
        mean: [3] 归一化后的均值
        std: [3] 归一化后的标准差
    
    Returns
    ---
        [height, width, channel] 归一化后的图片
    """
    img = (img - mean) / std    
    return img.astype(np.float32)


def imdenormalize(norm_img, mean, std):
    """逆归一化
    
    Args
    ---
        norm_img: [height, width, channel]
        mean: [3] 归一化均值
        std: [3] 归一化标准差
    
    Returns
    ---
        [height, width, channel] 逆归一化后的图片
    """
    img = norm_img * std + mean
    return img.astype(np.float32)

#######################################
#
# Utility Functions for Data Formatting
#
#######################################


def get_original_image(img, img_meta, mean=(0, 0, 0), std=(1, 1, 1)):
    """复原为原始图片
    
    Args
    ---
        img: [height, width, channels] 待复原图片
        img_meta: [11] 图片信息
        mean: [3] 均值，用于逆归一化
        std: [3] 标准差，用于逆归一化
    
    Returns
    ---
        img: [ori_height, ori_witdth, channels] 复原后的图片
    """
    img_meta_dict = parse_image_meta(img_meta)
    ori_shape = img_meta_dict['ori_shape']
    img_shape = img_meta_dict['img_shape']
    flip = img_meta_dict['flip']
    
    img = img[:img_shape[0], :img_shape[1]]                 # 去除填充部分
    if flip:                                                # 水平翻转
        img = img_flip(img)
    img = cv2.resize(img, (ori_shape[1], ori_shape[0]),     # 缩放为原始尺寸
                     interpolation=cv2.INTER_LINEAR)
    img = imdenormalize(img, mean, std)                     # 逆归一化

    return img


def compose_image_meta(img_meta_dict):
    """图片信息：字典 -> 列表

    Args
    ---
        img_meta_dict: dict 字典形式的图片信息

    Returns
    ---
        img_meta: [11]
    """
    ori_shape = img_meta_dict['ori_shape']
    img_shape = img_meta_dict['img_shape']
    pad_shape = img_meta_dict['pad_shape']
    scale_factor = img_meta_dict['scale_factor']
    flip = 1 if img_meta_dict['flip'] else 0

    img_meta = np.array(
        ori_shape +               # size=3
        img_shape +               # size=3
        pad_shape +               # size=3
        tuple([scale_factor]) +   # size=1
        tuple([flip])             # size=1
    ).astype(np.float32)

    return img_meta


def parse_image_meta(img_meta):
    """图片信息：列表 -> 字典

    Args
    ---
        img_meta: [11]

    Returns
    ---
        dict{
            'ori_shape': [3] 图片原始尺寸,
            'img_shape': [3] 图片缩放后尺寸,
            'pad_shape': [3]图 片缩放+填充后尺寸,
            'scale_factor': float 缩放后尺寸 = int(缩放前尺寸*scale_factor + 0.5)
            'flip': bool 是否进行过左右翻转
        }
    """
    ori_shape = img_meta[0:3]
    img_shape = img_meta[3:6]
    pad_shape = img_meta[6:9]
    scale_factor = img_meta[9]
    flip = img_meta[10]

    return {
        'ori_shape': ori_shape.astype(np.int32),
        'img_shape': img_shape.astype(np.int32),
        'pad_shape': pad_shape.astype(np.int32),
        'scale_factor': scale_factor.astype(np.float32),
        'flip': flip.astype(np.bool),
    }
