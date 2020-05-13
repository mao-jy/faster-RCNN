import tensorflow as tf


def trim_zeros(boxes, name=None):
    """去除零填充框（四个坐标全是0的框）
    
    Args
    ---
        boxes: [N * [y1, x1, y2, x2]] 框坐标

    Returns
    ---
        boxes: [N2 * [y1, x1, y2, x2]] N2为去除四个坐标全0的框后剩余的框的数量
        non_zeros: [N,] 四个坐标全0的框标记为False，否则为True
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def parse_image_meta(meta):
    """解析图片信息
    
    Args
    ---
        meta: [..., 11]

    Returns
    ---
        图片信息字典
        {
            'ori_shape': [..., 3],
            'img_shape': [..., 3],
            'pad_shape': [...m 3],
            ‘scale': [...,],
            'flip': [...,]
        }
    """
    meta = meta.numpy()
    ori_shape = meta[..., 0:3]
    img_shape = meta[..., 3:6]
    pad_shape = meta[..., 6:9]
    scale = meta[..., 9]  
    flip = meta[..., 10]
    return {
        'ori_shape': ori_shape,
        'img_shape': img_shape,
        'pad_shape': pad_shape,
        'scale': scale,
        'flip': flip
    }


def calc_batch_padded_shape(meta):
    """将这个batch中所有图片的高、宽的最大值作为零填充目标返回
    Args
    ---
        meta: [batch_size, 11]
    
    Returns
    ---
        [height, width] 这个batch中的所有图片都应该零填充到这个高、宽
    """
    return tf.cast(tf.reduce_max(meta[:, 6:8], axis=0), tf.int32).numpy()


def calc_img_shapes(meta):
    """计算图片进行缩放后的尺寸
    Args
    ---
        meta: [..., 11]
    
    Returns
    ---
        [..., [height, width]]
    """
    return tf.cast(meta[..., 3:5], tf.int32).numpy()


def calc_pad_shapes(meta):
    """获取填充后的图片的尺寸

    Args
    ---
        meta: [N, [ori_shape(3), img_shape(3), pad_shape(3), scale_factor(1), flip(1)]] 图片信息
    
    Returns
    ---
        [N, [height, width]]
    """
    return tf.cast(meta[..., 6:8], tf.int32).numpy()
