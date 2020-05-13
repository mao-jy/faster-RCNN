import tensorflow as tf


def compute_overlaps(boxes1, boxes2):
    """计算boxes1中的每个框和boxes2中的每个框的IoU

    Args:
        boxes1: [N1, (y1, x1, y2, x2)]
        boxes2: [N2, (y1, x1, y2, x2)]

    Returns:
        overlaps: [N1, N2]
            overlaps[m][n]表示boxes1中的第m个框和boxes2中的第n个框的IoU
    """
    # b1: [N1 * N2, (y1, x1, y2, x2)]
    # b2: [N1 * N2, (y1, x1, y2, x2)]
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    # 计算并区域面积
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    # 计算交区域面积
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    # 计算IoU
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

    return overlaps
