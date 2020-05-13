import tensorflow as tf
from tensorflow import keras


def smooth_l1_loss(y_true, y_pred):
    """Smooth L1损失
    
    Args
    ---
        y_true: [N, [dy, dx, dh, dw]] 真实值
        y_pred: [N, [dy, dx, dh, dw] 预测值

    Returns
    ---
        loss: smooth_l1损失值
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

    return loss


def rpn_class_loss(target_matchs, rpn_class_logits):
    """RPN分类损失

    Args
    ---
        target_matchs: [batch_size, num_anchors] 给出每张图片中的每个建议框的标记，正例为1，负例为-1，中立为0
        rpn_class_logits: [batch_size, num_anchors, 2] 每张图片中所有建议框为前景或背景的概率

    Returns
    ---
        loss: rpn分类损失
    """
    # 整体效果是：把1, 0, -1中的0去掉，再将1转为0
    anchor_class = tf.cast(tf.equal(target_matchs, 1), tf.int32)
    indices = tf.where(tf.not_equal(target_matchs, 0))
    anchor_class = tf.gather_nd(anchor_class, indices)

    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)

    num_classes = rpn_class_logits.shape[-1]

    # anchors_class: [正例+反例数量,]　正例为1，负例为0
    # rpn_class_logits: [正例+反例数量, 2]
    loss = keras.losses.categorical_crossentropy(tf.one_hot(anchor_class, depth=num_classes),
                                                 rpn_class_logits, from_logits=True)

    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)

    return loss


def rpn_bbox_loss(target_deltas, target_matchs, rpn_deltas):
    """计算RPN回归损失

    Args
    ---
        target_deltas: [batch, num_rpn_deltas, (dy, dx, dh, dw)] 所有图片的num_rpn_deltas相同，不足的部分使用零填充
        target_matchs: [batch, anchors] 建议框的标签，正例为1，负例为-1，中立为0
        rpn_deltas: [batch, anchors, [dy, dx, dh, dw]]

    Returns
    ---
        loss: rpn回归损失
    """
    def batch_pack(x, counts, num_rows):
        outputs = []
        for i in range(num_rows):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)

    # rpn_deltas: [所有图片的所有正例, [y1, x1, y2, x2]] 筛选出正例的偏移量(由rpn计算得来)
    indices = tf.where(tf.equal(target_matchs, 1))
    rpn_deltas = tf.gather_nd(rpn_deltas, indices)

    # [batch_size,] 记录每张图片中正例的数量
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32), axis=1)

    # [所有图片的所有正例, [dy, dx, dh, dw]] 将每张图片中的正例偏移量放到一起
    # [图片0正例0的偏移量, 图片0正例1的偏移量, ..., 图片1正例0的偏移量, 图片1正例1的偏移量, ...]将每张图片中的正例偏移量放到一起
    target_deltas = batch_pack(target_deltas, batch_counts, target_deltas.shape.as_list()[0])

    loss = smooth_l1_loss(target_deltas, rpn_deltas)
    
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    
    return loss


def rcnn_class_loss(target_matchs_list, rcnn_class_logits_list):
    """rcnn分类损失
    
    Args
    ---
        target_matchs_list: 一个长度为batch_size的列表，表项为：[num_rois,]
                前num_pos_rios个值是正例对应的GT框序号，后num_neg_rios个值是零填充
        rcnn_class_logits_list: [batch_size, num_rois, num_classes] 不同图片的num_rois不同

    Returns
    ---
        loss: rcnn分类损失值
    """
    class_ids = tf.cast(tf.concat(target_matchs_list, 0), 'int64')     # [所有图片的所有roi,]
    class_logits = tf.concat(rcnn_class_logits_list, 0)    # [所有图片的所有roi, num_classes]
    num_classes = class_logits.shape[-1]

    # 交叉熵损失
    loss = keras.losses.categorical_crossentropy(tf.one_hot(class_ids, depth=num_classes),
                                                 class_logits, from_logits=True)

    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)

    return loss


def rcnn_bbox_loss(target_deltas_list, target_matchs_list, rcnn_deltas_list):
    """rcnn回归损失
    
    Args
    ---
        target_deltas_list: 一个长度为batch_size的列表，表项为：[num_positive_rois, [dy, dx, dh, dw]]
        target_matchs_list: 一个长度为batch_size的列表，表项为：[num_rois,]
                前num_pos_rios个值是正例对应的GT框序号，后num_neg_rios个值是零填充
        rcnn_deltas_list: [batch_size, num_rois, num_classes, [dy, dx, dh, dw]] 不同图片的num_rois不同

    Returns
    ---
        rcnn回归损失值
    """
    
    target_deltas = tf.concat(target_deltas_list, 0)    # [所有图片的正例框个数, [dy, dx, dh, dw]]
    target_class_ids = tf.concat(target_matchs_list, 0)    # [所有图片的所有roi,]
    rcnn_deltas = tf.concat(rcnn_deltas_list, 0)    # [所有图片的所有roi, num_classes, [dy, dx, dh, dw]]

    # indices: [所有图片的所有正例框, 2] 存放这些框在所有框中的位置及其对应的类别号
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]  # [所有图片的所有正例框,] 正例框在target_class_ids中的序号
    positive_roi_class_ids = tf.cast(                   # [所有图片的所有正例框,] 正例框对应的类别号
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    
    # [所有图片的所有正例框, [dy, dx, dh, dw]] 所有正例框的预测偏移量
    rcnn_deltas = tf.gather_nd(rcnn_deltas, indices)

    loss = smooth_l1_loss(target_deltas, rcnn_deltas)
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)

    return loss
