import tensorflow as tf
from detection.core.bbox import geometry, transforms
from detection.utils.misc import trim_zeros


class AnchorTarget:
    """计算分类和回归任务的target

    Attributes
    ---
        target_means: [4]. Bounding box refinement mean for RPN.
        target_stds: [4]. Bounding box refinement standard deviation for RPN.
        num_rpn_deltas: 从所有的建议框中选择num_rpn_deltas个框用于分类和回归任务，
            正例框优先，但比例不得多于positive_fraction，不足的用负例填充
        positive_fraction: 正例所占比例
        pos_iou_thr: 建议框与任一GT框的IoU大于pos_iou_thr，则标记为1
        neg_iou_thr: 建议狂与所有GT框的IoU小于neg_iou_thr，则标记为-1
    """
    def __init__(self,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rpn_deltas = num_rpn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr

    def build_targets(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        """
        判断每张图片中的每个建议框中是正例还是负例，以及偏移量

        Args:
            anchors: [num_anchors, (y1, x1, y2, x2)] 将当前batch里的所有图片zero_padding到同一尺寸，导致了所有图片有着相同的
                特征金字塔维度，导致了提取出来的num_anchors是相同的
            valid_flags: [batch_size, num_anchors] 标记每张图片的每个建议框是否合法（框的中心是否在原始图片内，而非零填充区域内）
                合法为1，不合法为0
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)]
            gt_class_ids: [batch_size, num_gt_boxex]

        Returns:
            rpn_target_matchs: [batch_size, num_anchors] 标记每张图片中的每个建议框为正例或者负例，正例为1，负例为-1，中立为0
            rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, dh, dw)] 所有图片中正例的偏移量，负例对应的为零填充
        """
        rpn_target_matchs = []
        rpn_target_deltas = []
        
        num_imgs = gt_class_ids.shape[0]
        for i in range(num_imgs):
            target_match, target_delta = self._build_single_target(
                anchors, valid_flags[i], gt_boxes[i], gt_class_ids[i])
            rpn_target_matchs.append(target_match)
            rpn_target_deltas.append(target_delta)
        
        rpn_target_matchs = tf.stack(rpn_target_matchs)
        rpn_target_deltas = tf.stack(rpn_target_deltas)
        
        rpn_target_matchs = tf.stop_gradient(rpn_target_matchs)
        rpn_target_deltas = tf.stop_gradient(rpn_target_deltas)
        
        return rpn_target_matchs, rpn_target_deltas

    def _build_single_target(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        """
        获取单张图片中的每个建议框是正例还是负例，以及偏移量

        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] 当前图片中所有的建议框
            valid_flags: [num_anchors] 当前图片中每个框是否合法
            gt_class_ids: [num_gt_boxes]
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
        
        Returns
        ---
            target_matchs: [num_anchors]
            target_deltas: [num_rpn_deltas, (dy, dx, dh, dw)]
        """
        # 去除那些四个坐标值都是0的框
        gt_boxes, _ = trim_zeros(gt_boxes)

        # overlaps: [num_anchors, num_gt_boxes] IoU矩阵
        overlaps = geometry.compute_overlaps(anchors, gt_boxes)

        # 1. 获取建议框标记
        # 标记为1：与每个GT框IoU最大的建议框标记为1；与某一GT框的IoU大于pos_iou_thr的建议框标记为1
        # 标记为-1：与所有的GT框的IoU都小于neg_iou_thr的框标记为-1
        # 标记为0：剩余的框标记为0
        # 超越上述的规则：非法的框标记为0

        anchor_iou_argmax = tf.argmax(overlaps, axis=1)     # [num_of_anchors,] 每个建议框与哪个GT框IoU最大
        anchor_iou_max = tf.reduce_max(overlaps, axis=[1])  # [num_of_anchors,] 获取相应的IoU数值

        # 所有GT框的IoU都小于neg_iou_thr的框，标记为-1，否则标记为0
        target_matchs = tf.where(anchor_iou_max < self.neg_iou_thr,
                                 -tf.ones(anchors.shape[0], dtype=tf.int32), tf.zeros(anchors.shape[0], dtype=tf.int32))

        # 在之前的基础上将非法的框都标记为0
        target_matchs = tf.where(tf.equal(valid_flags, 1),
                                 target_matchs, tf.zeros(anchors.shape[0], dtype=tf.int32))

        # 在之前的基础上，与某GT框的IoU大于pos_iou_thr的框，标记为1
        target_matchs = tf.where(anchor_iou_max >= self.pos_iou_thr, 
                                 tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

        # [num_gt_boxes] 得到与每个GT框IoU最大的建议框，并将其标记为1
        gt_iou_argmax = tf.argmax(overlaps, axis=0)
        target_matchs = tf.compat.v1.scatter_update(tf.Variable(target_matchs), gt_iou_argmax, 1)

        # 2. 将框的数量减少至self.num_rpn_deltas
        # 若正例框数量大于self.num_rpn_deltas * self.positive_fraction，则用随机选取的方式将多余的框标记为0
        # 用负例框填补，使得：正例框数量 + 负例框数量 = self.num_rpn_deltas，删除多余负例框的方式仍然是随机选取

        # 只需要self.num_rpn_deltas * self.positive_fraction个正例框，如果多出来了，则将多余的删去（随机删除）
        ids = tf.where(tf.equal(target_matchs, 1))
        ids = tf.squeeze(ids, 1)
        extra = ids.shape.as_list()[0] - int(self.num_rpn_deltas * self.positive_fraction)
        if extra > 0:
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)

        # 无论正例框的数量是否能达到self.num_rpn_deltas * self.positive_fraction，不足的部分都用负例框填补
        # 使得框数量总共为self.num_rpn_deltas
        ids = tf.where(tf.equal(target_matchs, -1))
        ids = tf.squeeze(ids, 1)
        extra = ids.shape.as_list()[0] - (self.num_rpn_deltas -
            tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32)))
        if extra > 0:
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)

        # 3. 对于所有正例，生成偏移量

        # 取出标记为1的anchors存放在a中
        ids = tf.where(tf.equal(target_matchs, 1))
        a = tf.gather_nd(anchors, ids)

        # 取出对应的GT框
        anchor_idx = tf.gather_nd(anchor_iou_argmax, ids)
        gt = tf.gather(gt_boxes, anchor_idx)

        # 计算[正例框数量 * [dy, dx, dh, dw]]
        target_deltas = transforms.bbox2delta(a, gt, self.target_means, self.target_stds)

        # 应该返回self.num_rpn_deltas个target_deltas，不足的部分进行零填充
        padding = tf.maximum(self.num_rpn_deltas - tf.shape(target_deltas)[0], 0)
        target_deltas = tf.pad(target_deltas, [(0, padding), (0, 0)])

        return target_matchs, target_deltas
