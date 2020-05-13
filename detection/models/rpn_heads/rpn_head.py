from tensorflow.keras import layers

from detection.core.bbox import transforms
from detection.utils.misc import *

from detection.core.anchor import anchor_generator, anchor_target
from detection.core.loss import losses


class RPNHead(tf.keras.Model):
    """
    Attributes:
        anchor_scales=(32, 64, 128, 256, 512),      # 建议框基数
        anchor_ratios=(0.5, 1, 2),                  # 建议框长宽比
        anchor_feature_strides=(4, 8, 16, 32, 64),  # 特征金字塔各层和原图的比例
        proposal_count=2000,                        # NMS后需要保留的建议框个数，top-k
        nms_threshold=0.7,                          # NMS阈值
        target_means=(0., 0., 0., 0.),              # dy, dx, dw, dh的均值，用于归一化
        target_stds=(0.1, 0.1, 0.2, 0.2),           # dy, dx, dw, dh的标准差，用于归一化
        num_rpn_deltas=256,                         # 从所有建议框中最终应该获得num_rpn_deltas用于训练rpn网络
        positive_fraction=0.5,                      # 用于训练rpn网络的建议框中的正例所占比例
        pos_iou_thr=0.7,                            # 建议框与任一GT框的IoU大于pos_iou_thr则标记为正例
        neg_iou_thr=0.3,                            # 建议框与所有GT框的IoU小于neg_iou_thr则标记为反例
    """

    def __init__(self,
                 anchor_scales=(32, 64, 128, 256, 512),
                 anchor_ratios=(0.5, 1, 2),
                 anchor_feature_strides=(4, 8, 16, 32, 64),
                 proposal_count=2000,
                 nms_threshold=0.7,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 **kwags):

        super(RPNHead, self).__init__(**kwags)

        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.target_means = target_means
        self.target_stds = target_stds

        self.generator = anchor_generator.AnchorGenerator(
            scales=anchor_scales,
            ratios=anchor_ratios,
            feature_strides=anchor_feature_strides)

        self.anchor_target = anchor_target.AnchorTarget(
            target_means=target_means,
            target_stds=target_stds,
            num_rpn_deltas=num_rpn_deltas,
            positive_fraction=positive_fraction,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr)

        self.rpn_class_loss = losses.rpn_class_loss
        self.rpn_bbox_loss = losses.rpn_bbox_loss

        # 特征金字塔经过一个3*3的卷积
        self.rpn_conv_shared = layers.Conv2D(512, (3, 3), padding='same',
                                             kernel_initializer='he_normal',
                                             name='rpn_conv_shared')

        # 卷积代替全连接进行分类
        self.rpn_class_raw = layers.Conv2D(len(anchor_ratios) * 2, (1, 1),
                                           kernel_initializer='he_normal',
                                           name='rpn_class_raw')

        # 卷积代替全连接进行回归
        self.rpn_delta_pred = layers.Conv2D(len(anchor_ratios) * 4, (1, 1),
                                            kernel_initializer='he_normal',
                                            name='rpn_bbox_pred')

    def call(self, inputs):
        """对于输入进来的特征金字塔，经过网络计算，输出分类结果和回归结果

        Args
        ---
            inputs: [5, batch_size, feat_map_height, feat_map_width, channels]
            5表示特征金字塔的5层，顺序为[P2, P3, P4, P5, P6]

        Returns
        ---
            rpn_class_logits: [batch_size, num_anchors, 2] 背景和前景的概率
            rpn_probs: [batch_size, num_anchors, 2]
            rpn_deltas: [batch_size, num_anchors, 4]
            num_anchors为单张图片特征金字塔所有层上的所有点对应的所有框的数量
        """
        layer_outputs = []

        # feat: [batch_size, h, w, channels] 对每一层进行单独处理，h, w分别表示当前层的高度和宽度
        for feat in inputs:
            # shared: [batch_size, h, w, 512] 对特征层进行3*3卷积
            shared = self.rpn_conv_shared(feat)
            shared = tf.nn.relu(shared)

            # 1*1卷积，结果用于分类
            x = self.rpn_class_raw(shared)  # [batch_size, h, w, len(anchor_ratios) * 2]
            rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])  # [batch_size, h * w * len(anchor_ratios), 2]
            rpn_probs = tf.nn.softmax(rpn_class_logits)

            # 1*1卷积，结果用于回归
            x = self.rpn_delta_pred(shared)  # [batch_size, h, w, len(anchor_ratios) * 4]
            rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])  # [batch_size, h * w * len(anchor_ratios), 4]

            layer_outputs.append([rpn_class_logits, rpn_probs, rpn_deltas])

        # layer_outputs.shape[0] = 5, layer_output.shape[1] = 3
        # outputs.shape[0] = 3, outputs.shape[1] = 5
        outputs = list(zip(*layer_outputs))
        outputs = [tf.concat(list(o), axis=1) for o in outputs]
        rpn_class_logits, rpn_probs, rpn_deltas = outputs

        # rpn_class_logits: [batch_size, 5层的h * w * len(anchor_ratios), 2]
        # rpn_probs: [batch_size, 5层的h * w * len(anchor_ratios), 2]
        # rpn_deltas: [batch_size, 5层的h * w * len(anchor_ratios), 4]
        return rpn_class_logits, rpn_probs, rpn_deltas

    def loss(self, rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas):
        """计算rpn loss

        Args
        ---
            rpn_class_logits: [batch_size, num_anchors, 2] 每张图片中所有建议框为背景或前景的概率
            rpn_deltas: [batch_size, num_anchors, [dy, dx, dh, dw]]
            gt_boxes: [batch_size, num_gt_boxes, [y1, x1, y2, x2]]
            gt_class_ids: [batch_size, num_gt_boxes] GT框的类别标签
            img_metas: [batch_size, 11] 图片信息

        Returns
        ---

        """

        # anchors: [统一尺寸的金字塔上提取出来的所有建议框, [y1, x1, y2, x2]]　anchors被此batch中所有图片共用
        # valid_flags: [batch_size, 统一尺寸的金字塔上提取出来的所有建议框的数量] anchors是否合法（框中心是否在零填充处）
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)

        # rpn_target_matchs: [batch_size, num_anchors] 给出每张图片中的每个建议框的标记，正例为1，负例为-1，中立为0
        # rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, dh, dw)] 所有图片中正例的偏移量，负例对应的为零填充
        rpn_target_matchs, rpn_target_deltas = self.anchor_target.build_targets(
            anchors, valid_flags, gt_boxes, gt_class_ids)

        # rpn_target_matchs 比较初始的建议框和GT框，进行一系列筛选后，对所有的框进行标记（正例为1，负例为-1，中立为0）
        # rpn_class_logits 特征图经过网络计算出来的每一个框是正例或者负例的概率
        rpn_class_loss = self.rpn_class_loss(
            rpn_target_matchs, rpn_class_logits)

        # rpn_target_matchs 比较初始的建议框和GT框，进行一系列筛选后，对所有的框进行标记（正例为1，负例为-1，中立为0）
        # rpn_target_deltas 通过比较初始建议框中的正例框和GT框，给出这些正例框应有的偏移量
        # rpn_deltas 特征图经过网络计算出来的每一个框的偏移量
        rpn_bbox_loss = self.rpn_bbox_loss(
            rpn_target_deltas, rpn_target_matchs, rpn_deltas)

        return rpn_class_loss, rpn_bbox_loss

    def get_proposals(self,
                      rpn_probs,
                      rpn_deltas,
                      img_metas,
                      with_probs=False):
        """
        根据预测信息得出候选框
        对这个batch中的每一张图片的所有建议框：
            去除非法框
            限制框的最大数量，保留前景概率更大的框
            根据rpn_deltas进行微调
            裁去超出图片范围的部分
            坐标值化为小数形式
            做NMS

        Args
        ---
            rpn_probs: [batch_size, num_anchors, (bg prob, fg prob)] 预测概率（背景和前景）
            rpn_deltas: [batch_size, num_anchors, (dy, dx, dh, dw)] 预测偏移量
            img_metas: [batch_size, 11] 图片信息
            with_probs: bool.

        Returns
        ---
            proposals_list: [batch_size , num_proposals, [y1, x1, y2, x2]] if with_probs=False
                [batch_size , num_proposals, [y1, x1, y2, x2, 相应的前景概率]] if with_probs=True
                坐标以小数形式给出
                不同图片的num_proposals不同，上述写法只是为了方便理解
        """

        # anchors: [统一尺寸的金字塔上提取出来的所有建议框的数量 * [y1, x1, y2, x2]] anchors被此batch中所有图片共用
        # valid_flags: [batch_size * 统一尺寸的金字塔上提取出来的所有建议框的数量] anchors是否合法（框中心是否在零填充区域）
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)

        # [batch_size, num_anchors] 每张图片上每个建议框是前景的概率
        rpn_probs = rpn_probs[:, :, 1]

        # [batch, 2] 填充后的高、宽
        pad_shapes = calc_pad_shapes(img_metas)

        proposals_list = [
            self._get_proposals_single(
                rpn_probs[i], rpn_deltas[i], anchors, valid_flags[i], pad_shapes[i], with_probs)
            for i in range(img_metas.shape[0])
        ]

        return proposals_list

    def _get_proposals_single(self,
                              rpn_probs,
                              rpn_deltas,
                              anchors,
                              valid_flags,
                              img_shape,
                              with_probs):
        """
        Calculate proposals.

        Args
        ---
            rpn_probs: [num_anchors] 每张图片上每个建议框是前景的概率
            rpn_deltas: [num_anchors, (dy, dx, dh, dw)] 只有正例，不足的用零填充
            anchors: [统一尺寸的金字塔上提取出来的所有建议框的数量 * [y1, x1, y2, x2]] anchors被此batch中所有图片共用
            valid_flags: [统一尺寸的金字塔上提取出来的所有建议框的数量] anchors标记
            img_shape: (img_height, img_width)
            with_probs: bool.

        Returns
        ---
            proposals: [num_anchors4, [y1, x1, y2, x2]] if with_probs = False
                [num_anchors4, [y1, x1, y2, x2, 相应的rpn_prob]] if with_probs = True
                其中坐标为小数形式
        """

        H, W = img_shape

        valid_flags = tf.cast(valid_flags, tf.bool)

        # 将valid_flags为0的框都删去，num_anchors -> num_anchors2(去除非法框后剩余的框的数量)
        # rpn_probs: [num_anchors2]
        # rpn_deltas: [num_anchors2, (dy, dx, dh, dw)]
        # anchors: [num_anchors2 * [y1, x1, y2, x2]]
        rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
        rpn_deltas = tf.boolean_mask(rpn_deltas, valid_flags)
        anchors = tf.boolean_mask(anchors, valid_flags)

        # 通过rpn_probs削减的框的数量，保留是前景概率更大的框
        # num_anchors2 -> num_anchors3
        pre_nms_limit = min(6000, anchors.shape[0])
        ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=True).indices
        rpn_probs = tf.gather(rpn_probs, ix)
        rpn_deltas = tf.gather(rpn_deltas, ix)
        anchors = tf.gather(anchors, ix)

        # 根据偏移量预测出的偏移量微调建议框
        proposals = transforms.delta2bbox(anchors, rpn_deltas,
                                          self.target_means, self.target_stds)

        # 将建议框超出图片范围的部分裁去
        window = tf.constant([0., 0., H, W], dtype=tf.float32)
        proposals = transforms.bbox_clip(proposals, window)

        # 建议框归一化
        proposals = proposals / tf.constant([H, W, H, W], dtype=tf.float32)

        # NMS
        # num_anchors3 -> num_anchors4
        indices = tf.image.non_max_suppression(
            proposals, rpn_probs, self.proposal_count, self.nms_threshold)
        proposals = tf.gather(proposals, indices)

        if with_probs:
            proposal_probs = tf.expand_dims(tf.gather(rpn_probs, indices), axis=1)
            proposals = tf.concat([proposals, proposal_probs], axis=1)

        return proposals
