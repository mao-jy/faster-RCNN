import numpy as np
import tensorflow as tf

from detection.core.bbox import geometry, transforms
from detection.utils.misc import *


class ProposalTarget:
    """
    对建议框建立target，用于分类和回归任务

    Attributes
    ---
        target_means: [4] 用于偏移量归一化
        target_stds: [4] 用于偏移量归一化
        num_rcnn_deltas: rcnn偏移量个数，最大允许的框个数
        positive_fraction: 正例框的个数（前景框个数）
        pos_iou_thr: 建议框与某一GT框的IoU大于pos_iou_thr，则为正例框
        neg_iou_thr: 建议框与所有GT框的IoU都小于neg_iou_thr，则为负例框
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rcnn_deltas=256,
                 positive_fraction=0.25,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5):

        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rcnn_deltas = num_rcnn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
            
    def build_targets(self, proposals_list, gt_boxes, gt_class_ids, img_metas):
        """对于一个batch中的每张图片建立target
        
        Args
        ---
            proposals_list: 一个长为batch_size的列表，列表中的项为：[num_proposals, [y1, x1, y2, x2]]，坐标为小数形式，
                且对于不同图片，num_proposals不相等
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] 坐标为小数形式，零填充使得所有图片的gt_boxes维度相同
            gt_class_ids: [batch_size, num_gt_boxes] GT框类别标签，零填充使得所有图片的gt_boxes维度相同
            img_metas: [batch_size, 11]
            
        Returns
        ---
            rois_list: 一个长为batch_size的列表，列表中项为：[num_rois, [y1, x1, y2, x2]]，坐标为小数形式，
                且对于不同图片，num_proposals不相等
            rcnn_target_matchs_list: 一个长度为batch_size的列表，表项为：[num_rois,]
                前num_pos_rios个值是正例对应的GT框序号，后num_neg_rios个值是零填充
            rcnn_target_deltas_list: 一个长度为batch_size的列表，表项为：[num_positive_rois, [dy, dx, dh, dw]]
        """

        # [num_proposals, [height, width]]
        pad_shapes = calc_pad_shapes(img_metas)
        
        rois_list = []
        rcnn_target_matchs_list = []
        rcnn_target_deltas_list = []
        
        for i in range(img_metas.shape[0]):

            rois, target_matchs, target_deltas = self._build_single_target(
                proposals_list[i], gt_boxes[i], gt_class_ids[i], pad_shapes[i])
            rois_list.append(rois)
            rcnn_target_matchs_list.append(target_matchs)
            rcnn_target_deltas_list.append(target_deltas)
        
        return rois_list, rcnn_target_matchs_list, rcnn_target_deltas_list
    
    def _build_single_target(self, proposals, gt_boxes, gt_class_ids, img_shape):
        """在单张图片上建立target
        Args
        ---
            proposals: [num_proposals, (y1, x1, y2, x2)] 建议框坐标，坐标为小数形式
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)] GT框坐标
            gt_class_ids: [num_gt_boxes,] GT框类别号
            img_shape: [img_height, img_width] 图片尺寸
            
        Returns
        ---
            rois: [num_rois, [y1, x1, y2, x2]] 坐标为小数形式
            target_matchs: [num_rois,] 前num_pos_rios个值是正例对应的GT框序号，后num_neg_rios个值是零填充
            target_deltas: [num_positive_rois, [dy, dx, dh, dw]]
        """

        H, W = img_shape

        # 去除零填充框，框的个数 num_gt_boxes -> num_gt_boxes2
        gt_boxes, non_zeros = trim_zeros(gt_boxes)    # gt_boxes: [num_gt_boxes2, 4], non_zeros: [num_gt_boxes,]
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros)    # gt_calss_ids: [num_gt_boxes2,]

        # 将GT框坐标化为小数形式
        gt_boxes = gt_boxes / tf.constant([H, W, H, W], dtype=tf.float32)

        # 计算建议框和GT框的IoU矩阵
        overlaps = geometry.compute_overlaps(proposals, gt_boxes)

        # 根据pos_iou_thr和neg_iou_thr获取所有正例和负例坐标
        # 规则为：与任一GT框的IoU大于pos_iou_thr的建议框为正例，与所有GT框的IoU都小于neg_iou_thr的建议框为负例
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        positive_roi_bool = (roi_iou_max >= self.pos_iou_thr)
        positive_indices = tf.where(positive_roi_bool)[:, 0]   # 降维 [N, 1] =>[N,]
        negative_indices = tf.where(roi_iou_max < self.neg_iou_thr)[:, 0]

        # 确保正例个数和正、负比例
        # 规则为：
        #     正例个数不大于self.num_rcnn_deltas * self.positive_fraction
        #     正例 : 负例 = self.positive_fraction : (1 - self.positive_fraction)
        positive_count = int(self.num_rcnn_deltas * self.positive_fraction)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]

        r = 1.0 / self.positive_fraction
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)
        
        # 找出上面选择的正例对应的GT框
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment) # [34, 4]
        target_matchs = tf.gather(gt_class_ids, roi_gt_box_assignment) # [34]

        # 计算正例与GT框之间的偏移量
        target_deltas = transforms.bbox2delta(positive_rois, roi_gt_boxes, self.target_means, self.target_stds)

        # [正例框+负例框个数　* [y1, x1, y2, x2]]
        rois = tf.concat([positive_rois, negative_rois], axis=0)

        # 将target_matchs的长度填充到正例框+负例框个数，使用零填充
        N = tf.shape(negative_rois)[0]
        target_matchs = tf.pad(target_matchs, [(0, N)])

        target_matchs = tf.stop_gradient(target_matchs)
        target_deltas = tf.stop_gradient(target_deltas)

        return rois, target_matchs, target_deltas
