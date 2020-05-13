import tensorflow as tf
from    tensorflow.keras import layers

from detection.core.bbox import transforms
from detection.core.loss import losses
from detection.utils.misc import *


class BBoxHead(tf.keras.Model):
    def __init__(self, num_classes, 
                 pool_size=(7, 7),
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 min_confidence=0.7,
                 nms_threshold=0.3,             # nms阈值
                 max_instances=100,             # 最终应保留的框的个数上限
                 **kwags):
        super(BBoxHead, self).__init__(**kwags)
        
        self.num_classes = num_classes
        self.pool_size = tuple(pool_size)
        self.target_means = target_means
        self.target_stds = target_stds
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances
        
        self.rcnn_class_loss = losses.rcnn_class_loss
        self.rcnn_bbox_loss = losses.rcnn_bbox_loss

        # 第一个全连接层
        self.rcnn_class_conv1 = layers.Conv2D(1024, self.pool_size, 
                                              padding='valid', name='rcnn_class_conv1')
        self.rcnn_class_bn1 = layers.BatchNormalization(name='rcnn_class_bn1')

        # 第二个全连接层
        self.rcnn_class_conv2 = layers.Conv2D(1024, (1, 1), 
                                              name='rcnn_class_conv2')
        self.rcnn_class_bn2 = layers.BatchNormalization(name='rcnn_class_bn2')
        
        # 分类结果
        self.rcnn_class_logits = layers.Dense(num_classes, name='rcnn_class_logits')

        # 回归结果
        self.rcnn_delta_fc = layers.Dense(num_classes * 4, name='rcnn_bbox_fc')
        
    def call(self, inputs, training=True):
        """
        Args
        ---
            inputs: pooled_rois_list: 统一尺寸后的roi，一个长为batch_size的列表，　列表中的项为：
                [num_rois, pool_size[0], pool_size[1], channels]，不同图片num_rois不相同，channels为特征金字塔每一层的通道数
        
        Returns
        ---
            rcnn_class_logits_list: [batch_size, num_rois, self.num_classes] 不同图片的num_rois不同
            rcnn_probs_list: [batch_size, num_rois, self.num_classes] 不同图片的num_rois不同
            rcnn_deltas_list: [batch_size, num_rois, self.num_classes, [dy, dx, dh, dw]] 不同图片的num_rois不同
        """

        pooled_rois_list = inputs

        # [batch_size,] 记录每张图片的roi数
        num_pooled_rois_list = [pooled_rois.shape[0] for pooled_rois in pooled_rois_list]

        # [所有图片所有roi数, pool_size[0], pool_size[1], channels]
        # [图片0roi0, 图片0roi1, ..., 图片1roi0, 图片1roi1, ...]
        pooled_rois = tf.concat(pooled_rois_list, axis=0)

        # 第一次全连接　x: [所有图片所有roi数, 1, 1, 1024]
        x = self.rcnn_class_conv1(pooled_rois)
        x = self.rcnn_class_bn1(x, training=training)
        x = tf.nn.relu(x)

        # 第二次全连接 x: [所有图片所有roi数, 1, 1, 1024]
        x = self.rcnn_class_conv2(x)
        x = self.rcnn_class_bn2(x, training=training)
        x = tf.nn.relu(x)

        # x: [所有图片所有roi数, 1024]
        x = tf.squeeze(tf.squeeze(x, 2), 1)

        # 经过一次全连接后做softmax
        # logits, probs: [所有图片所有roi数, self.num_classes]
        logits = self.rcnn_class_logits(x)
        probs = tf.nn.softmax(logits)

        # 全连接后做reshape
        # deltas: [所有图片所有roi数, self.num_classes * 4]
        # deltas: [所有图片所有roi数, self.num_classes, 4]
        deltas = self.rcnn_delta_fc(x)
        deltas = tf.reshape(deltas, (-1, self.num_classes, 4))

        # 将多张图片得到的结果进行分离，参数维度见Returns
        rcnn_class_logits_list = tf.split(logits, num_pooled_rois_list, 0)
        rcnn_probs_list = tf.split(probs, num_pooled_rois_list, 0)
        rcnn_deltas_list = tf.split(deltas, num_pooled_rois_list, 0)
            
        return rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list

    def loss(self, 
             rcnn_class_logits_list, rcnn_deltas_list, 
             rcnn_target_matchs_list, rcnn_target_deltas_list):
        """计算rcnn损失

        Args
        ---
            rcnn_class_logits_list: [batch_size, num_rois, self.num_classes] 不同图片的num_rois不同
            rcnn_deltas_list: [batch_size, num_rois, self.num_classes, [dy, dx, dh, dw]] 不同图片的num_rois不同
            rcnn_target_matchs_list: 一个长度为batch_size的列表，表项为：[num_rois,]
                前num_pos_rios个值是正例对应的GT框序号，后num_neg_rios个值是零填充
            rcnn_target_deltas_list: 一个长度为batch_size的列表，表项为：[num_positive_rois, [dy, dx, dh, dw]]

        Returns
        ---
            rcnn_class_loss: rcnn分类损失
            rcnn_bbox_loss: rcnn回归损失
        """
        rcnn_class_loss = self.rcnn_class_loss(
            rcnn_target_matchs_list, rcnn_class_logits_list)
        rcnn_bbox_loss = self.rcnn_bbox_loss(
            rcnn_target_deltas_list, rcnn_target_matchs_list, rcnn_deltas_list)
        
        return rcnn_class_loss, rcnn_bbox_loss
        
    def get_bboxes(self, rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas):
        """获取一个batch中所有图片的最终检测框

        对一张图片的所有的rois进行筛选，步骤如下：
            1. 剔除背景框
            2. 对于每个类做nms
            3. top-k

        Args
        ---
            rcnn_probs_list: [batch_size, num_rois, num_classes] 不同图片的num_rois不同
            rcnn_deltas_list: [batch_size, num_rois, num_classes, [dy, dx, dh, dw]] 不同图片的num_rois不同
            rois_list: [batch_size , num_rois, [y1, x1, y2, x2]] 不同图片的num_rois不同
            img_meta: [batch_size, 11]
        
        Returns
        ---
            detections_list: [batch_size, 最终保留的框的个数, [y1, x1, y2, x2, class_id, score]]
                不同图片最终保留的框的个数不一定相等，种类包含背景
        """
        
        pad_shapes = calc_pad_shapes(img_metas)
        detections_list = [
            self._get_bboxes_single(
                rcnn_probs_list[i], rcnn_deltas_list[i], rois_list[i], pad_shapes[i])
            for i in range(img_metas.shape[0])
        ]
        return detections_list  
    
    def _get_bboxes_single(self, rcnn_probs, rcnn_deltas, rois, img_shape):
        """获取单张图片的最终检测框

        Args
        ---
            rcnn_probs: [num_rois, num_classes]
            rcnn_deltas: [num_rois, num_classes, (dy, dx, dh, dw)]
            rois: [num_rois, (y1, x1, y2, x2)]
            img_shape: [img_height, img_width]

        Returns
        ---
            detections: [最终保留的框的个数, [y1, x1, y2, x2, class_id, score]]
        """
        H, W = img_shape   

        # [num_rois,] 每个roi最可能属于的类
        class_ids = tf.argmax(rcnn_probs, axis=1, output_type=tf.int32)

        # [0, 1, 2, ..., num_rois-1]和class_ids进行拼接
        # indices: [num_rios, 2]
        indices = tf.stack([tf.range(rcnn_probs.shape[0]), class_ids], axis=1)

        # 得到最高类得分和对应的偏移量
        class_scores = tf.gather_nd(rcnn_probs, indices)   # [num_rois,]
        deltas_specific = tf.gather_nd(rcnn_deltas, indices)    # [num_rois, [dy, dx, dh, dw]]

        # [num_rois, [y1, x1, y2, x2]] 对roi进行微调
        refined_rois = transforms.delta2bbox(rois, deltas_specific, self.target_means, self.target_stds)
        
        # 将框坐标化为非归一化形式，并将超出图片的部分进行裁剪
        refined_rois *= tf.constant([H, W, H, W], dtype=tf.float32)
        window = tf.constant([0., 0., H * 1., W * 1.], dtype=tf.float32)
        refined_rois = transforms.bbox_clip(refined_rois, window)   # [num_rois, [y1, x1, y2, x2]]

        # 对所有的rois进行筛选，步骤如下：
        #   1. 剔除背景框
        #   2. 对于每个类做nms
        #   3. top-k

        # 1
        # 非背景框的序号
        keep = tf.where(class_ids > 0)[:, 0]

        if self.min_confidence:

            # 置信度>=min_confidence的框的序号
            conf_keep = tf.where(class_scores >= self.min_confidence)[:, 0]
            keep = tf.compat.v2.sets.intersection(tf.expand_dims(keep, 0),
                                                  tf.expand_dims(conf_keep, 0))

            # [num_rois2,] 去除背景框和小置信度框后剩余的框序号
            keep = tf.sparse.to_dense(keep)[0]

        pre_nms_class_ids = tf.gather(class_ids, keep)  # [num_rois2,]
        pre_nms_scores = tf.gather(class_scores, keep)  # [num_rois2,]
        pre_nms_rois = tf.gather(refined_rois,   keep)  # [num_rois2, [y1, x1, y2, x2]]
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]  # 去掉重复的类别标签

        def nms_keep_map(class_id):
            """对于每一类框进行nms，返回这一类下应该保留的框在class_ids, class_scores, class_rois中的序号"""

            # 找出该类别的位置
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]

            # class_keep中保存着应该保留的框在ixs中的位置
            class_keep = tf.image.non_max_suppression(
                    tf.gather(pre_nms_rois, ixs),           # [属于该类的框, [y1, x1, y2, x2]]
                    tf.gather(pre_nms_scores, ixs),         # [属于该类的框,] 得分
                    max_output_size=self.max_instances,
                    iou_threshold=self.nms_threshold)

            # 找出应该保留的框在class_ids, class_scores, class_rois中的序号
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))

            return class_keep

        # 2
        # nms_keep: 经过nms后应该保留的框的序号
        nms_keep = []
        for i in range(unique_pre_nms_class_ids.shape[0]):
            nms_keep.append(nms_keep_map(unique_pre_nms_class_ids[i]))
        if len(nms_keep) != 0:
            nms_keep = tf.concat(nms_keep, axis=0)
        else:
            nms_keep = tf.zeros([0, ], tf.int64)

        keep = tf.compat.v2.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(nms_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

        # 3.
        # 对每个框的置信度进行top-k，使得最终保留的框的个数不超过max_instances
        roi_count = self.max_instances
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)  

        # [最终保留的框的个数, [y1, x1, y2, x2, class_id, score]]
        detections = tf.concat([
            tf.gather(refined_rois, keep),                  # [最终保留的框的个数, [y1, x1, y2, x2]]
            tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],   # [最终保留的框的个数,1]
            tf.gather(class_scores, keep)[..., tf.newaxis]      # [最终保留的框的个数, 1]
            ], axis=1)
        
        return detections
