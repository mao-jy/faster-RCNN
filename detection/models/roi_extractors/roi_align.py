from detection.utils.misc import *


class PyramidROIAlign(tf.keras.layers.Layer):
    """ROI Align

    Attributes
    ---
        pool_shape: [height, width] 输出的高、宽
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)

        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        """对于一个batch中的所有图片的所有roi进行roi align
        Args
        ---
            inputs: (rois_list, feature_map_list, img_metas)
                rois_list: 一个长为batch_size的列表，列表中项为：[num_rois, [y1, x1, y2, x2]]，坐标为小数形式，
                    且对于不同图片，num_proposals不相等
                feature_map_list: [4, batch_size, feat_map_height, feat_map_width, channels]　4:[P2, P3, P4, P5]
                img_metas: [batch_size, 11]

        Returns
        ---
            pooled_rois_list: 统一尺寸后的roi，一个长为batch_size的列表，　列表中的项为：
                [num_rois, self.pool_shape[0], self.pool_shape[1], 特征图通道数]，不同图片的num_rois不相同
        """
        rois_list, feature_map_list, img_metas = inputs

        pad_shapes = calc_pad_shapes(img_metas)  # [batch_size, 2]
        pad_areas = pad_shapes[:, 0] * pad_shapes[:, 1]  # [batch_size]

        num_rois_list = [rois.shape.as_list()[0] for rois in rois_list]  # [batch_size] 记录了每张图片中roi的个数

        # [图片0中的roi个数 * 0, 图片1中的roi个数 * 1, ...]
        roi_indices = tf.constant(
            [i for i in range(len(rois_list))
             for _ in range(rois_list[i].shape.as_list()[0])],
            dtype=tf.int32
        )

        # [图片0的面积 * 图片0中的roi个数, 图片1的面积 * 图片1中的roi个数, ...]
        areas = tf.constant(
            [pad_areas[i] for i in range(pad_areas.shape[0]) for _ in range(num_rois_list[i])],
            dtype=tf.float32
        )

        # [当前batch中的所有图片的roi个数, [y1, x1, y2, x2]]
        rois = tf.concat(rois_list, axis=0)

        y1, x1, y2, x2 = tf.split(rois, 4, axis=1)
        h = y2 - y1
        w = x2 - x1

        # 根据roi的面积大小确定应该从特征金字塔的哪一层为其提取特征
        # roi_level = int(4 + log (2) (根号下rio面积 / 224 / 根号下原图面积)) 再将roi_level限制在{2, 3, 4, 5}内
        # 因为roi的坐标是小数形式，所以这里需要根号下原图面积这一项
        # roi_level的形式为: [图片0rio0属于的层, 图片0rio1属于的层, ...图片1rio0属于的层, 图片1rio1属于的层, ...]
        roi_level = tf.math.log(
            tf.sqrt(tf.squeeze(h * w, 1))
            / tf.cast((224.0 / tf.sqrt(areas * 1.0)), tf.float32)
        ) / tf.math.log(2.0)
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))

        pooled_rois = []
        roi_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_rois = tf.gather_nd(rois, ix)
            level_roi_indices = tf.gather_nd(roi_indices, ix)
            roi_to_level.append(ix)

            level_rois = tf.stop_gradient(level_rois)
            level_roi_indices = tf.stop_gradient(level_roi_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_rois, pool_height, pool_width, channels]

            # feature_map_list[i]: [batch_size, feat_height, feat_width, channels]
            # level_rois: [所有图片中被分第i层的rio, [y1, x1, y2, x2]]
            # level_roi_indices: [所有图片中被分第i层的rio,] level_rois中的每个roi属于哪张图片
            pooled_rois.append(tf.image.crop_and_resize(
                feature_map_list[i], level_rois, level_roi_indices, self.pool_shape,
                method="bilinear"))

        # pooled_rios: [4(P2, P3, P4, P5),　此batch的所有图片中被分配给该层的rio数,
        #               self.pool_shape[0], self.pool_shape[1], 特征图通道数]
        # pooled_rios: [所有图片的所有rio数, pool_height, pool_width, 特征图通道数]
        pooled_rois = tf.concat(pooled_rois, axis=0)

        # roi_to_level记录了pooled_rois中的每一个rio在roi_level中对应的位置
        roi_to_level = tf.concat(roi_to_level, axis=0)  # [所有图片的所有rio数,]
        roi_range = tf.expand_dims(tf.range(tf.shape(roi_to_level)[0]), 1)  # [[0], [1], ..., [所有图片的所有rio数-1]]
        roi_to_level = tf.concat([tf.cast(roi_to_level, tf.int32), roi_range], axis=1)  # [所有图片的所有rio数, 2]

        # 这两行代码的作用实质上是将roi_to_level中的元素按照[:, 0]升序排列
        sorting_tensor = roi_to_level[:, 0] * 100000 + roi_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(roi_to_level)[0]).indices[::-1]

        # [所有图片的所有rio数,]　此时ix中的内容指示了roi_level中的元素在pool_rois中对应的位置
        ix = tf.gather(roi_to_level[:, 1], ix)
        pooled_rois = tf.gather(pooled_rois, ix)

        # num_rois_list: [batch_size,] 存放了每个图片有多少个roi
        # pool_rios: [所有图片的所有rio数, self.pool_shape[0], self.pool_shape[1], 特征图通道数]　存放了池化后的roi
        pooled_rois_list = tf.split(pooled_rois, num_rois_list, axis=0)

        # pooled_rois_list: 一个长为batch_size的列表，　列表中的项为：
        #   [num_rois, self.pool_shape[0], self.pool_shape[1], 特征图通道数]，不同图片的num_rois不相同
        return pooled_rois_list
