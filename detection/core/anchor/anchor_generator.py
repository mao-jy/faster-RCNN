import tensorflow as tf
from detection.utils.misc import calc_img_shapes, calc_batch_padded_shape


class AnchorGenerator:

    # 对于一组图片，零填充到尺寸，使得特征金字塔尺寸完全相同，初步的建议框完全相同（这里的建议框是在原图上的坐标信息）
    # 然后根据每一张图片的高、宽检测，这些建议框对本图片是否合法（建议框的中心是否在图片内部，而不是零填充区域）
    # 将在本类中获取的框叫做建议框

    def __init__(self,
                 scales=(32, 64, 128, 256, 512),
                 ratios=(0.5, 1, 2),
                 feature_strides=(4, 8, 16, 32, 64)):

        # 要生成的建议框的大小基数，这个scale是针对原图的尺寸，在特征金字塔的不同层上选用不同尺寸，金字塔越上层的scale越大
        self.scales = scales
        self.ratios = ratios  # 生成的建议框的长宽比
        self.feature_strides = feature_strides  # feature map和原图的比例关系

    def generate_pyramid_anchors(self, img_metas):
        """生成建议框

        Args
        ---
            img_metas: [图片数量(batch_size) , [ori_shape(3), img_shape(3), pad_shape(3), scale_factor(1), flip(1)]]

        Returns
        ---
            anchors: [统一尺寸的金字塔上提取出来的所有建议框 , [y1, x1, y2, x2]] anchors被此batch中所有图片共用
            valid_flags: [batch_size , 统一尺寸的金字塔上提取出来的所有建议框的数量] anchors是否合法（框中心是否在零填充处）
        """
        # 获取所有图片中宽和高的最大值，将所有图片都零填充至[max_height, max_width]
        # 所有图片的宽高相同，则特征金字塔维度完全相同，则初步的建议框完全相同
        pad_shape = calc_batch_padded_shape(img_metas)
        feature_shapes = [(pad_shape[0] // stride, pad_shape[1] // stride)
                          for stride in self.feature_strides]
        anchors = [
            self._generate_level_anchors(level, feature_shape)
            for level, feature_shape in enumerate(feature_shapes)
        ]
        # anchors: [金字塔所有层上提出来的建议框 * [y1, x1, y2, x2]]
        anchors = tf.concat(anchors, axis=0)

        # img_shapes： [图片数量 * [height, width]]
        img_shapes = calc_img_shapes(img_metas)
        valid_flags = [
            self._generate_valid_flags(anchors, img_shapes[i])
            for i in range(img_shapes.shape[0])
        ]
        valid_flags = tf.stack(valid_flags, axis=0)

        anchors = tf.stop_gradient(anchors)
        valid_flags = tf.stop_gradient(valid_flags)

        return anchors, valid_flags

    def _generate_valid_flags(self, anchors, img_shape):

        # 移除零填充区域的建议框
        # anchors: 当前图片的上的所有建议框, 当前图片的[height, width]

        y_center = (anchors[:, 2] + anchors[:, 0]) / 2
        x_center = (anchors[:, 3] + anchors[:, 1]) / 2

        valid_flags = tf.ones(anchors.shape[0], dtype=tf.int32)
        zeros = tf.zeros(anchors.shape[0], dtype=tf.int32)

        valid_flags = tf.where(y_center <= img_shape[0], valid_flags, zeros)
        valid_flags = tf.where(x_center <= img_shape[1], valid_flags, zeros)

        # len(valid_flags)=len(anchors)
        # 如果这个anchor的中心点在图片范围内，则相应标记位置1；否则置0（此时中心店在领填充区域内）
        return valid_flags

    def _generate_level_anchors(self, level, feature_shape):

        # 获取特征金字塔某一层生成的所有box
        # level: 当前层数，其中P2: level=0, P3: level=1, P4: level=2, P5: level=3, P4: level=5

        scale = self.scales[level]  # 要生成的建议框的大小基数，这个scale是针对原图的尺寸，金字塔越上层的scale越大
        ratios = self.ratios  # 生成的建议框的宽高比
        feature_stride = self.feature_strides[level]  # feature map和原图的比例关系

        scales, ratios = tf.meshgrid([float(scale)], ratios)
        scales = tf.reshape(scales, [-1])
        ratios = tf.reshape(ratios, [-1])

        heights = scales / tf.sqrt(ratios)
        widths = scales * tf.sqrt(ratios)

        shifts_y = tf.multiply(tf.range(feature_shape[0]), feature_stride)
        shifts_x = tf.multiply(tf.range(feature_shape[1]), feature_stride)

        shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
        shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

        box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

        box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), (-1, 2))
        box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), (-1, 2))

        boxes = tf.concat([box_centers - 0.5 * box_sizes,
                           box_centers + 0.5 * box_sizes], axis=1)

        # boxes: [num_of_box * [y1, x1, y2, x2]]
        # boxes: [num_of_box * [center_y-0.5*height, center_x-0.5*width, center_y+0.5*height, center_x+0.5*width]]
        return boxes
