import os.path as osp
import cv2
import numpy as np
from pycocotools.coco import COCO

from detection.datasets import transforms, utils


class CocoDataSet(object):

    def __init__(self, dataset_dir, subset,
                 flip_ratio=0,
                 pad_mode='fixed',
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 scale=(1024, 800),
                 debug=False):
        """加载coco数据集

        Attributes
        ---
            dataset_dir: 数据集根目录
            subset: 'train'或者'val'
            flip_ratio: Float. 按照flip_ratio为概率决定图片是否进行左右翻转
            pad_mode: 使用那种pad模式(fixed, non-fixed)，如果是fixed则会填充至scale中的较大值，
                否则以高、宽都达到64的倍数为目的进行最小填充
            mean: [3] 均值，用于归一化
            std: [3] 标准差，用于归一化
            scale: [2] 用于决定放缩大小
        """
        if subset not in ['train', 'val', 'test']:
            raise AssertionError('subset must be "train" or "val" or "test".')

        self.coco = COCO("{}/{}/{}.json".format(dataset_dir, subset, subset))
        self.image_dir = "{}/{}/images".format(dataset_dir, subset)  # 图片路径

        self.flip_ratio = flip_ratio

        if pad_mode in ['fixed', 'non-fixed']:
            self.pad_mode = pad_mode
        elif subset == 'train':  # 对于训练集，进行固定填充，即将图片宽、高都填充至固定值（正方形）
            self.pad_mode = 'fixed'
        else:  # 对于非训练集，进行非固定填充，即讲图片宽、高都填充至一个固定值的整数倍
            self.pad_mode = 'non-fixed'

        self.cat_ids = self.coco.getCatIds()  # 标注文件中的所有类别号
        self.cat2label = {  # 标注文件中的类别编号：新的类别编号
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.label2cat = {  # 新的类别编号：标注文件中的类别编号
            i + 1: cat_id
            for i, cat_id in enumerate(self.cat_ids)
        }

        self.img_ids, self.img_infos = self._filter_imgs()  # 图片编号以及相应的标签信息

        if debug:
            self.img_ids, self.img_infos = self.img_ids[:50], self.img_infos[:50]

        self.img_transform = transforms.ImageTransform(scale, mean, std, pad_mode)  # 图片预处理类
        self.bbox_transform = transforms.BboxTransform()  # 边界框相应处理类

    def _filter_imgs(self, min_size=32):
        """过滤掉以下图片：尺寸太小，没有标注

        Args
        ---
            min_size: 图片尺寸的最小值

        Returns
        ---
            img_ids: list 保留下来的图片序号
            img_infos: [dict] 字典的key包括id, file_name, height, width
        """
        # 从所有的标注中提取所有的图片id
        all_img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))

        img_ids = []
        img_infos = []
        for i in all_img_ids:
            info = self.coco.loadImgs(i)[0]  # 某张图片信息的字典，key包括id, file_name, height, width

            ann_ids = self.coco.getAnnIds(imgIds=i)
            ann_info = self.coco.loadAnns(ann_ids)
            ann = self._parse_ann_info(ann_info)

            if min(info['width'], info['height']) >= min_size and ann['labels'].shape[0] != 0:
                img_ids.append(i)
                img_infos.append(info)
        return img_ids, img_infos

    def _load_ann_info(self, idx):
        """获取某张图片的所有标注

        Args
        ---
            idx: 第几张图片

        Returns
        ---
            anno_info: list(dict) 该图片的原始标注信息（coco格式）
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)

        return ann_info

    def _parse_ann_info(self, ann_info):
        """处理标注数据

        Args
        ---
            ann_info: list[dict] coco形式的标注数据

        Returns
        ---
            dict: dict 字典的key为bboxes, labels, bboxes_ignore
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        return ann

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        """根据下标号返回图片和标注信息

        Args
        ---
            idx: 图片下标

        Returns
        ---
            img: [height, width, channels]
            img_meta: [11]
            bboxes: [num_boxes, 4]
            labels: [num_boxes]
        """
        # img_infos: dict dict为coco格式的图片信息
        # ann_info: [dict] dict为coco格式的标注信息
        img_info = self.img_infos[idx]
        ann_info = self._load_ann_info(idx)

        # cv2加载BGR图片，转换成RGB
        img = cv2.imread(osp.join(self.image_dir, img_info['file_name']), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ori_shape = img.shape

        # ann: dict 字典的key为bboxes, labels, bboxes_ignore
        ann = self._parse_ann_info(ann_info)
        bboxes = ann['bboxes']
        labels = ann['labels']

        flip = True if np.random.rand() < self.flip_ratio else False  # 按概率随机决定是否翻转

        # 图片和边界框的预处理
        img, img_shape, scale_factor = self.img_transform(img, flip)
        bboxes = self.bbox_transform(bboxes, img_shape, scale_factor, flip)

        # 整合图片信息
        img_meta_dict = dict({
            'ori_shape': ori_shape,  # 图片原始尺寸
            'img_shape': img_shape,  # 图片缩放后的尺寸
            'pad_shape': img.shape,  # 图片缩放+填充后的尺寸
            'scale_factor': scale_factor,  # 缩放因子，缩放后尺寸 = int(缩放前尺寸*scale_factor + 0.5)
            'flip': flip  # 是否进行左右翻转
        })

        img_meta = utils.compose_image_meta(img_meta_dict)

        return img, img_meta, bboxes, labels

    def get_categories(self):
        """获取所有种类名

        Returns
        ---
            ['bg', cat1, cat2, ...] bg表示背景
        """
        return ['bg'] + [self.coco.loadCats(i)[0]["name"] for i in self.cat2label.keys()]
