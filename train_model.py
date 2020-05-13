import os
import json
from pycocotools.cocoeval import COCOeval
import tensorflow as tf
from tensorflow import keras
import numpy as np

from detection.datasets import coco, data_generator
from detection.models.detectors import faster_rcnn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 可用的gpu块号
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # log等级

EPOCHS = 2
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
CHECKPOINT = ''  # 是否继续上次训练
OUTPUT_DIR = 'weights'  # 权重文件保存路径
SAVE_INTERVAL = 1  # 多少个batch保存一次权重文件
PRINT_INTERVAL = 1  # 多少个batch打印一次loss信息
VALIDATE_INTERVAL = 2  # 多少个batch进行一次验证

tf.random.set_seed(22)
np.random.seed(22)

# 数据处理
img_mean = (0., 0., 0.)
img_std = (1., 1., 1.)
train_dataset = coco.CocoDataSet('dataset', 'train',
                                 flip_ratio=0,
                                 pad_mode='fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1216))
val_dataset = coco.CocoDataSet('dataset', 'train',
                                 flip_ratio=0,
                                 pad_mode='non-fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1216))

train_generator = data_generator.DataGenerator(train_dataset, shuffle=False)
train_tf_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(BATCH_SIZE).prefetch(100)

# 模型设置
num_classes = len(train_dataset.get_categories())
model = faster_rcnn.FasterRCNN(num_classes=num_classes)
optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=0.9, nesterov=True)
if CHECKPOINT != '':
    batch_imgs, batch_metas, batch_bboxes, batch_labels = train_dataset[0]
    with tf.GradientTape() as tape:
        _, _, _, _ = model((np.array([batch_imgs]), np.array([batch_metas]),
                            np.array([batch_bboxes]), np.array([batch_labels])), training=True)

    model.load_weights(CHECKPOINT, by_name=True)

# 训练过程中的一些细节参数
max_num = -1
for filename in os.listdir(OUTPUT_DIR):
    if filename.startswith('checkpoint_'):
        words = filename.split('_')
        if int(words[1]) > max_num:
            max_num = int(words[1])  # 为了防止权重命名文件冲突
num_of_batch = np.ceil(len(train_dataset) / BATCH_SIZE)  # 整个数据集可以分成多少个batch

# 训练
for epoch in range(EPOCHS):
    for (batch, inputs) in enumerate(train_tf_dataset):
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = model(
                (batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)

            loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 每隔PRINT_INTERVAL个batch打印一次loss信息
        if (batch + 1) % PRINT_INTERVAL == 0 or (batch + 1) == num_of_batch:
            print('epoch: %d  batch: %d  total_loss: %.2f  rpn_class_loss: %.2f  rpn_bbox_loss: %.2f  '
                  'rcnn_class_loss: %.2f  rcnn_bbox_loss: %.2f' % (epoch, batch, loss_value.numpy(),
                                                                   rpn_class_loss.numpy(), rpn_bbox_loss.numpy(),
                                                                   rcnn_class_loss.numpy(), rcnn_bbox_loss.numpy()))

        # 每隔SAVE_INTERVAL个batch保存一次checkpoint
        if (batch + 1) % SAVE_INTERVAL == 0 or (batch + 1) == num_of_batch:
            save_path = os.path.join(OUTPUT_DIR, 'checkpoint_' + str(max_num + 1) + '_' +
                                     str(epoch) + '_' + str(batch) + '.h5')
            model.save_weights(save_path)

        # 每隔VALIDATE_INTERVAL计算一次模型在验证集上的mAP
        if (batch + 1) % VALIDATE_INTERVAL == 0 or (batch + 1) == num_of_batch:

            all_img_id = []
            all_detections = []
            for index in range(len(val_dataset)):
                print(index)
                img, img_meta, _, _ = val_dataset[index]
                proposals = model.simple_test_rpn(img, img_meta)
                detections = model.simple_test_bboxes(img, img_meta, proposals)

                img_id = val_dataset.img_ids[index]
                all_img_id.append(img_id)

                for pos in range(detections['rois'].shape[0]):
                    result = dict()
                    result['score'] = float(detections['scores'][pos])
                    result['category_id'] = val_dataset.label2cat[int(detections['class_ids'][pos])]
                    y1, x1, y2, x2 = [float(item) for item in list(detections['rois'][pos])]
                    result['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
                    result['image_id'] = img_id
                    all_detections.append(result)

            if not all_detections == []:
                json_path = os.path.join(OUTPUT_DIR, 'validate_' + str(max_num + 1) + '_' +
                                         str(epoch) + '_' + str(batch) + '.json')
                with open(json_path, 'w') as f:
                    f.write(json.dumps(all_detections))

                coco_dt = val_dataset.coco.loadRes(json_path)
                cocoEval = COCOeval(val_dataset.coco, coco_dt, 'bbox')
                cocoEval.params.imgIds = all_img_id

                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()

                content = 'fsfdss' \
                          'f'

