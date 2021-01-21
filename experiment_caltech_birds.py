import os
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


@tf.function
def true_label_preprocess(image, label):
    return (image, tf.expand_dims(label, 1)), label


@tf.function
def crop_box_ratio(image, bbox):
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    image = tf.image.crop_to_bounding_box(image, tf.cast(bbox[0] * height, tf.int32), tf.cast(bbox[1] * width, tf.int32), tf.cast((bbox[2] - bbox[0]) * height, tf.int32), tf.cast((bbox[3] - bbox[1]) * width, tf.int32))
    return image


@tf.function
def random_crop(image, minval=0.8):
    image_size = tf.shape(image)
    if image_size.shape == 3:
        image_height = image_size[0]
        image_width = image_size[1]
        image = tf.image.random_crop(image,
                                     [tf.cast(tf.random.uniform([], minval=minval) * tf.cast(image_height, tf.float32), tf.int32),
                                      tf.cast(tf.random.uniform([], minval=minval) * tf.cast(image_width, tf.float32), tf.int32), 3])

    if image_size.shape == 4:
        batchsize = image_size[0]
        image_height = image_size[1]
        image_width = image_size[2]
        image = tf.image.random_crop(image,
                                     [batchsize, tf.cast(tf.random.uniform([], minval=minval) * tf.cast(image_height, tf.float32), tf.int32),
                                      tf.cast(tf.random.uniform([], minval=minval) * tf.cast(image_width, tf.float32), tf.int32), 3])
    return image


@tf.function
def preprocess_caltech_image(features, img_size=tf.constant(224)):
    image = features['image']
    bbox = features['bbox']
    label = features['label']

    image = crop_box_ratio(image, bbox)
    image = tf.image.resize_with_pad(image, img_size, img_size)
    return image, label


@tf.function
def picture_augment(image, img_size=tf.constant(224)):
    import tensorflow_addons as tfa

    image = tf.image.adjust_brightness(image, tf.random.uniform([], minval=0.5, maxval=1))
    image = tf.image.adjust_hue(image, tf.random.uniform([], minval=-0.01, maxval=0.01))
    image = tf.image.adjust_saturation(image, tf.random.uniform([], minval=0.9, maxval=1.1))
    image = tf.image.random_flip_left_right(image)

    image = tfa.image.rotate(image, tf.constant(np.pi / 8) * tf.random.uniform([], minval=-1, maxval=1), name='rotate')
    image = random_crop(image)
    image = tf.image.resize_with_pad(image, img_size, img_size)

    return image


def create_caltechbirds2011_dataset(data_dir, BATCH_SIZE=32, n_shuffle=1000):
    (train_ds, test_ds), info = tfds.load('caltech_birds2011', split=['train', 'test'], shuffle_files=True, data_dir=data_dir, with_info=True)

    train_batches = train_ds.map(lambda x: preprocess_caltech_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(n_shuffle).batch(BATCH_SIZE).map(lambda x, y: (picture_augment(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).map(true_label_preprocess)
    test_batches = test_ds.map(lambda x: preprocess_caltech_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(BATCH_SIZE).map(true_label_preprocess)

    num_classes = info.features['label'].num_classes
    return train_batches, test_batches, num_classes


def create_model(modelname, num_classes, img_size=224):
    from CustomLayer import CosineLayer

    feature_extractor_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", name='efficientnetB0')
    feature_extractor_layer.trainable = True

    input_image = tf.keras.Input(shape=(img_size, img_size, 3), dtype=tf.float32, name='input_image')
    efficientnet_output = feature_extractor_layer(input_image)

    y_true = tf.keras.Input(shape=1, dtype=tf.int64, name='true_label')

    if modelname == 'AdaCos':
        from CustomLayer import AdaCos_logits

        cos_layer = CosineLayer(num_classes=num_classes)
        cos_layer_output = cos_layer(efficientnet_output)

        logits = AdaCos_logits()([cos_layer_output, y_true])

        model = tf.keras.models.Model(inputs=(input_image, y_true), outputs=tf.keras.layers.Softmax()(logits))
        return model

    elif modelname == 'fixedAdaCos':
        from CustomLayer import CorrectCosMean

        cos_layer = CosineLayer(num_classes=num_classes)
        cos_layer_output = cos_layer(efficientnet_output)

        logits = CorrectCosMean()([cos_layer_output, y_true])

        fixed_s = tf.constant(tf.math.sqrt(2.) * tf.math.log(tf.cast(num_classes - 1, tf.float32)), name='fixed_s')

        model = tf.keras.models.Model(inputs=[input_image, y_true], outputs=tf.keras.layers.Softmax()(fixed_s * logits))
        return model

    if modelname == 'ArcFace':
        from CustomLayer import ArcFace_logits

        cos_layer = CosineLayer(num_classes=num_classes)
        cos_layer_output = cos_layer(efficientnet_output)

        logits = ArcFace_logits()([cos_layer_output, y_true])

        model = tf.keras.models.Model(inputs=[input_image, y_true], outputs=tf.keras.layers.Softmax()(logits))
        return model

    elif modelname == 'l2-softmax':
        alpha = 16
        normalize_output = tf.math.l2_normalize(efficientnet_output)

        top_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

        model = tf.keras.models.Model(inputs=[input_image, y_true], outputs=top_layer(alpha * normalize_output))
        return model

    elif modelname == 'softmax':
        top_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
        model = tf.keras.models.Model(inputs=[input_image, y_true], outputs=top_layer(efficientnet_output))

        return model


def train(model, train_batches, test_batches, epochs, logdir):
    import re
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logdir), profile_batch='800,900')

    checkpoint_prefix = logdir / Path("ckpt_{epoch}")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(str(checkpoint_prefix), save_weights_only=True, verbose=1)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())

    initial_epoch = 0
    ckpt = tf.train.latest_checkpoint(logdir, latest_filename=None)
    if ckpt:
        model.load_weights(ckpt)
        initial_epoch = int(re.sub(r".*_", "", ckpt))

    history = model.fit(train_batches,
                        initial_epoch=initial_epoch,
                        epochs=epochs,
                        validation_data=test_batches,
                        callbacks=[tensorboard_callback, cp_callback])

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelname", choices=["AdaCos", "ArcFace", "fixedAdaCos", "l2-softmax", "softmax"], default="AdaCos")
    args = parser.parse_args()
    modelname = args.modelname

    train_batches, test_batches, num_classes = create_caltechbirds2011_dataset(data_dir=Path("tmp"))
    model = create_model(modelname, num_classes)

    model.summary()

    _ = train(model, train_batches, test_batches, epochs=200, logdir=Path("tflog") / Path(modelname))


if __name__ == '__main__':
    main()
