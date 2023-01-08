from functools import partial
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tensorflow as tf


def resample_spacing(image, new_spacing=[1, 1, 1]):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(new_spacing)

    orig_size = np.array(image.GetSize(), dtype=int)
    orig_spacing = image.GetSpacing()
    new_size = orig_size * (orig_spacing / np.array(new_spacing))
    new_size = np.ceil(new_size).astype(int)  #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    newimage = resample.Execute(image)
    return newimage


def read_image(image_path, label_path):
    image = sitk.ReadImage(image_path.decode("utf-8"))
    image = resample_spacing(image)

    label = sitk.ReadImage(label_path.decode("utf-8"))
    label = resample_spacing(label)

    return sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(label)


def augment(image, label):
    # TODO YONIGO: use rng
    if tf.random.uniform([]) > 0.9:
        image = tf.reverse(image, [0])
        label = tf.reverse(label, [0])
    if tf.random.uniform([]) > 0.9:
        image = tf.reverse(image, [1])
        label = tf.reverse(label, [1])
    if tf.random.uniform([]) > 0.9:
        image = tf.reverse(image, [2])
        label = tf.reverse(label, [2])
    return image, label


def random_crop(image, label, img_size, rng):
    seed = rng.make_seeds(2)[0]
    image = tf.image.stateless_random_crop(image, size=img_size, seed=seed)
    label = tf.image.stateless_random_crop(label, size=img_size, seed=seed)
    return image, label


def normalize(image, label, min_hu=-1000.0, max_hu=1000.0):
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    image = tf.clip_by_value(image, min_hu, max_hu)
    image = (image - min_hu) / (max_hu - min_hu)
    image = tf.expand_dims(image, -1)  # channel last
    return image, label


def ds_base(ds, bs, img_size, shuffle=False, do_augment=False):

    if shuffle:
        ds = ds.shuffle(len(ds))
    ds = ds.map(lambda x, y: tf.numpy_function(read_image, inp=[x, y], Tout=(tf.int16, tf.uint8)))
    ds = ds.cache()  # TODO YONIGO: cache to file for large dataset
    rng = tf.random.Generator.from_seed(123, alg="philox")  # TODO YONIGO: get seed randomly?
    ds = ds.map(partial(random_crop, img_size=img_size, rng=rng))
    ds = ds.map(normalize)
    if do_augment:
        ds = ds.map(augment)
    ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.batch(bs)
    return ds


def gen_dataset(raw_data, bs, img_size):

    labels = list((raw_data / "Training/label").glob("*.gz"))
    labels = sorted(labels)

    train_labels = labels[:-6]
    train_images = [(raw_data / "Training/img/" / f"img{i.name[5:]}").as_posix() for i in train_labels]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, list(map(Path.as_posix, train_labels))))
    train_ds = ds_base(train_ds, bs, img_size, shuffle=True, do_augment=True)
    train_ds = train_ds.repeat()

    val_labels = labels[-6:]
    val_images = [(raw_data / "Training/img/" / f"img{i.name[5:]}").as_posix() for i in val_labels]
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, list(map(Path.as_posix, val_labels))))
    val_ds = ds_base(val_ds, bs, img_size)

    return train_ds, val_ds


def gen_full_images_ds(raw_data, bs):
    labels = list((raw_data / "Training/label").glob("*.gz"))
    labels = sorted(labels)

    val_labels = labels[-6:]
    val_images = [(raw_data / "Training/img/" / f"img{i.name[5:]}").as_posix() for i in val_labels]
    ds = tf.data.Dataset.from_tensor_slices((val_images, list(map(Path.as_posix, val_labels))))
    ds = ds.map(lambda x, y: tf.numpy_function(read_image, inp=[x, y], Tout=(tf.int16, tf.uint8)))
    ds = ds.map(normalize)
    ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.batch(bs)
    return ds
