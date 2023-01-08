from pathlib import Path

import tensorflow as tf

from data import gen_dataset
from model import UNETR


def main():
    model = UNETR(feature_size=32, output_channels=14)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    ds_train, ds_val = gen_dataset(Path("RawData"), bs=6, img_size=(96, 96, 96))
    optimizer = tf.keras.optimizers.Adam(0.0001)

    model.compile(optimizer, loss)

    cb = [tf.keras.callbacks.ModelCheckpoint("UNETR/best_model", save_best_only=True, save_weights_only=True)]
    model.fit(ds_train, epochs=30000, steps_per_epoch=100, validation_data=ds_val, callbacks=cb)


if __name__ == "__main__":
    main()
