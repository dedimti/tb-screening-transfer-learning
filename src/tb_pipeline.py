"""
tb_pipeline.py
Reusable functions for the TB chest X-ray transfer-learning experiments.

Authors: Dedi Irawan & Sudarmaji, Universitas Muhammadiyah Metro.
License: MIT (see LICENSE).

These functions mirror the logic in the Colab notebooks so the pipeline can be
imported and run programmatically. The decision threshold is always selected on
the VALIDATION set (never the test set) to avoid data leakage.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import (
    ResNet50, EfficientNetB0, MobileNetV2, DenseNet121, VGG16,
    resnet50, efficientnet, mobilenet_v2, densenet, vgg16)
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_recall_fscore_support, f1_score)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD, EPOCHS_FT = 8, 12
LR_HEAD, LR_FT = 1e-3, 1e-5
VAL_SPLIT, TEST_SPLIT = 0.15, 0.15

# (builder, preprocessing function) for each supported architecture
MODELS = {
    "ResNet50":       (ResNet50,       resnet50.preprocess_input),
    "EfficientNetB0": (EfficientNetB0, efficientnet.preprocess_input),
    "MobileNetV2":    (MobileNetV2,    mobilenet_v2.preprocess_input),
    "DenseNet121":    (DenseNet121,    densenet.preprocess_input),
    "VGG16":          (VGG16,          vgg16.preprocess_input),
}


def build_datasets(data_dir, preprocess, seed):
    """Create train/val/test tf.data pipelines from a directory of
    {Normal, Tuberculosis} subfolders. The seed controls the split + shuffle."""
    full = tf.keras.utils.image_dataset_from_directory(
        data_dir, labels="inferred", label_mode="binary",
        class_names=["Normal", "Tuberculosis"], image_size=IMG_SIZE,
        batch_size=None, shuffle=True, seed=seed)
    n = full.cardinality().numpy()
    n_test, n_val = int(n * TEST_SPLIT), int(n * VAL_SPLIT)
    test_ds = full.take(n_test)
    rest = full.skip(n_test)
    val_ds = rest.take(n_val)
    train_ds = rest.skip(n_val)

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
        layers.RandomZoom(0.10), layers.RandomContrast(0.10)])
    AT = tf.data.AUTOTUNE

    def prep(img, lab, train):
        img = preprocess(tf.cast(img, tf.float32))
        if train:
            img = aug(img, training=True)
        return img, lab

    train_ds = train_ds.map(lambda x, y: prep(x, y, True), AT).batch(BATCH_SIZE).prefetch(AT)
    val_ds = val_ds.map(lambda x, y: prep(x, y, False), AT).batch(BATCH_SIZE).prefetch(AT)
    test_ds = test_ds.map(lambda x, y: prep(x, y, False), AT).batch(BATCH_SIZE).prefetch(AT)
    return train_ds, val_ds, test_ds


def build_model(builder):
    """Pre-trained backbone + custom classification head for binary output."""
    base = builder(include_top=False, weights="imagenet",
                   input_shape=IMG_SIZE + (3,), pooling="avg")
    base.trainable = False
    inp = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base(inp, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out), base


def get_probs(model, ds):
    """Return (y_true, y_prob) arrays for a dataset."""
    yt, yp = [], []
    for xb, yb in ds:
        yp.extend(model.predict(xb, verbose=0).ravel().tolist())
        yt.extend(yb.numpy().ravel().tolist())
    return np.array(yt), np.array(yp)


def best_threshold(y_true, y_prob):
    """Pick the threshold (on validation data) that maximizes F1."""
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def metrics_at(y_true, y_prob, thr):
    """Compute accuracy/precision/recall/F1/AUC at a given threshold."""
    pred = (y_prob >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {"accuracy": accuracy_score(y_true, pred), "precision": pr,
            "recall": rc, "f1": f1, "auc": auc(fpr, tpr)}


def train_and_evaluate(name, data_dir, seed=42):
    """Full two-stage fine-tuning for one model + one seed.
    Returns a dict of tuned-threshold metrics. Class weights handle imbalance."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    builder, preprocess = MODELS[name]
    train_ds, val_ds, test_ds = build_datasets(data_dir, preprocess, seed)
    model, base = build_model(builder)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                 restore_best_weights=True)
    class_weight = {0: 1.0, 1: 3500 / 700}

    # Stage 1: train head only
    model.compile(optimizers.Adam(LR_HEAD), "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
              callbacks=[es], class_weight=class_weight, verbose=0)

    # Stage 2: fine-tune last 30 layers
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizers.Adam(LR_FT), "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT,
              callbacks=[es], class_weight=class_weight, verbose=0)

    # Threshold chosen on validation, then applied to test
    yv, pv = get_probs(model, val_ds)
    thr = best_threshold(yv, pv)
    yt, yp = get_probs(model, test_ds)
    result = metrics_at(yt, yp, thr)
    result["threshold"] = thr
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train one TB classifier.")
    parser.add_argument("--data_dir", required=True,
                        help="Path to folder with Normal/ and Tuberculosis/ subfolders")
    parser.add_argument("--model", default="EfficientNetB0", choices=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    res = train_and_evaluate(args.model, args.data_dir, args.seed)
    print(f"{args.model} (seed {args.seed}):")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")
