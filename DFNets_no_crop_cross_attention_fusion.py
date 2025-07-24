import logging, os
import tensorflow as tf
print(tf.__version__)

import random
import pathlib
import glob

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py

import pandas as pd
from pathlib import Path
from tensorflow.keras import layers

from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy

SEED = 42
tf.random.set_seed(SEED)

video_input = tf.keras.Input(shape=(10, 224, 224, 3))

class VideoFramesDataset:
    def __init__(self, root_dir, annotation_file, transform=False):
        self.root_dir = root_dir
        self.transform = transform
        #self._parse_annotationfile()

        self.video_list = annotation_file[0]
        self.label = annotation_file[0]

    def __len__(self):
        return len(self.video_list)

    def on_epoch_end(self):
        pass

    def  __call__(self):
        for idx in range(len(self.video_list)):
            frame = np.load(os.path.join(self.root_dir, self.video_list[idx][3:]+".npy"))
            #frame = frames_from_video_file(os.path.join(self.root_dir, self.video_list[idx][3:]+".avi"), 20, output_size = (224,224), frame_step = 15)
            frame = np.transpose(frame, axes=[1, 2, 3, 0])
            #print(frame.dtype)
            if self.label[idx][0] == "+":
                label = tf.convert_to_tensor([0])
            else:
                label = tf.convert_to_tensor([1])
            #label = tf.convert_to_tensor(int(self.label[idx][-1])-1)
            yield tf.convert_to_tensor(frame, dtype=tf.float32), (label,label,label)

n_frames = 10
batch_size = 12

AUTOTUNE = tf.data.AUTOTUNE
path = "."

# def augmentation(frame):
#   frame = tf.image.random_contrast(frame,0,2)
#   frame = tf.image.random_flip_left_right(frame)
#   frame = tf.image.random_brightness(frame,0.1)
#   return frame

def augmentation(color_image):
    color_image = tf.image.random_brightness(color_image, max_delta=0.1, seed=SEED)
    color_image = tf.image.random_contrast(color_image, 0, 2, seed=SEED)
    # color_image = tf.image.random_saturation(color_image, lower=0.8, upper=1.2, seed=SEED)
    # color_image = tf.image.random_hue(color_image, max_delta=0.2, seed=SEED)
    color_image = tf.image.random_flip_left_right(color_image, seed=SEED)
    # rot_k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=42)

    # def rotate(frame):
    #     frame = tf.image.rot90(frame, k=rot_k)
    #     return frame

    # color_image = tf.map_fn(rotate, color_image)
    # color_image = tf.clip_by_value(color_image, 0.0, 255)
    return color_image

train_dir = path+"/OULU-NPU/Train_files/Train_files/output/"
dev_dir = path+"/OULU-NPU/Dev_files/Dev_files/output/"
test_dir = path+"/OULU-NPU/Test_files/Test_files/output/"

#train_dir = path+"/Limited Source/O to I and M/Train OULU/Step 4/"
#train_annotation = pd.read_csv(os.path.join(path,"OULU-NPU/Train_files/Train_files/output", "annotation.txt"), sep = ' ', header=None).sample(frac=0.5, replace=None, weights=None, random_state=42, axis=None, ignore_index=True)

train_annotation = pd.read_csv(os.path.join(path, "OULU-NPU/Protocols/Protocols/Protocol_1/Train.txt"), sep = ' ', header=None)
dev_annotation = pd.read_csv(os.path.join(path, "OULU-NPU/Protocols/Protocols/Protocol_1/Dev.txt"), sep = ' ', header=None)
test_annotation = pd.read_csv(os.path.join(path, "OULU-NPU/Protocols/Protocols/Protocol_1/Test.txt"), sep = ' ', header=None)
#.sample(frac=0.5, replace=False, weights=None, axis=None, ignore_index=True)
output_signature = (
    tf.TensorSpec(shape=(10, 224, 224, 3), dtype=tf.float32),  # input video frames
    (
        tf.TensorSpec(shape=(1,), dtype=tf.int32),  # for output
        tf.TensorSpec(shape=(1,), dtype=tf.int32),  # for gap
        tf.TensorSpec(shape=(1,), dtype=tf.int32),  # for gap_saliency
    )
)

train_ds_oulu = tf.data.Dataset.from_generator(VideoFramesDataset(train_dir, train_annotation), output_signature = output_signature)
train_ds_oulu = train_ds_oulu.shuffle(32, seed=SEED, reshuffle_each_iteration=True)
train_ds_oulu = train_ds_oulu.map(lambda x, y: (augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds_oulu = train_ds_oulu.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache().repeat()

dev_ds_oulu = tf.data.Dataset.from_generator(VideoFramesDataset(dev_dir, dev_annotation), output_signature = output_signature)
dev_ds_oulu = dev_ds_oulu.shuffle(32).batch(batch_size).prefetch(AUTOTUNE)

test_ds_oulu = tf.data.Dataset.from_generator(VideoFramesDataset(test_dir, test_annotation), output_signature = output_signature)
test_ds_oulu = test_ds_oulu.batch(batch_size).prefetch(AUTOTUNE)

class SpatialAttentionModule(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def compute_output_shape(self, input_shape):
        # input_shape: (B, H, W, C)
        # output: same spatial dims, last channel = 1
        return input_shape[:-1] + (1,)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        return self.conv(concat)

def build_decomposition(video_input):
    base = tf.keras.applications.ResNet50(include_top=False, input_shape=(224,224,3), weights="imagenet")
    base.trainable = True

    base_sal = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224,224,3), weights="imagenet")
    base_sal.trainable = True

    # CNN1 branch
    features = layers.TimeDistributed(base)(video_input)
    drop = layers.Dropout(0.3)(features)
    dense = layers.Dense(1, name="dense")(drop)
    gap = layers.GlobalAveragePooling3D(name='cnn1_output')(dense)

    # CNN2 branch
    # masks = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(features)
    # masks = layers.TimeDistributed(layers.Conv2D(1, 1, activation="sigmoid"))(features)
    masks = layers.TimeDistributed(SpatialAttentionModule())(features)
    masks = layers.TimeDistributed(layers.Resizing(224, 224))(masks)
    masks = layers.Lambda(lambda x: 1.0 - x + 1e-6)(masks)
    masked_input = layers.Multiply(name="masked_input")([video_input, masks])

    features_sal = layers.TimeDistributed(base_sal)(masked_input)
    drop_saliency = layers.Dropout(0.3)(features_sal)
    dense_saliency = layers.Dense(1, name="dense_saliency")(drop_saliency)
    gap_sal = layers.GlobalAveragePooling3D(name='cnn2_output')(dense_saliency)

    return features, features_sal, gap, gap_sal, masked_input

from tensorflow.keras import layers, models, Input, Model, callbacks, optimizers

def CrossAttention(features1, features2, dim=32, num_heads=4):
    # Asumsikan input: (B, T, H, W, C)
    # Gabung dimensi: (B*T, H*W, C)
    B, T, H, W, C = features1.shape

    reshaped_q = layers.Reshape((-1, features1.shape[-1]))(features1)  # (B, T*H*W, C)
    reshaped_k = layers.Reshape((-1, features2.shape[-1]))(features2)  # (B, T*H*W, C)

    # Lakukan Attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=dim
    )(reshaped_q, reshaped_k, reshaped_k)  # (B, T*H*W, C)

    # Kembalikan ke shape asli
    output = layers.Reshape((T, H, W, C))(attn)
    return output

def psi_net(features1, features2, dim=32, num_heads=4):
    # features1 -> features2
    cross12 = CrossAttention(features1, features2, dim=dim, num_heads=num_heads)
    # features2 -> features1
    cross21 = CrossAttention(features2, features1, dim=dim, num_heads=num_heads)

    fused = layers.Concatenate(axis=-1)([cross12, cross21])
    fused = layers.TimeDistributed(layers.Conv2D(dim, (1,1), padding='same'))(fused)
    fused = layers.TimeDistributed(layers.BatchNormalization())(fused)
    fused = layers.TimeDistributed(layers.ReLU())(fused)

    return fused

### -------------------------------
### MODEL
### -------------------------------
def build_full_model(input_shape, dim=32, num_heads=4):
    video_input = Input(shape=input_shape, name='video_input')

    ### CNN1: Fitur utama
    cnn1_feat, cnn2_feat, gap, gap_sal, masks = build_decomposition(video_input)

    ### Fusion
    fusion = psi_net(cnn1_feat, cnn2_feat, dim=dim, num_heads=num_heads)

    ### Output
    drop = layers.Dropout(0.3)(fusion)
    dense = layers.Dense(32, activation='relu', name="dense_fusion")(drop)
    gap_fusion = layers.GlobalAveragePooling3D()(dense)
    output = layers.Dense(1, name='fusion_output')(gap_fusion)

    ### Model utama untuk training
    main_model = Model(inputs=video_input, outputs=[gap, gap_sal, output])

    ### Model untuk saliency output saja
    saliency_model = Model(inputs=video_input, outputs=masks)

    return main_model, saliency_model

### -------------------------------
### WARMUP CALLBACK
### -------------------------------
class WarmupCallback(callbacks.Callback):
    def __init__(self, warmup_epoch=5):
        super().__init__()
        self.warmup_epoch = warmup_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.warmup_epoch:
            print(f"[INFO] Unfreeze psi_dense @ epoch {epoch}")
            self.model.get_layer("psi_dense").trainable = True

### -------------------------------
### MAIN TRAINING
### -------------------------------
if __name__ == "__main__":
    input_shape = (10, 224, 224, 3)
    model, model_saliency = build_full_model(input_shape=(10, 224, 224, 3), dim=128, num_heads=4)

    # Freeze psi_dense at first
    # model.get_layer("psi_dense").trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1 = 0.9, weight_decay = 0.0005),
        loss={
            "cnn1_output": BinaryFocalCrossentropy(from_logits=True),
            "cnn2_output": BinaryFocalCrossentropy(from_logits=True),
            "fusion_output": BinaryFocalCrossentropy(from_logits=True),
        },
        loss_weights={
            "cnn1_output": 1.0,
            "cnn2_output": 1.0,
            "fusion_output": 1.0,
        },
        metrics={"fusion_output": AUC(from_logits=True, name="auc")}
        )

    print(model.summary())

    # Dummy data
    x_train = np.random.rand(1, *input_shape)

    _ = model(x_train)

    model.load_weights('DFNets_no_crop_fusion_60.weights.h5')

    # Train with warmup
    model.fit(
        train_ds_oulu,
        epochs=20,
        steps_per_epoch=125,
        # callbacks=[WarmupCallback(warmup_epoch=5)]
    )

model.save_weights('DFNets_no_crop_fusion_60.weights.h5')

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc

_,_,y_pred_keras = model.predict(test_ds_oulu)
y_pred_keras = y_pred_keras.ravel()

y_test = []
for idx in range(len(y_pred_keras)):
    if test_annotation[0][idx][0] == '+': # 0 as genuine
        y_test.extend([0])
    else:
        y_test.extend([1])  # 1 as fake

y_test = np.array(y_test)

fpr = list()
fnr = list()

thresholds = np.linspace(y_pred_keras.min(), y_pred_keras.max(), 1000)
for thr in thresholds:
    pred = y_pred_keras >= thr
    TN = np.sum((y_test == 0) & (pred == False))  # Genuine detected as Genuine: True Negative   -- True Accept
    FN = np.sum((y_test == 1) & (pred == False))  # Fake detected as genuine: False Negative  -- False Accept
    FP = np.sum((y_test == 0) & (pred == True))   # Genuine detected as fake: False Positive  -- False Reject
    TP = np.sum((y_test == 1) & (pred == True))   # Fake detected as Fake: True Positive   -- True Reject
    x = 0 if FP == 0 else FP/(FP+TN)
    y = 0 if FN == 0 else FN/(FN+TP)
    fpr.extend([x])
    fnr.extend([y])

fpr = np.array(fpr)
fnr = np.array(fnr)

auc = auc(fpr, 1-fnr)
bpcer = fpr[np.argmin(abs(fpr+fnr))]
apcer = fnr[np.argmin(abs(fpr+fnr))]
acer = (apcer+bpcer)/2

print(apcer)
print(bpcer)
print(acer)
print(auc)

# # Visualize Mask
# # Ambil output layer Multiply yang sudah dikasih nama
# masked_output = model.get_layer("masked_input").output

# # Buat sub-model baru
# masked_model = tf.keras.Model(inputs=model.input, outputs=masked_output)

# Prepare sample input
sample_video = next(iter(train_ds_oulu))[0]  # shape (B, N, 224, 224, 3)

# Sekarang bisa panggil
masks_val = model_saliency.predict(sample_video)

import matplotlib.pyplot as plt

# Convert to numpy if needed
if isinstance(masks_val, tf.Tensor):
    masks_val = masks_val.numpy()

import matplotlib.pyplot as plt
# np.save('masks.npy', masks_val[0, 0, :, :, :])

plt.imshow(masks_val[0, 0, :, :, :])
plt.axis("off")
plt.title("Inverted Saliency Mask")
plt.savefig("mask_sample.png", bbox_inches='tight')  # saves to disk