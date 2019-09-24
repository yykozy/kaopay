%%time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%config InlineBackend.figure_formats = {'png', 'retina'}

import os, zipfile, io, re,tarfile
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception
from keras.models import Model, load_model
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

image_size = 100
#classes = ["male", "female"]
classes = ["shiraishi","saito","naoko"]
num_classes = len(classes)

X = []
Y = []

# TAR読み込み
with tarfile.open("./detect.tar.gz", mode="r:gz") as tar:
    for tarinfo in tar:
        if(tarinfo.isreg() and re.match('^.+jpg$',tarinfo.name)):
            #bytesで格納
            imgData = tar.extractfile(tarinfo).read()
            # TARから画像読み込み
            image = Image.open(io.BytesIO(imgData))
            print(tarinfo.name)
            # RGB変換
            image = image.convert('RGB')
            # リサイズ
            image = image.resize((image_size, image_size))
            # 画像から配列に変換
            data = np.asarray(image)
            file = os.path.basename(tarinfo.name)
            #file_split = [i for i in file.split('_')]
            X.append(data)
            #Y.append(file_split[1])
            Y.append(0)

# TAR読み込み
with tarfile.open("./detect2.tar.gz", mode="r:gz") as tar:
    for tarinfo in tar:
        print(tarinfo.name)
        if(tarinfo.isreg() and re.match('^.+jpg$',tarinfo.name)):
            #bytesで格納
            imgData = tar.extractfile(tarinfo).read()
            # TARから画像読み込み
            image = Image.open(io.BytesIO(imgData))
            # RGB変換
            image = image.convert('RGB')
            # リサイズ
            image = image.resize((image_size, image_size))
            # 画像から配列に変換
            data = np.asarray(image)
            file = os.path.basename(tarinfo.name)
            #file_split = [i for i in file.split('_')]
            X.append(data)
            #Y.append(file_split[1])
            Y.append(1)            
            
# TAR読み込み
with tarfile.open("./detect3.tar.gz", mode="r:gz") as tar:
    for tarinfo in tar:
        print(tarinfo.name)
        if(tarinfo.isreg() and re.match('^.+jpg$',tarinfo.name)):
            #bytesで格納
            imgData = tar.extractfile(tarinfo).read()
            # TARから画像読み込み
            image = Image.open(io.BytesIO(imgData))
            # RGB変換
            image = image.convert('RGB')
            # リサイズ
            image = image.resize((image_size, image_size))
            # 画像から配列に変換
            data = np.asarray(image)
            file = os.path.basename(tarinfo.name)
            #file_split = [i for i in file.split('_')]
            X.append(data)
            #Y.append(file_split[1])
            Y.append(2)
            
X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)

# trainデータとtestデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    random_state = 0,
    stratify = Y,
    test_size = 0.2
)
del X,Y
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# データ型の変換＆正規化
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
# one-hot変換
y_train = to_categorical(y_train, num_classes = num_classes)
y_test = to_categorical(y_test, num_classes = num_classes)

# trainデータからvalidデータを分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    random_state = 0,
    stratify = y_train,
    test_size = 0.2
)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape) 


#kerasモデル取得
base_model = Xception(
    include_top = False,
    weights = "imagenet",
    input_shape = None
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False
)

# EarlyStopping
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    verbose = 1
)

# ModelCheckpoint
weights_dir = './weights/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    period = 3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 3,
    verbose = 1
)

# log for TensorBoard
logging = TensorBoard(log_dir = "log/")

# ネットワーク定義
model = Model(inputs = base_model.input, outputs = predictions)

#108層までfreeze
for layer in model.layers[:108]:
    layer.trainable = False

    # Batch Normalizationのfreeze解除
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True
    if layer.name.endswith('bn'):
        layer.trainable = True

#109層以降、学習させる
for layer in model.layers[108:]:
    layer.trainable = True

# layer.trainableの設定後にcompile
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ["accuracy"]
)

hist = model.fit_generator(
    datagen.flow(X_train, y_train, batch_size = 32),
    steps_per_epoch = X_train.shape[0] // 32,
    epochs = 50,
    validation_data = (X_valid, y_valid),
    callbacks = [early_stopping, reduce_lr],
    shuffle = True,
    verbose = 1
)