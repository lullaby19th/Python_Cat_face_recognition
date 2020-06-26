import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import mobilenet_v2
import numpy as np

img_size = 224

mode = 'lmks' # [bbs, lmks]

if mode is 'bbs':
  output_size = 4
  
# 이번에는 landmark들을 학습시킬 것이기 때문에 output_size = 18이다
# (x1,y1) (x2,y2) 이렇게 9개 이므로 2x9 = 18개의 output_size!

elif mode is 'lmks':
  output_size = 18

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


# 에러 발생 // ValueError: Object arrays cannot be loaded when allow_pickle=False
# 먼저 기존의 np.load를 np_load_old에 저장해둠.
np_load_old = np.load

# 기존의 parameter을 바꿔줌
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# 이러고도 오류가 발생하면 restart kernel을 추천함.

print('dataloads start!')

data_00 = np.load('E:\\cat_hipsterizer-master\\dataset\\lmks_CAT_00.npy')
data_01 = np.load('E:\\cat_hipsterizer-master\\dataset\\lmks_CAT_01.npy')
data_02 = np.load('E:\\cat_hipsterizer-master\\dataset\\lmks_CAT_02.npy')
data_03 = np.load('E:\\cat_hipsterizer-master\\dataset\\lmks_CAT_03.npy')
data_04 = np.load('E:\\cat_hipsterizer-master\\dataset\\lmks_CAT_04.npy')
data_05 = np.load('E:\\cat_hipsterizer-master\\dataset\\lmks_CAT_05.npy')
data_06 = np.load('E:\\cat_hipsterizer-master\\dataset\\lmks_CAT_06.npy')

print('dataloads finish!')
print('data preprocessing start!')

# x_train : 학습용 이미지 데이터
# y_train : 학습용 정답 레이블 (landmark 레이블)
# x_test : 테스트용 이미지 데이터
# y_test : 테스트용 정답 레이블 (landmark 레이블)

x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs'), data_04.item().get('imgs'), data_05.item().get('imgs')), axis=0)
y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode), data_04.item().get(mode), data_05.item().get(mode)), axis=0)

x_test = np.array(data_06.item().get('imgs'))
y_test = np.array(data_06.item().get(mode))

# image를 0~1까지의 값으로 정규화하여 training을 원할하게 함.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# np.reshape로 크기를 img_size = 224에 맞게 setting 해줌.
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))

y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

inputs = Input(shape=(img_size, img_size, 3))

print('data preprocessing finish!')
print('model build start!')

# 에러 발생 // TypeError: ('Invalid keyword argument: %s', 'depth_multiplier')
# depth_multiplier을 삭제해줌

mobilenetv2_model = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, \
                                            include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()

# training
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

model.fit(x_train, y_train, epochs=50, batch_size=32, shuffle=True,
  validation_data=(x_test, y_test), verbose=1,
  callbacks=[
    TensorBoard(log_dir='E:\\cat_hipsterizer-master\\logs\\%s_lmks' % (start_time)),
    ModelCheckpoint('E:\\cat_hipsterizer-master\\models\\%s_lmks.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
  ]
)

print('model training finish!')