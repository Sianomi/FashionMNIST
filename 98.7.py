import keras
import math
import matplotlib.pyplot as plt

# loss 그래프 그리는 함수
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

# accuracy 그래프 그리는 함수
def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)


mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes=10)

kernel_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.0012, seed=None)

input_tensor = keras.layers.Input(shape=input_shape, name='input_tensor')
# start

x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='lecun_normal')(input_tensor)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='lecun_normal')(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=200, activation='relu', kernel_initializer='lecun_normal')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(units=50, activation='relu', kernel_initializer='lecun_normal')(x)
x = keras.layers.BatchNormalization()(x)

adam = keras.optimizers.Adam(beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

# end
output_tensor = keras.layers.Dense(units=10, activation='softmax', name='output_tensor')(x)
model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00005)

history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping, reduce_lr])
score = model.evaluate(x_test, y_test, verbose=2)

print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

# model.save("test.h5")

# 학습 loss 값 과 accuracy 결과 그래프
plot_loss(history)
plt.show()
plot_acc(history)
plt.show()
