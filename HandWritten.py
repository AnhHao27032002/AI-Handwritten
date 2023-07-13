import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Load dataset
mnist = tf.keras.datasets.mnist

#  Chia thành chạy thử và training
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))     #input layer(1)
model.add(tf.keras.layers.Dense(128, activation='relu'))     #hidden layer(2)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))   #output layer(3)

#Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training Model
model.fit(x_train, y_train, epochs=10)
#Test model
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')

for x in range(1,6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'Kết quả sẽ là:{np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
