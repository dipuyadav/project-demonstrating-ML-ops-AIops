from keras.datasets import mnist
dataset=mnist.load_data()


train, test = dataset
X_train, y_train = train
X_test, y_test = test


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Flatten

model = Sequential()

model.add(Convolution2D(filters = 2, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
parameter=1
filt=6
for parameter in range(parameter):
    model.add(Convolution2D(filters=filt, kernel_size=(3,3), activation='relu'))
    filt=filt*2
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=1)


a=model.history.history.get('accuracy')
a= a[0] * 100
a=int(a)


print("Accurcay of the model is:", a)


accfile = open("/acc/acc.txt", "w+")
accfile.write(str(a))
accfile.close()
