import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("data_zh.csv").values
data = np.delete(data, [4, 5, 28, 29], 0)

x = data[:, 1:5]
y = data[:, 10] # 第15案

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(units=10, input_dim=4, kernel_initializer="uniform"))
model.add(Dense(units=5, kernel_initializer="uniform", activation="relu"))
model.add(Dense(units=1,kernel_initializer="uniform", activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=x_train,
                          y=y_train,
                          validation_split=0.1,
                          epochs=10,
                          batch_size=10,
                          verbose=2)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history,'loss', 'val_loss')

# evaluate model accuracy
scores = model.evaluate(x_test, y_test, verbose=0)
print(scores[1])