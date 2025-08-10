import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt
import zipfile
import gdown
from itertools import chain


gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l7/tesla.zip', 'tesla.zip', quiet=True)

with zipfile.ZipFile('tesla.zip', 'r') as zip_ref:
    zip_ref.extractall('./tesla/')

print("Архив успешно распакован в папку tesla/")

# Объявляем функции для чтения файла. На вход отправляем путь к файлу
def read_text(file_name):

  read_file = open(file_name, 'r')
  text = read_file.read()
  text = text.replace("\n", ".")
  text = text.replace(", ,","")

  return text

class_names = ["Негативный отзыв", "Позитивный отзыв"]

num_classes = len(class_names)

texts_list = []

for j in os.listdir('./tesla/'):
        texts_list.append(read_text('./tesla/' + j))

        print(j, 'добавлен в обучающую выборку')
print(len(texts_list))

positive_feedback = texts_list[0].split('.')
negative_feedback = texts_list[1].split('.')
positive_feedback = [line for line in positive_feedback if line.strip()]
negative_feedback = [line for line in negative_feedback if line.strip()]
print(positive_feedback[0])
print(negative_feedback[0])


data_positive = list(chain(
    zip(positive_feedback, [1] * len(positive_feedback))
))
data_negative = list(chain(
    zip(negative_feedback, [0] * len(negative_feedback))
))


print(data_positive[0])
print(data_negative[0])

x_train_positive = [value for value, key in data_positive]
y_train_positive = [key for value, key in data_positive]
x_train_negative = [value for value, key in data_negative]
y_train_negative = [key for value, key in data_negative]
print(x_train_positive[2])
print(y_train_positive[2])
print(x_train_negative[2])
print(y_train_negative[2])

tokenizer_1 = Tokenizer(num_words=3500, filters='!"#$%&()*+,-–—./…:..;<=>?@[\\]^_`" "{|}~«»\t\n\xa0\ufeff', lower=True, split=" ", oov_token='?')
tokenizer_0 = Tokenizer(num_words=3500, filters='!"#$%&()*+,-–—./…:..;<=>?@[\\]^_`" "{|}~«»\t\n\xa0\ufeff', lower=True, split=" ",oov_token='?')
tokenizer_1.fit_on_texts(x_train_positive)
tokenizer_0.fit_on_texts(x_train_negative)

positive_seq = tokenizer_1.texts_to_sequences(positive_feedback)
negative_seq = tokenizer_0.texts_to_sequences(negative_feedback)

x = positive_seq + negative_seq
y = y_train_positive + y_train_negative
x_data, y_data = x, y
print(x_data[0])
print(y_data[0])

data_x_y = list(zip(x_data, y_data))

shuffle_x_y = shuffle(data_x_y, random_state=32)

x_train, y_train = zip(*shuffle_x_y)
print(len(x_train))
print(len(y_train))

maxlen=15
positive_seq = pad_sequences(x_data, maxlen=maxlen, padding='post')

maxlen
num_classes
max_words = 10000

x_train = np.array(positive_seq)
y_train = utils.to_categorical(y_train, num_classes)
print(x_train[555], y_train[555])

y_train = np.argmax(y_train, axis=1)
print(x_train[555], y_train[555])

model_bow = Sequential()
model_bow.add(Dense(128, activation='relu', input_shape=(maxlen,)))
model_bow.add(Dense(64, activation='relu'))
model_bow.add(Dense(1, activation='sigmoid'))

model_bow.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_bow = model_bow.fit(x_train,
          y_train,
          epochs=25,
          batch_size=256,
          validation_split=0.2,
          shuffle=True)

model_bow.save('tesla_model.h5')
print('Модель сохранена')
