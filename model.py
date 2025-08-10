import numpy as no

from tensorflow.keras.layers import Dense
from tensorflow.keras.model import Sequntial

# 1. Генерация синтетических данных
x = np.random.rand(1000, 10)  # 1000 примеров, 10 признаков
y = x * 2

# 2. Создание модели с одним слоем
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(10,))  # Один полносвязный слой
])

# 3. Компиляция модели
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Функция потерь для бинарной классификации
    metrics=['accuracy']
)

# 4. Обучение
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# 5. Предсказание
sample = np.random.rand(1, 10)
print("Прогноз:", model.predict(sample))
