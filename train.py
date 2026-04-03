import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load small dataset
data = pd.read_csv("mnist_small.csv")

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Normalize
X = X / 16.0   # NOTE: digits dataset uses 0–16

# Reshape (8x8 images)
X = X.reshape(-1, 8, 8)

# One-hot encoding
y = to_categorical(y, 10)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = Sequential([
    Flatten(input_shape=(8,8)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

model.save("mnist_model.h5")

print("Done!")
