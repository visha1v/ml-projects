import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils
import os

dataset_path = 'path_to_your_gtzan_dataset'

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None

def load_data(dataset_path):
    features = []
    labels = []

    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue
        
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(genre)
    
    return np.array(features), np.array(labels)

X, y = load_data(dataset_path)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = np_utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(256, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_one_hot.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

model.save('music_genre_classification_model.h5')
