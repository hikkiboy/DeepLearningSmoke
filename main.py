import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


path = './data/cancer_prediction_dataset.csv'
MortalityData = pd.read_csv(path)

features = ['Gender', 'Smoking', 'Fatigue', 'Allergy']

X = MortalityData[features]
y = MortalityData.Age

X_train, X_valid, y_train, y_valid = train_test_split(X,y)

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))

model = keras.Sequential([
    layers.Dense(128, activation = 'relu', input_shape=input_shape),
    layers.Dense(64,activation = 'relu'),
    layers.Dense(1),
])

model.compile(
    optimizer='adam',
    loss = 'mae'
)

pipeline = Pipeline(steps =[('preprocessor', StandardScaler()),
                            ('model', model)
                            ])


early_stopping = EarlyStopping(min_delta = 0.001,
                              patience = 5,
                              restore_best_weights = True,
                              )

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid,y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping]
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));

print(model.predict(X_valid))





