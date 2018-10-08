import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("./data/dataR2.csv")
x = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, 9].values
y = np.reshape(y, (len(y), 1))

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
scaled_x = scaler_x.fit_transform(x)
scaled_y = scaler_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(x_train, y_train)

y_predictions = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predictions)
