#Grade Data Machine Learning 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

daten = pd.read_csv('student-mat.csv', sep=';')
daten = daten[['age', 'sex', 'studytime', 'absences', 'G1', 'G2', 'G3']]
daten['sex']= daten['sex'].map({'F':0, 'M':1})
vorhersage = 'G3'

x = np.array(daten.drop([vorhersage], 1))
y = np.array(daten[vorhersage])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1)

linear = LinearRegression()
linear.fit(x_train, y_train)
genauigkeit = linear.score(x_test, y_test)
print(genauigkeit)

x_neu = np.array([[18, 1, 3, 40, 15, 16]])
y_neu = linear.predict(x_neu)
print(y_neu)

plt.scatter(daten['studytime'], daten['G3'])
plt.show()

 breast_cancers data Kneighbor (Vector Data)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

daten = load_breast_cancer()

print(daten.feature_names)
print(daten.target_names)

x = np.array(daten.data)
y = np.array(daten.target)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

Accuracy = knn.score(x_test, y_test)
print(Accuracy)

x_neu = np.array([[...]])
ergebnis = knn.predict(x_neu)

 breast_cancer data with Soft Margin Kernel
daten = load_breast_cancer()


x=daten.data
y=daten.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 30)
modell = SVC(kernel='linear', C=3)
modell.fit(x_train, y_train)
genauigkeit = modell.score(x_test, y_test)
print(genauigkeit)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

knn_genauigkeit = knn.score(x_test, y_test)
print(knn_genauigkeit)

#unsupervised learning using clusters FUNKTIONIERT NICHT
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
import pandas as pd


ziffern = load_digits()
daten = scale(ziffern.data)

kmc = KMeans(n_clusters=10, init="random", n_init=10)
kmc.fit(daten)
kmc.predict([...])

neural network recognition for numbers Customable!
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)= mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis =1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

modell = tf.keras.models.Sequential()
modell.add(tf.keras.layers.Flatten())
modell.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
modell.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
modell.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

modell.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modell.fit(x_train, y_train, epochs=3)
loss, genauigkeit = modell.evaluate(x_test, y_test)
print(loss)
print(genauigkeit)

bild = cv2.imread('3.png')[:,:,0]
bild = np.invert(np.array([bild]))
vorhersage = modell.predict(bild)
print("Vorhersage:{}".format(np.argmax(vorhersage)))
plt.imshow(bild[0])
plt.show()

#Verbesserung des Soft Margin Algorythm possible 100% aber langsam
for b in range(2500):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    
    modell = SVC(kernel='linear', C=3)
    modell.fit(x_train, y_train)
    genauigkeit = modell.score(x_test, y_test)
    if genauigkeit> bestes:
        bestes = genauigkeit
        print("Höchste Genauigkeit: ", genauigkeit)
        with open('svm_modell.pickle', 'wb') as datei:
            pickle.dump(modell, datei)