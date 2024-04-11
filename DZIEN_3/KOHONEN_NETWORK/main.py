import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from kohonen import KohonenNetwork

#wczytywanie danych
iris = load_iris()
data = iris.data
target = iris.target

#normalizacja danych
data_norm = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

#budowa i trenowanie sieci Kohonena

kohonen = KohonenNetwork(input_dim=data_norm.shape[1],output_dim=10,larning_rate=0.1)
kohonen.train(data_norm,epochs=120)

#predykcja

predictions = kohonen.predict(data_norm)

#wizualizacja

plt.scatter(data[:,0],data[:,1],c=predictions)
plt.title('Sieć kohonena -> Irysy')
plt.xlabel('Cecha1')
plt.ylabel('Cecha2')
plt.colorbar(label='Klaster')
plt.show()
