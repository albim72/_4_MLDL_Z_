import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class KohonenNetwork:
    def __init__(self,input_dim,output_dim,larning_rate=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.larning_rate = larning_rate
        self.weights = np.random.rand(output_dim,input_dim)


    def find_winner(self,sample):
        distances = np.linalg.norm(self.weights - sample,axis=1)
        return np.argmin(distances)

    def neighberhood(self,winner_idx,nuron_idx,epoch,max_epochs):
        sigma = self.output_dim/2*(1-epoch/max_epochs)
        distance = abs(winner_idx - nuron_idx)
        return np.exp(-(distance**2)/(2*sigma**2))

    def upadate_weights(self,sample,winner_idx,epoch,max_epochs):
        learning_rate = self.larning_rate*(1-epoch/max_epochs)
        for i in range(self.output_dim):
            self.weights[i] += learning_rate*self.neighberhood(winner_idx,i,epoch,max_epochs)*(sample-self.weights[i])

    def train(self,data,epochs=100):
        for epoch in range(epochs):
            for sample in data:
                winner_idx = self.find_winner(sample)
                self.upadate_weights(sample,winner_idx,epoch,epochs)

    def predict(self,data):
        predictions = []
        for sample in data:
            winner_idx = self.find_winner(sample)
            predictions.append(winner_idx)
        return predictions
