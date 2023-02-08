import cv2
import numpy as np
from NeuralNetwork import network
import time

def ready(img_path):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (56, 71))
    img = (img/255)
    return (np.array(img).reshape(1,3976))



output = [[0 for x in range(10)] for x in range(10)]
for i in range(len(output)):
    output[i][i] = 1
NN = network(output)

NN.w1, NN.w2, NN.w3 = [np.genfromtxt(f"{x}.csv", delimiter=",") for x in ("w1", "w2", "w3")]

"""
w1 = NN.generate_wt(3976, 497)
w2 = NN.generate_wt(497, 50)
w3 = NN.generate_wt(50, 10)

NN.weights = (w1, w2, w3)

input = [ready(f"train/{x}.png") for x in range(10)]
NN.train(input, 0.01, 1000)

w1, w2, w3 = NN.weights
np.savetxt("w1.csv", w1, delimiter=",")
np.savetxt("w2.csv", w2, delimiter=",")
np.savetxt("w3.csv", w3, delimiter=",")
"""

while 1:
    img = input("Enter name of image of number: ")
    img = ready(f"test/{img}.png")
    NN.predict(img)