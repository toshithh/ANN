import numpy as np
import cv2


class network:
    def __init__(self, y):
        self._output = y

    def activation_func(self, x):
        return (1/(1+np.exp(-x)))

    def f_forward(self, x):
        z1 = x.dot(self.w1)
        a1 = self.activation_func(z1)
        z2 = a1.dot(self.w2)
        a2 = self.activation_func(z2)
        a3 = self.activation_func(a2.dot(self.w3))
        return a3
    
    @property
    def weights(self):
        return (self.w1, self.w2, self.w3)

    @weights.setter
    def weights(self, x):
        self.w1, self.w2, self.w3 = x

    def generate_wt(self, x, y):
        l = []
        for i in range(x*y):
            l.append(np.random.randn())
        return (np.array(l).reshape(x, y))

    def loss_func(self, Y, y):
        s = np.square(y-Y)
        s = np.sum(s)/len(Y)
        return(s)

    def back_prop(self,x, y, w1, w2, w3, alpha):
        z1 = x.dot(w1)
        a1 = self.activation_func(z1)
        z2 = a1.dot(w2)
        a2 = self.activation_func(z2)
        a3 = self.activation_func(a2.dot(self.w3))
        
        d3 = (a3-y)
        d2 = np.multiply((w3.dot(d3.transpose())).transpose(),
        (np.multiply(a2, 1-a2)))
        d1 = np.multiply((w2.dot(d2.transpose())).transpose(),
        (np.multiply(a1, 1-a1)))
        w1_adj = x.transpose().dot(d1)
        w2_adj = a1.transpose().dot(d2)
        w3_adj = a2.transpose().dot(d3)
        w1 = w1-(alpha*(w1_adj))
        w2 = w2-(alpha*(w2_adj))
        w3 = w3-(alpha*(w3_adj))
        
        return(w1, w2, w3)

    def train(self, input, alpha=1, epoch=10):
        x = input
        y = self._output
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        accuracy = []
        loss = []
        for j in range(epoch):
            l = []
            l1 = []
            for i in range(len(x)):
                out = self.f_forward(x[i])
                l.append(self.loss_func(out, y[i]))
                w1, w2, w3 = self.back_prop(x[i], y[i],w1, w2, w3, alpha)
                self.w1 = w1
                self.w2 = w2
                self.w3 = w3
                acc = (1-(sum(l)/len(x)))*100
            print("epochs:", j+1, "====== acc:",acc)
            accuracy.append(acc)
            loss.append(sum(l)/len(x))
        return(w1, w2, w3)

    def predict(self, x):
        out = self.f_forward(x)
        print(out)
        max = -1
        k = 0
        for i in range(len(out[0])):
            if max<out[0][i]:
                max = out[0][i]
                k = i
        
        print(k)
