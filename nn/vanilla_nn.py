## https://sharavaa.blogspot.com/2017/08/vanilla.html

import numpy as np

# амарсан хугацаа болон хичээл давтсан хугацаа
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)

# Шалгалтанд авсан огноо
y = np.array(([92], [86], [89]), dtype=float)

# тэлэлтийн нэгж
X = X/np.amax(X, axis=0) # X массивийн хамгийн их утга
y = y/100
print(y)

class NeuralNetwork(object):
    # Неорон сүлжээний байгуулагч функц
    def __init__(self):
        self.input_size  = 2
        self.output_size = 1
        self.hidden_size = 3

        # жингүүд
        # оролтоос далд давхарга хүртэлхи жингийн матриц
        self.W1 = np.random.randn(self.input_size, self.hidden_size)

        # далд давхаргаас гаралтын давхарга хүртэлхи жингийн матриц
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    # сүлжээгээр урагш тархалт хийх
    def forward(self, X):
        # X оролт болон эхний тохируулсан матриц хоёрын үржвэр
        self.z = np.dot(X, self.W1)

        # activation функц
        self.z2 = self.sigmoid(self.z)

        # далд давхарга болох z2 болон хоёр дахь матриц хоёрын хоорондох үржвэр
        self.z3 = np.dot(self.z2, self.W2)

        # сүүлийн activation функц
        output = self.sigmoid(self.z3)

        return output

    # activation функц
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    # sigmoid ийн уламжлал
    def sigmoid_prime(self, s):
        return s*(1-s)

    # cүлжээгээр буцан тархах
    def backward(self, X, y, output):
        # гаралт утгын алдааны хэмжээ
        self.output_error = y-output

        # sigmoid функцийн уламжлалыг гаралтын алдааны утганд хэрэглэх
        self.output_delta = self.output_error*self.sigmoid_prime(output)

        # z2 алдаа : далд давхаргын жингийн утгууд нь гаралтын алдаанд
        # ямархуу хэмжээний оролцоотой байна вэ гэдгийг илтгэнэ.
        self.z2_error = self.output_delta.dot(self.W2.T)

        # sigmoid ийн уламжлалыг z2 алдаанд хэрэглэх
        self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2)

        # (оролтоос --> далд давхарга) эхний жингүүдийг тохируулах
        self.W1 += X.T.dot(self.z2_delta)

        # (далд давхаргаас --> гаралт) хоёр дахь жингүүдийн матрицийг тохируулах
        self.W2 += self.z2.T.dot(self.output_delta)


    # neuron surgah function
    def train(self, X, y):
        output = self.forward(X)
        self.backward(X, y, output)



neural_network = NeuralNetwork()

# неорон сүлжээг 1-ээс 1000 удаа давтан сургана
for i in range(1000):
    # print("Оролт : \n"+str(X))
    print("Жинхэнэ гаралтын утга: \n"+str(y))
    print("Таамгалсан гаралтын утга: \n"+str(neural_network.forward(X)))
    print("Loss: \n"+str(np.mean(np.square(y-neural_network.forward(X)))))
    neural_network.train(X, y)

print("\n")
print("8 цаг амарч 1 цаг хичээл давтсан :")
print(str(neural_network.forward(np.array(([8, 1]), dtype=float))))
print("1 цаг амарч 10 цаг хичээл давтсан :")
print(str(neural_network.forward(np.array(([1, 10]), dtype=float))))
