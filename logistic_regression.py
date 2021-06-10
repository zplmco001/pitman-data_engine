import numpy


def sigmoid(x):
    return float(1.0 / float(1.0 + numpy.exp(-1.0 * x)))


class LogisticRegression:
    def __init__(self, iterations, alpha, dataset):
        self.iterations = iterations
        self.alpha = alpha
        self.dataset = dataset
        self.theta = numpy.zeros(len(dataset[0]) - 1)

    def hypothesis(self, coefficients):
        z = 0
        for i in range(len(self.theta)):
            z += coefficients[i] * self.theta[i]
        return sigmoid(z.astype(numpy.float128))

    def gradient_descent(self):
        new_theta = []
        for i in range(len(self.theta)):
            new_theta.append(self.cost_derivative(i))
        return new_theta

    def cost_derivative(self, index):
        error_sum = 0
        size = len(self.dataset)
        for i in range(size):
            xi = self.dataset[i]
            xij = xi[index]
            hi = self.hypothesis(xi)
            error_sum += (hi - xi[-1]) * xij
        c = float(self.alpha) / float(size)
        self.theta[index] = self.theta[index] - (error_sum * c)
        return self.theta[index]

    def cost_function(self):
        error_sum = 0
        error = 0
        for i in range(len(self.dataset)):
            if self.dataset[i][-1] == 1:
                error = self.dataset[i][-1] * numpy.log(self.hypothesis(self.dataset[i]))
            elif self.dataset[i][-1] == 1:
                error = (1 - self.dataset[i][-1]) * numpy.log(1 - self.hypothesis(self.dataset[i]))
            error_sum += error
        return (-1 / len(self.dataset)) * error_sum

    def train(self):
        for k in range(self.iterations):
            new_theta = self.gradient_descent()
            self.theta = new_theta
            # print("theta: ", self.theta)


'''dataset = [[3.393533211, 2.331273381, 0],
           [3.110073483, 1.781539638, 0],
           [1.343808831, 3.368360954, 0],
           [3.582294042, 4.67917911, 0],
           [2.280362439, 2.866990263, 0],
           [7.423436942, 4.696522875, 1],
           [5.745051997, 3.533989803, 1],
           [9.172168622, 2.511101045, 1],
           [7.792783481, 3.424088941, 1],
           [7.939820817, 0.791637231, 1]]'''

# theta = numpy.zeros(len(dataset[0]) - 1)
'''lg = LogisticRegression(100, 0.3, dataset)
lg.train()
prediction = round(lg.hypothesis(dataset[2]))
print(prediction)'''
