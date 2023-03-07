#!/usr/bin/env python3
from mytools import *

print("START")


seed = 124
rng = np.random.default_rng(seed)


mean1=2
std1=1
mean2 = 3
std2 = 2

size = 1000

input_dim=6  # input layer length
h1 = 6       # hidden layer length

learning_rate = 0.06
num_iterations=1000

train, train_labels, test, test_label = initialize_train_test(mean1, std1, mean2, std2, size, rng)
print("train.shape ", train.shape)
print("test.shape ", test.shape)
#make_probability_labels(train, test, )

#exit()





#w, b = initialize_weights_random()
w1, b1, w2, b2 = initialize_weights_random(input_dim, h1, rng)

print("w1 ", w1)
print("b1", b1)
print("w2 ", w2)
print("b2", b2)

params, grads, costs, test_accuracy_array, train_accuracy_array = optimize(w1, b1, w2, b2, train, train_labels, num_iterations, learning_rate, test, test_label)

#print("test_accuracy_array")
#print(test_accuracy_array)


plot_learning_curve(costs)
plot_accuracy_curve(test_accuracy_array, train_accuracy_array)

print("optimized w1 ", params['w1'])
print("optimized b1 ", params['b1'])
print("optimized w2 ", params['w2'])
print("optimized b2 ", params['b2'])

Y_prediction_test = predict(params, test)
Y_prediction_train = predict(params, train)

Y_formula_prediction_test = formula_predict(test, mean1, std1, mean2, std2)



print("learning_rate = ", learning_rate, ", num_iterations = ", num_iterations)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_labels)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_label)) * 100))
print("formula test accuracy: {} %".format(100 - np.mean(np.abs(Y_formula_prediction_test - test_label)) * 100))
