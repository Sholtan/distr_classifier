#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def normal(x, mean, std):
    firstpart = std*np.sqrt(2*np.pi)
    inter = ((x-mean)/std) * ((x-mean)/std)
    secpart = np.exp(-0.5*inter)
    f = secpart/firstpart
    return f


def plot_input_data(distr1, distr2):
    samples = distr1.shape[0]
    distr1 = distr1.reshape(samples*6, 1)
    distr2 = distr2.reshape(samples*6, 1)
    x = np.concatenate((distr1, distr2), axis=1)


    fig, ax = plt.subplots()


    ax.hist(x, 100, density=False, histtype='step', stacked=False)
    ax.set_title("gaussians")


    

    #plt.show(block=False)
    #fig.canvas.draw()
    fig.savefig("gaussians.png")
   


def sigmoid(z):
    """
    Compute the sigmoid of z
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(z)
    """
    s = 1/(1+np.exp(-z))
    return s

def make_probability_labels(distr1, distr2, mean1, std1, mean2, std2, size):
    '''
    calculates for every vector of 6 numbers probability it being from 1 distribution and probability it being from 2 distribution
    '''

    k1 = 1/(std1*np.sqrt(2*np.pi))
    k2 = 1/(std2*np.sqrt(2*np.pi))
    distr1_prob_dens1 = k1*np.exp(-0.5*np.power((distr1-mean1)/std1,2))
    distr1_prob_dens2 = k2*np.exp(-0.5*np.power((distr1-mean2)/std2,2))
    
    distr1_prob_dens1_flat = np.zeros(size)
    for i in range(size):
        distr1_prob_dens1_flat[i] = distr1_prob_dens1[i,0]*distr1_prob_dens1[i,1]*distr1_prob_dens1[i,2]*distr1_prob_dens1[i,3]*distr1_prob_dens1[i,4]*distr1_prob_dens1[i,5]

    distr1_prob_dens2_flat = np.zeros(size)
    for i in range(size):
        distr1_prob_dens2_flat[i] = distr1_prob_dens2[i,0]*distr1_prob_dens2[i,1]*distr1_prob_dens2[i,2]*distr1_prob_dens2[i,3]*distr1_prob_dens2[i,4]*distr1_prob_dens2[i,5]

    distr1_prob_labels = np.zeros(size)
    for i in range(size):
        distr1_prob_labels[i] = distr1_prob_dens2_flat[i]/(distr1_prob_dens2_flat[i]+distr1_prob_dens1_flat[i])


    return 100*distr1_prob_labels


def formula_predict(distr1, mean1, std1, mean2, std2):
    k1 = 1/(std1*np.sqrt(2*np.pi))
    k2 = 1/(std2*np.sqrt(2*np.pi))
    distr1_prob_dens1 = k1*np.exp(-0.5*np.power((distr1.T-mean1)/std1,2))
    distr1_prob_dens2 = k2*np.exp(-0.5*np.power((distr1.T-mean2)/std2,2))

    size = distr1.shape[1]

    print("SIZE ", size)

    distr1_prob_dens1_flat = np.zeros(size)
    for i in range(size):
        distr1_prob_dens1_flat[i] = distr1_prob_dens1[i,0]*distr1_prob_dens1[i,1]*distr1_prob_dens1[i,2]*distr1_prob_dens1[i,3]*distr1_prob_dens1[i,4]*distr1_prob_dens1[i,5]

    distr1_prob_dens2_flat = np.zeros(size)
    for i in range(size):
        distr1_prob_dens2_flat[i] = distr1_prob_dens2[i,0]*distr1_prob_dens2[i,1]*distr1_prob_dens2[i,2]*distr1_prob_dens2[i,3]*distr1_prob_dens2[i,4]*distr1_prob_dens2[i,5]

    distr1_prob_labels = np.zeros(size)

    #print("distr1_prob_dens1_flat")
    #print(distr1_prob_dens1_flat)
    #print("distr1_prob_dens2_flat")
    #print(distr1_prob_dens2_flat)


    for i in range(size):
        distr1_prob_labels[i] = distr1_prob_dens2_flat[i]/(distr1_prob_dens2_flat[i]+distr1_prob_dens1_flat[i])

    distr1_prob_labels[distr1_prob_labels>0.5] = 1
    distr1_prob_labels[distr1_prob_labels<=0.5] = 0

    return distr1_prob_labels   



def initialize_train_test(mean1, std1, mean2, std2, size, rng):
    
    
    normalizer = 1
    
    distr1 = (mean1 + std1*rng.standard_normal((size,6)))
    distr1_labels = np.zeros(size)
    
    distr2 = (mean2 + std2*rng.standard_normal((size,6)))
    distr2_labels = np.ones(size)
    
    distr1_prob_labels = make_probability_labels(distr1, distr2, mean1, std1, mean2, std2, size)

    plot_input_data(distr1, distr2)


    input_x = np.concatenate((distr1, distr2), axis=0)
    input_labels = np.concatenate((distr1_labels, distr2_labels), axis=0)
    permut = rng.permutation(np.arange(2*size))
    input_x = input_x[permut]
    input_labels = input_labels[permut]

    train_size = int(0.8*2*size)

    train = input_x[0:train_size]
    train_labels = input_labels[0:train_size]

    test = input_x[train_size:-1]
    test_label = input_labels[train_size:-1]

    return train.T, train_labels.T, test.T, test_label.T


def initialize_weights_random(n1, n2, rng):
    """
    This function creates a vector with random values btw [0,1] of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    #rng = np.random.default_rng(seed)
    w1 = rng.random([n2, n1])*0.1
    b1 = rng.random([n2,1])*0.1
    w2 = rng.random([1, n2])*0.1
    b2 = 0.1
    return w1, b1, w2, b2

def initialize_weights_zeros(n1=6, n2 = 2):
    """
    This function creates a vector filled with zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w1 = np.zeros((n2, n1))
    b1 = np.zeros((n2,1))
    w2 = np.zeros((1, n2))
    b2 = 0.
    return w1, b1, w2, b2


def propagate(w1, b1, w2, b2, X, Y):
    """
    Arguments:
    w1 -- weights, size (n2, n1)
    b1 -- bias, a vector size (n2, 1)
    w2 -- weights, size (1, n2)
    b2 -- bias, a float
    X -- data of size (6, number of examples)
    Y -- true "label" vector (containing 0 if left, 1 if right) of size (1, number of examples)
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    m = X.shape[1] # number of examples
    
    # FORWARD PROPAGATION (FROM X TO COST)

    z1 = np.dot(w1, X) + b1  #  (n2, n1)(n1, m) + (n2, 1)
    a1 = np.tanh(z1)         #  (n2, m)
    z2 = np.dot(w2, a1) + b2 #  (1, n2)(n2, m) + (1)
    a2 = sigmoid(z2)         #  (1, m)  compute activation
    
    cost = -np.sum(Y*np.log(a2)+(1-Y)*np.log(1-a2))/m   # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dz2 = a2-Y                                        # (1, m)
    dw2 = np.dot(dz2, a1.T)/m                         # (1, m)(m, n2)
    db2 = np.sum(dz2, axis=1, keepdims=True)/m        # (1)

    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))   # (n2, 1)(1, m) * (n2, m)
    dw1 = (np.dot(dz1, X.T))/m                        # (n2, m)(m, n1)
    db1 = np.sum(dz1, axis=1, keepdims=True)/m        # (n2, 1)



   

    #assert(dw.shape == w.shape)
    #assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
    
    return grads, cost



def optimize(w1, b1, w2, b2, X, Y, num_iterations, learning_rate, test, test_label):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (6, 1)
    b -- bias, a scalar
    X -- data of shape (6, number of examples)
    Y -- true "label" vector (containing -1 if left, 1 if right), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    test_accuracy_array = []
    train_accuracy_array = []
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code) 
        grads, cost = propagate(w1, b1, w2, b2, X, Y)
        
        # Retrieve derivatives from grads
        dw1 = grads["dw1"]
        db1 = grads["db1"]
        dw2 = grads["dw2"]
        db2 = grads["db2"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w1 = w1 - learning_rate*dw1
        b1 = b1 - learning_rate*db1
        w2 = w2 - learning_rate*dw2
        b2 = b2 - learning_rate*db2
        ### END CODE HERE ###
        
        # Record the costs
        #if i%10==0:
        costs.append(cost)
        
        Y_prediction_test = predict({'w1':w1, 'w2':w2, 'b1':b1, 'b2':b2}, test)
        Y_prediction_train = predict({'w1':w1, 'w2':w2, 'b1':b1, 'b2':b2}, X)


        train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y)) * 100
        test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - test_label)) * 100
        #print("test_accuracy: ", test_accuracy)
        test_accuracy_array.append(test_accuracy)
        train_accuracy_array.append(train_accuracy)

        # Print the cost every 100 training iterations
        
        #if print_cost and i % 100 == 0:
            #print ("Cost after iteration %i: %f" %(i, cost))
 

    params = {"w1": w1,
              "b1": b1,
              "w2": w2,
              "b2": b2}
    
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
    
    return params, grads, costs, test_accuracy_array, train_accuracy_array


def predict(params, input_data):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (6, 1)
    b -- bias, a scalar
    input_data -- data of size (n1, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in input_data
    '''

    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']


    m = input_data.shape[1]   # number of examples
    y_prediction = np.zeros((1,m))


    z1 = np.dot(w1, input_data) + b1  #  (n2, n1)(n1, m) + (n2, 1)
    a1 = np.tanh(z1)         #  (n2, m)
    z2 = np.dot(w2, a1) + b2 #  (1, n2)(n2, m) + (1)
    A = sigmoid(z2)         #  (1, m)  compute activation
    



    A[A>0.5] = 1
    A[A<=0.5] = 0
    y_prediction = A
    return y_prediction


def plot_learning_curve(costs):
    plt.figure(2)
    
    plt.plot(costs)
    
    plt.ylabel('cost')
    y_ticks = np.linspace(0, 0.8, 17)  # 11 ticks from -1 to 1

    plt.yticks(y_ticks)

    plt.grid(True, which='both')
    plt.xlabel('iterations')
    plt.title("Learning curve")
    plt.savefig("cost_curve.png")

def plot_accuracy_curve(test_accuracy_array, train_accuracy_array):
    sze = len(test_accuracy_array)
    plt.figure(3)
    
    plt.plot(np.arange(1, sze+1), test_accuracy_array, 'r--', np.arange(1, sze+1), train_accuracy_array)
    plt.ylabel('test accuracy')
    plt.grid(True, which='both')
    plt.xlabel('iterations')
    plt.savefig("test_accuracy_curve.png")

