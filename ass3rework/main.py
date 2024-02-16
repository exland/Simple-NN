""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import approx_fprime
from typing import Callable

np.random.seed(0)

def task():
    """ Neural Network Training
        Requirements for the plots:
            - fig1 (make sure that all curves include labels)
                - ax[0] logarithmic plot for training loss with constant step size and Armijo backtracking (include a label!)
                - ax[1] plot of training accuracy with constant step size and Armijo backtracking (include a label!)
            - fig2
                - ax[0] already plots the training data
                - ax[1] for the training with constant step size, plot the predicted class for a dense meshgrid over the input data range
                - ax[2] make the same plot as in ax[1] for the training using Armijo backtracking
            - fig3: (bonus task), this should be the same as fig2 but with weight decay
    """

    # load data
    with np.load('data.npz') as data_set: 
            # get the training data
            x_train = data_set['x_train']
            y_train = data_set['y_train']

            # get the test data
            x_test = data_set['x_test']
            y_test = data_set['y_test']

    print(f'Training set with {x_train.shape[0]} data samples.')
    print(f'Test set with {x_test.shape[0]} data samples.')

    # plot training loss/accuracy
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].set_title('Training loss')
    ax1[1].set_title('Training accuracy')
    ax1[0].legend()
    ax1[1].legend()

    fig2, ax2 = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10,3.5))
    ax2[0].scatter(x_train[:,0],x_train[:,1],c=y_train), ax2[0].set_title('Training Data'), ax2[0].set_aspect('equal', 'box')
    ax2[0].set_title('Training Data'), ax2[0].set_aspect('equal', 'box')
    ax2[1].set_title('Decision BD (constant)'), ax2[1].set_aspect('equal', 'box')
    ax2[2].set_title('Decision BD (Armijo)'), ax2[2].set_aspect('equal', 'box')

    lam = 1e-3
    fig3, ax3 = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10,3.5))
    plt.suptitle(r'Regularization $\lambda$=%.6f' %lam)
    ax3[0].scatter(x_train[:,0],x_train[:,1],c=y_train), ax3[0].set_title('Training Data'), ax3[0].set_aspect('equal', 'box')
    ax3[1].set_title('Decision BD (constant)'), ax3[1].set_aspect('equal', 'box')
    ax3[2].set_title('Decision BD (Armijo)'), ax3[2].set_aspect('equal', 'box')
    
        ### Helper functions
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    def d_sigmoid(x):
        act = sigmoid(x)
        return act * (1 - act)

    def softmax(x):
        z = x - max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator/denominator
        return softmax

    def loss(y, y_star):
        return -np.sum(y_star * np.log(y))

    def d_loss(y, y_star): 
        return y - y_star

    def init_params(ni,nh,no):
        W = [np.random.uniform(-(1/np.sqrt(ni)),(1/np.sqrt(ni)),size=(nh,ni)),
             np.random.uniform(-(1/np.sqrt(nh)),(1/np.sqrt(nh)),size=(no,nh))]
        b = [np.random.uniform(-(1/np.sqrt(ni)),(1/np.sqrt(ni)),size=nh),
             np.random.uniform(-(1/np.sqrt(nh)),(1/np.sqrt(nh)),size=no)]
        return W,b

    def feed_forward(x, W, b, act):
        a = [x]
        z = [x]
        for i, (Wi, bi, acti) in enumerate(zip(W, b, act)):
            z.append(np.dot(Wi, a[i]) + bi)
            a.append(acti(z[-1]))
        return a, z

    def back_prop(y_star, W, b, d_act, a, z):
        assert(len(a) == len(z))
        delta = [None] * len(a)
        delta[-1] = d_loss(a[-1], y_star)
        for l in range(len(a) - 2, 0, -1):
            delta[l] = d_act[l-1](z[l]) * W[l].T.dot(delta[l+1])                                
        dW = [None] * len(a)
        db = [None] * len(a)
        for l in range(len(a) - 1):
            dW[l] = np.outer(delta[l+1], a[l])
            db[l] = delta[l+1]
        return dW, db, delta

    def f(x, ipt, y_star, activations):
        W0 = x[:24].reshape(12,2)
        W1 = x[24:60].reshape(3,12)
        b0 = x[60:72]
        b1 = x[72:]
        W = [W0, W1]
        b = [b0, b1]
        a, _ = feed_forward(ipt, W, b, activations)
        L = loss(a[-1], y_star)
        return L

    def one_hot(yi):
        if yi == 0:
            return np.array(np.array([1,0,0]).T)
        if yi == 1:
            return np.array(np.array([0,1,0]).T)
        return np.array(np.array([0,0,1]).T)

    activations = [sigmoid, softmax]
    d_act = [d_sigmoid]

    ######### NN
    ni = 2
    nh = 12
    no = 3
    # 1) Python 
    W,b = init_params(ni=ni, nh=nh, no=no)

    # 2 & 3 are implemented above

    # 4) Python
    def verify_gradients(sample_x, sample_y, W, b):
        a, z = feed_forward(sample_x, W, b, activations)
        dW, db, delta = back_prop(sample_y, W, b, d_act, a, z)

        x_init = np.concatenate((W[0].ravel(), W[1].ravel(), b[0].ravel(), b[1].ravel()))
        grads = approx_fprime(x_init, f, 1e-6, sample_x, sample_y, activations)

        dW0 = grads[:24].reshape(12,2)
        dW1 = grads[24:60].reshape(3,12)
        db0 = grads[60:72]
        db1 = grads[72:]
        g1,g2,g3,g4 = np.allclose(dW[0], dW0), np.allclose(dW[1], dW1), np.allclose(db[0], db0), np.allclose(db[1], db1)
        
        return g1 == True and g2 == True and g3 == True and g4 == True

    assert(verify_gradients(x_train[0], one_hot(y_train[0]), W, b) == True)
    

    def armijo(W, b, dW, db, sigma, beta, initial_alpha, current_loss, activations, training_sample_size):
        W_next = [0, 0]
        b_next = [0, 0]
        loss_next = 0
        mk = 0
        alpha = 0
        while True:
            loss_next = 0
            alpha = initial_alpha * (beta ** mk)
            for idx in range(2):
                W_next[idx] = W[idx] - alpha * dW[idx]
                b_next[idx] = b[idx] - alpha * db[idx]
            for xi, yi in zip(x_train, y_train):
                yi = one_hot(yi)
                ai, zi = feed_forward(xi, W_next, b_next, activations)
                loss_next += loss(ai[-1], yi) / training_sample_size
            gradients = 0
            for idx in range(2):
                gradients += np.sum(dW[idx] * dW[idx]) + np.sum(db[idx] * db[idx])

            if(current_loss - loss_next >= sigma * alpha * gradients):
                break
            mk += 1
        
        return alpha

    def accuracy(test_data, truth_val, weights, biases):
        correct = 0 
        for xi, yi in zip(test_data, truth_val):
            truth = yi
            yi =  one_hot(yi)
            ai, zi =  feed_forward(xi,weights, biases, activations)
            pred = np.array(ai[-1]).argmax()
            if pred == truth:
                correct += 1
        
        return len(test_data) - correct, correct / len(x_test)

    def plot_decision_boundaries(weights, biases, subplot, index):
        x_min = x_train[:,0].min()
        x_max = x_train[:,0].max()
        y_min = x_train[:,1].min()
        y_max = x_train[:,1].max()

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        
        it = 0
        all_preds = np.zeros(len(np.c_[xx.ravel(), yy.ravel()]))
        for xi in zip(np.c_[xx.ravel(), yy.ravel()]):
            ai, zi =  feed_forward(xi[0],Wloaded, bloaded, activations)
            pred = np.array(ai[-1]).argmax()
            all_preds[it] = (pred)
            it += 1
            
        Z = all_preds.reshape(xx.shape)

        subplot[index].contourf(xx, yy, Z)

    def learn(training_sample_size, alpha, max_it, armijo_stepsize, weight_decay, W, b, x_train, y_train, activations, d_act):
        L = np.zeros(max_it)

        x_accuracy = np.zeros(max_it)
        y_accuracy = np.zeros(max_it)

        for it in range(max_it):
            dW = [0, 0]
            db = [0, 0]
            correct = 0
            for xi, yi in zip(x_train, y_train):
                truth = yi
                yi = one_hot(yi)
                ai, zi = feed_forward(xi, W, b, activations)

                dWi, dbi, delta = back_prop(yi, W, b, d_act, ai, zi)
                
                pred = np.array(ai[-1]).argmax()

                if pred == truth:
                    correct += 1

                L[it] += loss(ai[-1], yi) / training_sample_size
            
                # sum up gradients
                for idx in range(2):
                    dW[idx] += (dWi[idx] / training_sample_size)
                    db[idx] += (dbi[idx] / training_sample_size)
            
            x_accuracy[it] = it
            y_accuracy[it] = correct/len(x_train)

            if it % 1000 == 0:
                print ('it=', it, 'loss=', L[it])
            
            if armijo_stepsize:
                alpha = armijo(W, b, dW, db, 0.0001,  0.5, 10, L[it], activations, training_sample_size)

            if weight_decay:
                for idx in range(2):
                    dW[idx] += 0.001 * W[idx]

            for idx in range(2):
                W[idx] = W[idx] - alpha * dW[idx]
                b[idx] = b[idx] - alpha * db[idx]

        print ('it=', it, 'loss=', L[it])

        return L, x_accuracy, y_accuracy
    
    #!! deep copy

    def deepcopy_parameters(W0, W1, b0, b1):
        originalW0 = np.array(W[0])
        originalW1 = np.array(W[1])
        originalb0 = np.array(b[0])
        originalb1 = np.array(b[1])
        originalW = [originalW0, originalW1]
        originalb = [originalb0, originalb1]

        return originalW, originalb

    Wc, bc = deepcopy_parameters(W[0], W[1], b[0], b[1])
    Wa, ba = deepcopy_parameters(W[0], W[1], b[0], b[1])
    Wcw, bcw = deepcopy_parameters(W[0], W[1], b[0], b[1])
    Waw, baw = deepcopy_parameters(W[0], W[1], b[0], b[1])

    print("Learning...")

    W0 = np.loadtxt("w1c.csv", delimiter=',')
    W1 = np.loadtxt("w2c.csv", delimiter=',')
    b0 = np.loadtxt("b1c.csv", delimiter=',')
    b1 = np.loadtxt("b2c.csv", delimiter=',')
    L = np.loadtxt("L_c.csv", delimiter=",")
    x_accuracy = np.loadtxt("x_c.csv", delimiter=",")
    y_accuracy = np.loadtxt("y_c.csv", delimiter=",")

    Wloaded = [W0,W1]
    bloaded = [b0,b1]
    
    # Plot constant step size loss & accuracy
    ax1[0].plot(x_accuracy,L, linewidth=2.0)
    ax1[1].plot(x_accuracy,y_accuracy, linewidth=2.0)

    test_error, test_accuracy = accuracy(x_test, y_test, Wloaded, bloaded)
    print("Loss: ", L[-1])
    print("Constant error: ", test_error)
    print("Constant accuracy: ", test_accuracy)

    plot_decision_boundaries(Wloaded, bloaded, ax2, 1)

    W0 = np.loadtxt("w1a.csv", delimiter=',')
    W1 = np.loadtxt("w2a.csv", delimiter=',')
    b0 = np.loadtxt("b1a.csv", delimiter=',')
    b1 = np.loadtxt("b2a.csv", delimiter=',')
    L = np.loadtxt("L_a.csv", delimiter=",")
    x_accuracy = np.loadtxt("x_a.csv", delimiter=",")
    y_accuracy = np.loadtxt("y_a.csv", delimiter=",")

    Wloaded = [W0,W1]
    bloaded = [b0,b1]

    # Plot armijo backtracking loss & accuracy
    ax1[0].plot(x_accuracy,L, linewidth=2.0)
    ax1[1].plot(x_accuracy,y_accuracy, linewidth=2.0)

    test_error, test_accuracy = accuracy(x_test, y_test, Wloaded, bloaded)
    print("Loss: ", L[-1])
    print("Armijo error: ", test_error)
    print("Armijo accuracy: ", test_accuracy)

    plot_decision_boundaries(Wloaded, bloaded, ax2, 2)

    W0 = np.loadtxt("w1cw.csv", delimiter=',')
    W1 = np.loadtxt("w2cw.csv", delimiter=',')
    b0 = np.loadtxt("b1cw.csv", delimiter=',')
    b1 = np.loadtxt("b2cw.csv", delimiter=',')

    Wloaded = [W0,W1]
    bloaded = [b0,b1]
    
    test_error, test_accuracy = accuracy(x_test, y_test, Wloaded, bloaded)
    print("Constant(Weight Decay) error: ", test_error)
    print("Constant(Weight Decay) accuracy: ", test_accuracy)
    
    plot_decision_boundaries(Wloaded, bloaded, ax3, 1)

    W0 = np.loadtxt("w1aw.csv", delimiter=',')
    W1 = np.loadtxt("w2aw.csv", delimiter=',')
    b0 = np.loadtxt("b1aw.csv", delimiter=',')
    b1 = np.loadtxt("b2aw.csv", delimiter=',')
    

    Wloaded = [W0,W1]
    bloaded = [b0,b1]

    test_error, test_accuracy = accuracy(x_test, y_test, Wloaded, bloaded)
    print("Armijo(Weigh Decay) error: ", test_error)
    print("Armijo(Weigh Decay) accuracy: ", test_accuracy)

    
    plot_decision_boundaries(Wloaded, bloaded, ax3, 2)

    ax1[0].set_yscale('log')
    ax1[1].set_yscale('log')

    ax1[0].legend(["constant", "armijo"], loc='upper left')
    ax1[1].legend(["constant", "armijo"], loc='upper left')
    
    return [fig1, fig2, fig3]

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')
    figures = task()
    for fig in figures:
            pdf.savefig(fig)
    pdf.close()
