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

    #print(f'Training set with {x_train.shape[0]} data samples.')
    #print(f'Test set with {x_test.shape[0]} data samples.')

    # plot training loss/accuracy
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].set_title('Training loss')
    ax1[1].set_title('Training accuracy')
    ax1[0].legend()
    ax1[1].legend()

    fig2, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10,3.5))
    ax[0].scatter(x_train[:,0],x_train[:,1],c=y_train), ax[0].set_title('Training Data'), ax[0].set_aspect('equal', 'box')
    ax[0].set_title('Training Data'), ax[0].set_aspect('equal', 'box')
    ax[1].set_title('Decision BD (constant)'), ax[1].set_aspect('equal', 'box')
    ax[2].set_title('Decision BD (Armijo)'), ax[2].set_aspect('equal', 'box')

    lam = 1e-3
    fig3, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10,3.5))
    plt.suptitle(r'Regularization $\lambda$=%.6f' %lam)
    ax[0].scatter(x_train[:,0],x_train[:,1],c=y_train), ax[0].set_title('Training Data'), ax[0].set_aspect('equal', 'box')
    ax[1].set_title('Decision BD (constant)'), ax[1].set_aspect('equal', 'box')
    ax[2].set_title('Decision BD (Armijo)'), ax[2].set_aspect('equal', 'box')


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

    def d_softmax(x):
        print("im here")
        return 1

    def loss(y, y_star):
        l = -np.sum(y_star * np.log(y))
        return l

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
    # compute deltas in reversed order
        assert(len(a) == len(z))
        delta = [None] * len(a)
        delta[-1] = d_loss(a[-1], y_star) # delta_L
        for l in range(len(a) - 2, 0, -1):
            delta[l] = d_act[l-1](z[l]) * W[l].T.dot(delta[l+1])
                                
        # compute gradient in W an b
        dW = [None] * len(a)
        db = [None] * len(a)
        for l in range(len(a) - 1):
            dW[l] = np.outer(delta[l+1], a[l])
            db[l] = delta[l+1]
            
        return dW, db, delta
    
    
    def f(x, ipt, y_star, activations):
        W0 = x[:8].reshape(4,2)
        W1 = x[8:20].reshape(3,4)
        b0 = x[20:24]
        b1 = x[24:]
        
        W = [W0, W1]
        b = [b0, b1]
        a, z = feed_forward(ipt, W, b, activations)
        L = loss(a[-1], y_star)
        return L

    ni = 2
    nh = 4
    no = 3
    d_act = [d_sigmoid, d_softmax]
    activations = [sigmoid, softmax]
    
    y_star = np.array([0,0,1]).T
    W,b = init_params(ni=ni, nh=nh, no=no)
    
    print(W[0].shape)
    print(W[1].shape)
    print(b[0].shape)
    print(b[1].shape)
    # a, z = feed_forward(x_train[0], W, b, activations)
    
    # dW, db, delta = back_prop(y_star, W, b, d_act, a, z)
    # #dW too small?
    # x_init = np.concatenate((W[0].ravel(), W[1].ravel(), b[0].ravel(), b[1].ravel()))
    # grads = approx_fprime(x_init, f, 1e-6, x_train[0], y_star, activations)
    # dW0 = grads[:8].reshape(4,2)
    # dW1 = grads[8:20].reshape(3,4)
    # db0 = grads[20:24]
    # db1 = grads[24:]
    # g1,g2,g3,g4 = np.allclose(dW[0], dW0), np.allclose(dW[1], dW1), np.allclose(db[0], db0), np.allclose(db[1], db1)
    # print(dW0)
    # print(dW[0])
    # print(g1,g2,g3,g4)
    n = len(x_train)
    alpha = 0.3
    max_it = 5000
    L = np.zeros(max_it)

    def one_hot(yi):
        if yi == 0:
            return np.array(np.array([1,0,0]).T)
        if yi == 1:
            return np.array(np.array([0,1,0]).T)
        return np.array(np.array([0,0,1]).T)
    

    x_accuracy = np.zeros(max_it)
    y_accuracy = np.zeros(max_it)
    for it in range(max_it):
        
        # iterate over all smaples and sum up gradient
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

            L[it] += loss(ai[-1], yi) / n
        
            # sum up gradients
            for idx in range(2):
                dW[idx] += (dWi[idx] / n)
                db[idx] += (dbi[idx] / n)
        
        x_accuracy[it] = it
        y_accuracy[it] = correct/len(x_train)

        if it % 50 == 0:
            print ('it=', it, 'loss=', L[it])
    # do gradient step constant
        # for idx in range(2):
        #     W[idx] = W[idx] - alpha * dW[idx]
        #     b[idx] = b[idx] - alpha * db[idx]
    
    #armijo
        sigma = 0.0001
        beta = 0.5  
        mk = 0
        alpha = 10
        alpha_armijo = 0

    
        W_proposed = [0, 0]
        b_propposed = [0, 0]
        loss_proposed = 0
        while True:
            loss_proposed = 0
            alpha_armijo = alpha*(beta**mk)
            for idx in range(2):
                W_proposed[idx] = W[idx] - alpha_armijo * dW[idx]
                b_propposed[idx] = b[idx] - alpha_armijo * db[idx]
            for xi, yi in zip(x_train, y_train):
                yi = one_hot(yi)
                ai, zi = feed_forward(xi, W_proposed, b_propposed, activations)
                loss_proposed += loss(ai[-1], yi) / n
            df_dx = 0
            for idx in range(2):
                df_dx += np.sum(dW[idx] * dW[idx]) + np.sum(db[idx] * db[idx])
            
            #print(L[it] - loss_proposed, "-----", sigma*alpha_armijo*df_dx)
            
            if(L[it] - loss_proposed >= sigma*alpha_armijo*df_dx):
                break
            mk+=1


        W = W_proposed
        b = b_propposed

    print ('it=', it, 'loss=', L[it])

    np.savetxt("w1.csv", W[0], delimiter=",")
    np.savetxt("w2.csv", W[1], delimiter=",")
    np.savetxt("b1.csv", b[0], delimiter=",")
    np.savetxt("b2.csv", b[1], delimiter=",")

    W0 = np.loadtxt("w1.csv", delimiter=',')
    W1 = np.loadtxt("w2.csv", delimiter=',')
    b0 = np.loadtxt("b1.csv", delimiter=',')
    b1 = np.loadtxt("b2.csv", delimiter=',')

    W = [W0,W1]
    b = [b0,b1]

    correct = 0
    false = 0

    for xi, yi in zip(x_test, y_test):
        a ,z = feed_forward(xi, W, b, activations)
        a = np.array(a)
        pred = a[-1].argmax()
        #print(pred, yi)
        if pred == yi:
            correct+= 1
        else:
            false += 1
    
    print(correct)
    print(false)
    ax1[0].plot(x_accuracy,L, linewidth=2.0)
    ax1[1].plot(x_accuracy,y_accuracy, linewidth=2.0)
    ax1[0].set_yscale('log')
    ax1[1].set_yscale('log')

    """ End of your code
    """

    return [fig1, fig2, fig3]

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')
    figures = task()
    for fig in figures:
            pdf.savefig(fig)
    pdf.close()
