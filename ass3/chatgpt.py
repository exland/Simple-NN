ni = 2
nh = 12
no = 3
d_act = [d_sigmoid, d_softmax]
activations = [sigmoid, softmax]

W,b = init_params(ni=ni, nh=nh, no=no)

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



for it in range(max_it):
    
    # iterate over all smaples and sum up gradient
    dW = [0, 0]
    db = [0, 0]
    for xi, yi in zip(x_train, y_train):
        yi = one_hot(yi)
        ai, zi = feed_forward(xi, W, b, activations)

        dWi, dbi, delta = back_prop(yi, W, b, d_act, ai, zi)
        
        pred = np.array(ai[-1]).argmax()

        L[it] += loss(ai[-1], yi) / n
    
        # sum up gradients
        for idx in range(2):
            dW[idx] += (dWi[idx] / n)
            db[idx] += (dbi[idx] / n)
    


# do gradient step
    for idx in range(2):
        W[idx] = W[idx] - alpha * dW[idx]
        b[idx] = b[idx] - alpha * db[idx]