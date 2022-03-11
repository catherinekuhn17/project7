# BMI 203 Project 7: Neural Network

# Import necessary dependencies here


# TODO: Write your test functions and associated docstrings below.


def test_single_forward_and forward():
    
    test_nn = nn.NeuralNetwork([{'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                             lr=3, 
                             seed = 10,
                             batch_size = 2,
                             epochs = 10, 
                             loss_function='mse')
    W_curr = np.array([[1,1,2]])
    b_curr = np.array([[3]])
    A_prev = np.array([[2,1,4]])
    activation = 'relu'
    out = np.array(test_nn._single_forward(W_curr, b_curr, A_prev.T, activation)).flatten()
    
    assert (out == np.array([14,14])).all()
    
    output, cache = test_nn.forward(np.array([[1,2,3]]).T)
    
    assert abs(cache['Z1'][0][0] -  -0.18824402733359977)<.000001
    
    assert (cache['A0'].flatten()== np.array([1,2,3])).all() # which is what we just put in


def test_single_backprop():
    test_nn = nn.NeuralNetwork([{'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                         lr=3, 
                         seed = 10,
                         batch_size = 2,
                         epochs = 10, 
                         loss_function='mse')

    W_curr = np.array([[1,1,2]])
    b_curr = np.array([[3]])
    A_prev = np.array([[2,1,4]])
    Z_curr = np.array([[2,1,2]])
    dA_curr = np.array([[1]])

    out=test_nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, "relu")
    assert (out[0] == np.array([[1, 1, 1],
                                [1, 1, 1],
                                [2, 2, 2]])).all()
    assert abs(out[1][0][0]- 2.3333333)<.000001
    assert out[2][0][0] == 1

def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    pass


def test_sample_seqs():
    pass
