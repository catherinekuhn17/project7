# BMI 203 Project 7: Neural Network

# Import necessary dependencies here


# TODO: Write your test functions and associated docstrings below.


def test_single_forward_and forward():
    '''
    tests is single and full forward work as expected
    '''
    
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


def test_single_backprop_predict():
    '''
    tests if backprop and prediction are as expected
    '''
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
    
    pred = test_nn.predict(np.array([2,1,5]))
    assert pred[0][0] == 0



def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    '''
    tests to see if bce backprop is working as expected
    '''
    test_nn = nn.NeuralNetwork([{'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                     lr=3, 
                     seed = 10,
                     batch_size = 2,
                     epochs = 10, 
                     loss_function='bce')
    assert sum(abs(test_nn._binary_cross_entropy_backprop(np.array([0,1]), np.array([.3,.4])) - np.array([1.42857143, -2.5]))) < .000001


def test_mean_squared_error_and backprop():
    
    test_nn = nn.NeuralNetwork([{'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}], 
                 lr=3, 
                 seed = 10,
                 batch_size = 2,
                 epochs = 10, 
                 loss_function='bce')
    mse = test_nn._mean_squared_error(np.array([0,1]), np.array([.3,.4]))
    
    assert (mse-0.22499999)<.000001
    
    mse_b = test_nn._mean_squared_error_backprop(np.array([0,1]), np.array([.3,.4]))
    
    assert mse_b == np.array([ 0.6, -1.2]).all()


def test_one_hot_encode():
    pass


def test_sample_seqs():
    pass
