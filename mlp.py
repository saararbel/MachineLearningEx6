import numpy as np
import math

BIAS = 1.0


def init_mlp(inputs, targets, nhidden):
    """ Initialize network """

    # Set up network size
    nin = np.shape(inputs)[1]
    nout = np.shape(targets)[1]
    ndata = np.shape(inputs)[0]
    nhidden = nhidden

    # Initialize network
    weights1 = (np.random.rand(nin+1, nhidden)-0.5)*2/np.sqrt(nin)
    weights2 = (np.random.rand(nhidden+1, nout)-0.5)*2/np.sqrt(nhidden)

    return weights1, weights2


def loss_and_gradients(input_x, expected_output_y, weights1, weights2):
    """compute loss and gradients for a given x,y

    this function gets an (x,y) pair as input along with the weights of the mlp,
    computes the loss on the given (x,y), computes the gradients for each weights layer,
    and returns a tuple of loss, weights 1 gradient, weights 2 gradient.
    The loss should be calculated according to the loss function presented in the assignment

    Arguments:
        input_x {numpy 1d array} -- an instance from the dataset
        expected_output_y {scalar} -- the ground truth
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer

    Returns:
        tuple -- loss, weights 1 gradient, weights 2 gradient, and activations[-1] which is y_hat
    """
    # Initialize gradients
    weights1_gradient, weights2_gradient = np.zeros(weights1.shape), np.zeros(weights2.shape)
    weighted_outputs, activations = mlpfwd(input_x, weights1, weights2)
    y_hat = activations[-1]
    loss = calc_loss(expected_output_y, y_hat)

    weights2_error = np.zeros(weights2.shape)
    calc_gradient_w2(activations, expected_output_y, weights2_error, weights2_gradient,
                     y_hat)
    calc_gradient_w1(activations, input_x, weights1_gradient, weights2, weights2_error)

    return loss, weights1_gradient, weights2_gradient, y_hat


def calc_gradient_w1(activations, input_x, weights1_gradient, weights2, weights2_error):
    for i, activation in enumerate(activations[0]):
        error = activation * (1 - activation) * sum(weights2_error[i] * weights2[i])
        for j, xj in enumerate(input_x):
            weights1_gradient[j][i] += xj * error


def calc_gradient_w2(activations, expected_output_y, weights2_error, weights2_gradient, y_hat):
    for i, output_neuron in enumerate(expected_output_y):
        error = (y_hat[i] - output_neuron) * y_hat[i] * (1-y_hat[i])
        for j, weights_to_output_neuron in enumerate(add_bias_to_activations(activations[0])):
            weights2_error[j][i] += error
            weights2_gradient[j][i] += weights_to_output_neuron * calc_error(i, output_neuron, y_hat)


def calc_error(i, output_neuron, y_hat):
    return (y_hat[i] - output_neuron) * y_hat[i] * (1 - y_hat[i])


def add_bias_to_activations(activation):
    return np.append(activation, np.ones(BIAS))


def calc_weights1_gradient(true_output, weighted_outputs, input_x, weights1_gradient, weights2, activations):
    for i, input_neuron in enumerate(weights1_gradient):
        for j, w in enumerate(input_neuron):
            w2 = weights2[i]
            weights1_gradient[i][j] = calc_new_weight1_via_gradient(true_output, weights2[i], weighted_outputs,
                                                                    j, input_x[i])


def calc_new_weight1_via_gradient(true_output, w2, activations, hidden_neuron_index, input):
    outz = activations[1][0]
    outh1 = activations[0][hidden_neuron_index]
    return np.asarray(-1 * (true_output - outz) * outz * (1 - outz) * w2 * outh1 * (1 - outh1) * input)


def calc_weights2_gradient(true_output, weighted_outputs, weights2_gradient, activations):
    for i, outh in enumerate(activations[0]):
        weights2_gradient[i] = calc_new_weight2_via_gradient(true_output, outh, activations)


def calc_new_weight2_via_gradient(true_output, outh, activations):
    outz = activations[1][0]
    return np.asarray(-1 * (true_output - outz) * outz * (1 - outz) * outh)


def calc_loss(true_outputs, temp_outputs):
    sum = 0.0
    for i, output in enumerate(true_outputs):
        sum += math.pow(output - temp_outputs[i], 2)

    return 0.5 * sum


def mlpfwd(input_x, weights1, weights2):
    """feed forward
    
    this function gets an input x and feeds it through the mlp.
    
    Arguments:
        input_x {numpy 1d array} -- an instance from the dataset
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer
    
    Returns:
        tuple -- list of weighted outputs along the way, list of activations along the way:
        
        1) The first part of the tuple consists of a list, where every item in the list
        holds the values of a layer in the network, before the activation function has been applied
        on it. The value of a layer in the network is the weighted sum of the layer before it.
        
        2) The second part of     the tuple consists of a list, where every item in the list holds
        the values of a layer in the network, after the activation function has been applied on it.
        Don't forget to add the bias to a layer, when required.
    """

    weighted_outputs, activations = [], []
    out = get_output_array(weights1, input_x)
    weighted_outputs.append(out)
    hidden_activations = [sigmoid(output) for output in out]
    activations.append(hidden_activations)

    new_input = hidden_activations + [BIAS]
    new_out = get_output_array(weights2, new_input)
    weighted_outputs.append(new_out)
    activations.append([sigmoid(output) for output in new_out])

    return weighted_outputs, activations


def get_output_array(weights, input):
    out = [0] * len(weights[0])
    for input_index, input_neuron_weights in enumerate(weights):
        for neuron_index, weight in enumerate(input_neuron_weights):
            out[neuron_index] += (weight * input[input_index])

    return out


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def accuracy_on_dataset(inputs, targets, weights1, weights2):
    """compute accuracy
    
    this function gets a dataset and returns model's accuracy on the dataset.
    The accuracy is calculated using a threshold of 0.5:
    if the prediction is >= 0.5 => y_hat = 1
    if the prediction is < 0.5 => y_hat = 0
    
    Arguments:
        inputs {numpy 2d array} -- instances
        targets {numpy 2d array} -- ground truths
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer

    Returns:
        scalar -- accuracy on the given dataset
    """

    num_iter, correct_hits = 0,0
    for i, input_x in enumerate(inputs):
        weights_output, activations = mlpfwd(input_x, weights1, weights2)
        for j, expected_j_output in enumerate(targets[i]):
            num_iter += 1
            if expected_j_output == round(activations[-1][j]):
                correct_hits += 1

    return float(correct_hits) / num_iter


def mlptrain(inputs, targets, eta, nepochs, weights1, weights2):
    """train the model
    
    Arguments:
        inputs {numpy 2d array} -- instances
        targets {numpy 2d array} -- ground truths
        eta {scalar} -- learning rate
        nepochs {scalar} -- number of epochs
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer
    """
    ndata = np.shape(inputs)[0]
    # Add the inputs that match the bias node
    inputs = np.concatenate((inputs, np.ones((ndata, 1))), axis=1)

    for n in range(nepochs):
        epoch_loss = 0
        predictions = []
        for ex_idx in range(len(inputs)):
            x = inputs[ex_idx]
            y = targets[ex_idx]

            # compute gradients and update the mlp
            loss, weights1_gradient, weights2_gradient, y_hat = loss_and_gradients(x, y, weights1, weights2)
            weights1 -= eta * weights1_gradient
            weights2 -= eta * weights2_gradient
            epoch_loss += loss
            predictions.append(y_hat)

        if (np.mod(n, 100) == 0):
            print n, epoch_loss, accuracy_on_dataset(inputs, targets, weights1, weights2)

    return weights1, weights2
