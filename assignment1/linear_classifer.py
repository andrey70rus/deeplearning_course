import numpy as np
from gradient_check import check_gradient


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    predictions = predictions.copy()
    predictions = predictions.astype(float)

    max_vals = np.max(predictions, axis=-1)
    predictions -= np.full_like(predictions.T, max_vals).T

    sum_exp_vals = np.sum(np.exp(predictions), axis=-1)
    probs = np.exp(predictions) / np.full_like(predictions.T, sum_exp_vals).T

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops

    loss = - np.sum(target_index * np.log(probs))

    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    Eps = 1e-14

    if not isinstance(target_index, np.ndarray):
        target_index = np.array([target_index])
    if len(predictions.shape) == 1:
        predictions = predictions[None, :]

    num_batches = len(predictions)
    probs = np.zeros(predictions.shape)
    loss = np.zeros(predictions.shape)
    dprediction = np.zeros(predictions.shape)

    probs += softmax(predictions)  # softmax
    probs = np.clip(probs, 0+Eps, 1-Eps)

    for idx in range(num_batches):
        loss[idx] = - np.sum(1 * np.log(probs[idx][target_index[idx]])) / probs[idx].shape[0]  # cross-entropy

        dprediction[idx] = probs[idx].copy()
        dprediction[idx][target_index[idx]] -= 1

    loss = np.sum(loss) / num_batches
    dprediction /= num_batches

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # l2_reg_loss = regularization_strength * sumij W[i, j]2

    loss = reg_strength * np.sum(W ** 2)

    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops

    loss, dpred = softmax_with_cross_entropy(predictions, target_index)

    y_true = np.zeros_like(predictions)

    # TODO избавиться от цикла
    for idx_row, idx_val in enumerate(target_index):
        y_true[idx_row][idx_val] += 1

    prob = softmax(predictions)
    gradient = np.dot(X.T, prob - y_true) / X.shape[0]

    return loss, gradient


class LinearSoftmaxClassifier:
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!

            losses = []
            for batch_idxs in batches_indices:
                loss_batches, gradient = linear_softmax(
                    X[batch_idxs], self.W, y[batch_idxs]
                )

                # for sample_idx in range(len(batch_idxs)):
                loss_l2, gradient_l2 = l2_regularization(self.W, reg)

                loss_batches += loss_l2
                gradient += gradient_l2

                losses.append(loss_batches)
                self.W -= learning_rate * gradient

            loss = np.mean(losses)
            loss_history.append(loss)

            #end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        y_pred += np.argmax(np.dot(X, self.W), axis=-1)

        return y_pred


if __name__ == '__main__':
    # def prepare_for_linear_classifier(train_X, test_X):
    #     train_flat = train_X.reshape(train_X.shape[0], -1).astype(np.float) / 255.0
    #     test_flat = test_X.reshape(test_X.shape[0], -1).astype(np.float) / 255.0
    #
    #     # Subtract mean
    #     mean_image = np.mean(train_flat, axis=0)
    #     train_flat -= mean_image
    #     test_flat -= mean_image
    #
    #     # Add another channel with ones as a bias term
    #     train_flat_with_ones = np.hstack([train_flat, np.ones((train_X.shape[0], 1))])
    #     test_flat_with_ones = np.hstack([test_flat, np.ones((test_X.shape[0], 1))])
    #     return train_flat_with_ones, test_flat_with_ones
    #
    # from dataset import load_svhn, random_split_train_val
    #
    # train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)
    # train_X, test_X = prepare_for_linear_classifier(train_X, test_X)
    # # Split train into train and val
    # train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val=1000)
    #
    # train_X = np.array(
    #     [
    #         [244, 250, 255, 250], [0, 0, 0, 10], [0, 10, 20, 0], [0, 0, 0, 0],
    #         [20, 0, 10, 10], [10, 0, 0, 0], [244, 250, 255, 250], [0, 0, 0, 0],
    #         [0, 0, 0, 0], [244, 250, 255, 250], [244, 250, 255, 250], [244, 250, 255, 250],
    #         [244, 250, 255, 250], [0, 0, 0, 10], [0, 10, 20, 0], [0, 0, 0, 0],
    #         [20, 0, 10, 10], [10, 0, 0, 0], [244, 250, 255, 250], [0, 0, 0, 0],
    #         [0, 0, 0, 0], [244, 250, 255, 250], [244, 250, 255, 250], [244, 250, 255, 250],
    #         [244, 250, 255, 250], [0, 0, 0, 10], [0, 10, 20, 0], [0, 0, 0, 0],
    #         [20, 0, 10, 10], [10, 0, 0, 0], [244, 250, 255, 250], [0, 0, 0, 0],
    #         [0, 0, 0, 0], [244, 250, 255, 250], [244, 250, 255, 250], [244, 250, 255, 250],
    #     ]
    # )
    # train_y = np.array(
    #     [
    #         1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
    #         1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
    #         1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
    #     ]
    # )
    #
    # train_X, _ = prepare_for_linear_classifier(train_X, train_X)
    #
    # classifier = LinearSoftmaxClassifier()
    # loss_history = classifier.fit(train_X, train_y, epochs=10, learning_rate=1e-3, batch_size=10, reg=1e1)
    #
    # classifier.predict(train_X)

    batch_size = 2
    num_classes = 2
    num_features = 3
    np.random.seed(42)
    W = np.random.randint(-1, 3, size=(num_features, num_classes)).astype(np.float)
    X = np.random.randint(-1, 3, size=(batch_size, num_features)).astype(np.float)
    target_index = np.ones(batch_size, dtype=np.int)

    loss, dW = linear_softmax(X, W, target_index)
    check_gradient(lambda w: linear_softmax(X, w, target_index), W)
