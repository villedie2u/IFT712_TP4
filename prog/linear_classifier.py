import numpy as np


class LinearClassifier(object):
    def __init__(self, x_train, y_train, x_val, y_val, num_classes, bias=False):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.bias = bias  # when bias is True then the feature vectors have an additional 1

        num_features = x_train.shape[1]
        if bias:
            num_features += 1

        self.num_features = num_features
        self.num_classes = num_classes
        self.W = self.generate_init_weights(0.01)

    def generate_init_weights(self, init_scale):
        return np.random.randn(self.num_features, self.num_classes) * init_scale

    def train(self, num_epochs=1, lr=1e-3, l2_reg=1e-4, lr_decay=1.0, init_scale=0.01):
        """
        Train the model with a cross-entropy loss
        Naive implementation (with loop)

        Inputs:
        - num_epochs: the number of training epochs
        - lr: learning rate
        - l2_reg: the l2 regularization strength
        - lr_decay: learning rate decay.  Typically a value between 0 and 1
        - init_scale : scale at which the parameters self.W will be randomly initialized

        Returns a tuple for:
        - training accuracy for each epoch
        - training loss for each epoch
        - validation accuracy for each epoch
        - validation loss for each epoch
        """
        loss_train_curve = []
        loss_val_curve = []
        accu_train_curve = []
        accu_val_curve = []

        self.W = self.generate_init_weights(init_scale)  # type: np.ndarray

        sample_idx = 0
        num_iter = num_epochs * len(self.x_train)
        for i in range(num_iter):
            # Take a sample
            x_sample = self.x_train[sample_idx]
            y_sample = self.y_train[sample_idx]
            if self.bias:
                x_sample = augment(x_sample)

            # Compute loss and gradient of loss
            loss_train, dW = self.cross_entropy_loss(x_sample, y_sample, reg=l2_reg)

            # Take gradient step
            # print("\t\t", y_sample, np.dot(self.W, x_sample), end="->")
            self.W -= lr * dW
            # print(np.dot(self.W, x_sample))

            # Advance in data
            sample_idx += 1
            if sample_idx >= len(self.x_train):  # End of epoch

                accu_train, loss_train = self.global_accuracy_and_cross_entropy_loss(self.x_train, self.y_train, reg=l2_reg)
                accu_val, loss_val, = self.global_accuracy_and_cross_entropy_loss(self.x_val, self.y_val, reg=l2_reg)

                loss_train_curve.append(loss_train)
                loss_val_curve.append(loss_val)
                accu_train_curve.append(accu_train)
                accu_val_curve.append(accu_val)

                sample_idx = 0
                lr *= lr_decay

        return loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve

    def predict(self, X):
        """
        return the class label with the highest class score i.e.

            argmax_c W.X

         X: A numpy array of shape (D,) containing one or many samples.

         Returns a class label for each sample (a number between 0 and num_classes-1)
        """
        class_label = np.zeros(X.shape[0])
        #############################################################################
        # TODO: Return the best class label.                                        #
        #############################################################################
        indice = 0
        for xi in X:
            if len(xi) != 3:
                xi = augment(xi)
            class_label[indice] = np.argmax(np.dot(self.W, xi))
            indice += 1

        
        
        
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return class_label

    def global_accuracy_and_cross_entropy_loss(self, X, y, reg=0.0):
        """
        Compute average accuracy and cross_entropy for a series of N data points.
        Naive implementation (with loop)
        Inputs:
        - X: A numpy array of shape (D, N) containing many samples.
        - y: A numpy array of shape (N) labels as an integer
        - reg: (float) regularization strength
        Returns a tuple of:
        - average accuracy as single float
        - average loss as single float
        """
        accu = 0
        loss = 0
        #############################################################################
        predictions = self.predict(X)
        for i in range(len(predictions)):
            if predictions[i] == y[i]:
                accu += 1
            loss += self.cross_entropy_loss(X[i], y[i], reg=0)[0]  # pour ne pas ajouter le terme de régularisation sur chaque terme

        accu /= len(y)
        loss /= len(y)
        loss += reg*((np.linalg.norm(self.W))**2)  # pour ajouter la régularisation une fois que la somme est faite

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return accu, loss

    def cross_entropy_loss(self, x, y, reg=0.0):
        """
        Cross-entropy loss function for one sample pair (X,y) (with softmax)
        C.f. Eq.(4.104 to 4.109) of Bishop book.

        Input have dimension D, there are C classes.
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - x: A numpy array of shape (D,) containing one sample.
        - y: training label as an integer
        - reg: (float) regularization strength
        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)

        #############################################################################
        # TODO: Compute the softmax loss and its gradient.                          #
        # Store the loss in loss and the gradient in dW.                            #
        # 1- Compute softmax => eq.(4.104) or eq.(5.25) Bishop                      #
        # 2- Compute cross-entropy loss => eq.(4.108)                               #
        # 3- Dont forget the regularization!                                        #
        # 4- Compute gradient => eq.(4.109)                                         #
        #############################################################################
        
        # 1 - Softmax
        if len(x) != 3:
            x = augment(x)
        a = np.dot(self.W.T, x)
        y0 = []
        
        sum_a = 0
        for aj in a:
            if aj >= 250:  # pour éviter les overflow
                aj = 250
            sum_a += np.exp(aj)
        for k in range(a.shape[0]):
            if sum_a == 0:  # pour éviter de diviser par 0
                sum_a = 0.001
            if a[k] >= 250:  # pour éviter les overflow
                a[k] = 250
            y0.append(np.exp(a[k])/sum_a)

        #2 - Cross-entropy Loss    
        if y0[y] == 0:  # pour éviter les divisions par 0
            y0[y] = 1e-5
        loss = - np.log(y0[y])
        
        # 3 - Regularisation
        regularization = reg * ((np.linalg.norm(self.W))**2)
        loss += regularization
        
        # 4 - Compute gradient
        if len(x) != 3:
            x = augment(x)
        for i in range(0, dW.shape[1]):
            if i == y:
                dW[i, :] = (y0[i] - 1) * x
            else:
                dW[i, :] = y0[i] * x
        
        dW += 2*reg*dW  # régularisation selon la formule p.4 du "kit de survie"
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return loss, dW


def augment(x):
    if len(x.shape) == 1:
        return np.concatenate([x, [1.0]])
    else:
        return np.concatenate([x, np.ones((len(x), 1))], axis=1)
