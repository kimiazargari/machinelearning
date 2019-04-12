import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        threshold = 0.0
        total_activation = nn.as_scalar(self.run(x))
        return 1 if total_activation >= threshold else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        num_mistakes = 1
        while num_mistakes > 0:
        	num_mistakes = 0
        	for x, y in dataset.iterate_once(batch_size=1):
        		prediction = self.get_prediction(x)
        		if (prediction != nn.as_scalar(y)):
        			self.w.update(x, nn.as_scalar(y)) # nn.as_scalar(y)
        			num_mistakes += 1

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.hidden_size = 100

        self.weights = nn.Parameter(1, self.hidden_size)
        self.bias = nn.Parameter(1, self.hidden_size)

        self.weights2 = nn.Parameter(self.hidden_size,1)
        self.bias2 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        data_x_weights = nn.Linear(x, self.weights)
        add_bias = nn.AddBias(data_x_weights, self.bias)
        activated = nn.ReLU(add_bias)
        #print(activated.data.shape)
        second = nn.Linear(activated, self.weights2)
        #print(second.ndim)
        add_second_bias = nn.AddBias(second, self.bias2)
        return add_second_bias


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # does prediction
        # uses predicted values to compute loss
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        for x, y in dataset.iterate_forever(batch_size=100):
        	loss = self.get_loss(x,y)
        	if nn.as_scalar(loss) < 0.018:
        		return
        	gradient_list = nn.gradients(self.get_loss(x, y), 
        		[self.weights, self.bias, self.weights2, self.bias2])
        	sign = -1
        	self.weights.update(gradient_list[0], self.learning_rate*sign)
        	self.bias.update(gradient_list[1], self.learning_rate*sign)
        	self.weights2.update(gradient_list[2], self.learning_rate*sign)
        	self.bias2.update(gradient_list[3], self.learning_rate*sign)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.06 #0.5
        self.hidden_size = 300 #300

        self.weights = nn.Parameter(784, self.hidden_size)
        self.bias = nn.Parameter(1, self.hidden_size)

        self.weights2 = nn.Parameter(self.hidden_size,10)
        self.bias2 = nn.Parameter(1,10)

        self.weights3 = nn.Parameter(self.hidden_size,10)
        self.bias3 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        data_x_weights = nn.Linear(x, self.weights)
        add_bias = nn.AddBias(data_x_weights, self.bias)
        activated = nn.ReLU(add_bias)
        #print(activated.data.shape)
        second = nn.Linear(activated, self.weights2)
        add_second_bias = nn.AddBias(second, self.bias2)
        #print(add_second_bias.data.shape)
        return add_second_bias

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        prediction = self.run(x) # a batch size x 10 
        return nn.SoftmaxLoss(prediction, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # if dataset.get_validation_accuracy() > .97:
        # 	return
        # if dataset.get_validation_accuracy() > 0.9:
        # 	self.learning_rate = 0.0001
        # if dataset.get_validation_accuracy() > 0.8:
        # 	self.learning_rate = 0.015


        # if dataset.get_validation_accuracy() > 0.75:
        # 	self.learning_rate = 0.1
        # if dataset.get_validation_accuracy() > 0.6:
        # 	self.learning_rate = 0.3
        # if dataset.get_validation_accuracy() > 0.6:
        # 	self.learning_rate = 0.4
        count = 0
        for x, y in dataset.iterate_forever(batch_size=5):
            loss = self.get_loss(x,y)
            if (count % 100 == 0):
                if dataset.get_validation_accuracy() > 0.98:
                    break
            gradient_list = nn.gradients(self.get_loss(x, y), 
        		[self.weights, self.bias, self.weights2, self.bias2])
            sign = -1
            self.weights.update(gradient_list[0], self.learning_rate*sign)
            self.bias.update(gradient_list[1], self.learning_rate*sign)
            self.weights2.update(gradient_list[2], self.learning_rate*sign)
            self.bias2.update(gradient_list[3], self.learning_rate*sign)
            count += 1

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.learning_rate = 0.06 #0.5
        self.hidden_size = 300 #50

        self.weights = nn.Parameter(47, self.hidden_size)
        self.bias = nn.Parameter(1, self.hidden_size)

        self.weights2 = nn.Parameter(self.hidden_size,5)
        self.bias2 = nn.Parameter(1, 5)

        #self.weights3 = nn.Parameter(self.hidden_size,10)
        #self.bias3 = nn.Parameter(1,1)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        batch_size = len(xs)
        func1 = None
        func2 = None
        for i in range(len(xs)):
            if i == 0:
                #print("xs[0] shape", xs[0].data.shape)
                #print("yo", nn.Linear(xs[0], self.weights).data.shape)
                func1 = nn.AddBias(nn.Linear(xs[i], self.weights), self.bias)
            else:
                func2 = nn.AddBias(nn.Linear(xs[i], self.weights), self.bias)
                sum = nn.ReLU(nn.Add(func1, func2))
                func1 = sum
        lne = nn.Linear(func1, self.weights2)
        return nn.AddBias(lne, self.bias2)
        # data_x_weights = nn.Linear(xs, self.weights)
        # add_bias = nn.AddBias(data_x_weights, self.bias)
        # activated = nn.ReLU(add_bias)
        # #print(activated.data.shape)
        # second = nn.Linear(activated, self.weights2)
        # add_second_bias = nn.AddBias(second, self.bias2)
        # #print(add_second_bias.data.shape)
        # return add_second_bias

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        prediction = self.run(xs) # a batch size x 10 
        return nn.SoftmaxLoss(prediction, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        #   self.learning_rate = 0.4
        count = 0
        for x, y in dataset.iterate_forever(batch_size=5):
            loss = self.get_loss(x,y)
            if (count % 100 == 0):
                if dataset.get_validation_accuracy() > 0.81:
                    break
            gradient_list = nn.gradients(self.get_loss(x, y), 
                [self.weights, self.bias, self.weights2, self.bias2])
            sign = -1
            self.weights.update(gradient_list[0], self.learning_rate*sign)
            self.bias.update(gradient_list[1], self.learning_rate*sign)
            self.weights2.update(gradient_list[2], self.learning_rate*sign)
            self.bias2.update(gradient_list[3], self.learning_rate*sign)
            count += 1
