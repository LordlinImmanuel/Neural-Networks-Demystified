from numpy import random,dot,exp,array,mean

class NeuralNetwork():
    
    
    def __init__(self):
        #Setting the numpy.random.seed
        random.seed(1)
        
        #Initialising random hidden layer weights
        self.weightsAtHiddenLayer = 2*random.random((3,4))-1
        
        #Initialising random hidden layer weights
        self.weightsAtOutputLayer = 2*random.random((4,1))-1
        
        #Fixed learning rate of 0.1
        self.learningRate = 0.1
        
    def nonLinearFunction(self,x, derivative=False):
        #here, sigmoid derivative is used
        if(derivative==True):
            return x*(1-x)
        return 1/(1+exp(-x))
    
    def forwardPropagation(self, X):
        # hidden layer input = dot_product(X and weights of hidden layer)
        self.hiddenLayerInput = dot(X, self.weightsAtHiddenLayer)
        self.hiddenLayerActivation = self.nonLinearFunction(self.hiddenLayerInput,derivative=False)
        self.outputLayer = dot(self.hiddenLayerActivation, self.weightsAtOutputLayer)
    
    def backwardPropagation(self,X,y,iteration):
        
        #error calculation
        self.errorAtOutputLayer = y - self.outputLayer
        
        #to print out the error
        if (iteration% 10000) == 0:
            print "Error in prediction:" + str(mean(abs(self.errorAtOutputLayer)))
        
        # delta_output = slope of output layer * Error(which is the error at output layer)
        self.slopeOfOutputLayer = self.nonLinearFunction(self.outputLayer, derivative=True)
        
        # Delta at output layer
        self.deltaOutput = self.slopeOfOutputLayer * self.errorAtOutputLayer
        
        # delta_hidden = slope of hidden layer * error at hidden layer
        
        # slope at hidden layer = nonLinearFunctionDerivative(hidden layer activation)
        self.slopeOfHiddenLayer = self.nonLinearFunction(self.hiddenLayerActivation, derivative=True)
        
        # error at hidden layer = dot_product (delta_output and Transpose(weights at output layer))
        self.errorAtHiddenLayer = dot(self.deltaOutput, self.weightsAtOutputLayer.T)
        
        # Delta at hidden layer
        self.deltaHidden = self.slopeOfHiddenLayer * self.errorAtHiddenLayer
        
        #Weights Update(the key ingredient)
        
        # weights_at_output += dot_product(Transpose(hidden layer activation) and delta_output)*learning_rate
        self.weightsAtOutputLayer += dot(self.hiddenLayerActivation.T, self.deltaOutput) * self.learningRate
        
        # weights_at_hidden += dot_product(Transpose(X) and delta_hidden)*learning_rate
        self.weightsAtHiddenLayer += dot(X.T, self.deltaHidden) * self.learningRate
        
    def train(self,X,y,iterations):
        for iteration in xrange(iterations):
            self.forwardPropagation(X)
            self.backwardPropagation(X,y,iteration)
        
if __name__ == "main":
        
        #input matrices 'X'
        X = array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])
        
        #Expected output
        y = array([[0],
             [1],
             [1],
             [0]])
        
        #initialise the neural network class
        nn = NeuralNetwork()
        
        #initial weights before training
        print nn.weightsAtOutputLayer
        
        #training the neural network
        nn.train(X,y,10000)
        
        #final weights after training
        print nn.weightsAtOutputLayer
        
        #final prediction
        print nn.outputLayer
