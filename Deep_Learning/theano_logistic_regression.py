import numpy as np
import theano
import theano.tensor as T


N = 400 # Training sample size
features = 784 # Number of input variables
training_steps = 10000

# Generate dataset
observations = np.random.randn(N, features)
targets = np.random.randint(size=N, low=0, high=2)
dataset = (observations, targets)

# Declare Theano symbolic variables
x = T.dmatrix('x')
y = T.dvector('y')

# Create shared weight and bias vectors. Initialize with random values
w = theano.shared(np.random.randn(features), name='w')
b = theano.shared(0., name='b')

print('Initial model: ')
print(w.get_value())
print(b.get_value())

# Construct a theano expression graph
activation = -T.dot(x, w) - b     # activation = Linear combination + bias
p_1 = 1 / (1 + T.exp(activation)) # Probability that target = 1
prediction = p_1 > 0.5            # Prediction threshold

# Cross entropy loss function. Returns an array of cross entropy's
cross_entropy = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)

# Get the average of all the cross entropy's, add regularization
cost = cross_entropy.mean() + 0.01 * (w ** 2).sum()

# Compute the gradient of the cost (w/ regularization), w.r.t. weight vector w
# and bias term b
gradient_w, gradient_b = T.grad(cost, [w,b])

# Compile
train = theano.function(
    inputs=[x,y],
    outputs=[prediction, cross_entropy],
    updates=[(w, w - 0.1 * gradient_w), (b, b - 0.1 * gradient_b)]
)

predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, cross_entropy_error = train(dataset[0], dataset[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(dataset[1])
print("prediction on D:")
print(predict(dataset[0]))
print(dataset[1] == predict(dataset[0]))
