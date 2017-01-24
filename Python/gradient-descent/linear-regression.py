#!/usr/bin/env python
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#               File Owner - Kartheek Palepu            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Loss function = half-MSE # # #
# # parameters
# y = original values of target variable
# yhat = predicted values of target variable
def loss(y, yhat):
	se = round(sum([(yhat[i] - y[i])**2 for i in range(len(y))]), 6)
	mse = se/(2.0 * len(y))
	return mse

# # # derivative function of theta0 # # #
# # parameters
# y = original values of target variable
# yhat = predicted values of target variable
def derivative1(y, yhat):
	d = sum([yhat[i] - y[i] for i in range(len(y))])
	d = d / float(len(y))
	return(d)

# # # derivative function of theta1 # # #
# # parameters
# y = original values of target variable
# yhat = predicted values of target variable
# x = list of independent data values in the data
def derivative2(y, yhat, x):
	d = sum([((yhat[i] - y[i]) * x[i]) for i in range(len(y))])
	d = d / float(len(y))
	return(d)

# # # A function that calculates the derivative change with learning rate # # #
# # parameters
# x = list of lists of independent feature values in the data
# y = original values of target variable
# yhat = predicted values of target variable
# params = parameters that needs optimization
# learning_rate = learning_rate to proceed with the gradient
def derivatives(x, y, yhat, params, learning_rate, complex = False):
	d1 = derivative1(y, yhat)
	d2 = derivative2(y, yhat, x)
	if complex:
		d1 = float(d1)/max(d1, d2)
		d2 = float(d2)/max(d1, d2)
	params['theta0'] = round(params['theta0'] - learning_rate * d1, 4)
	params['theta1'] = round(params['theta1'] - learning_rate * d2, 4)
	return(params)


# # # A function that calculates y value given x and params # # #
# # parameters
# x = list of lists of independent feature values in the data
# params = parameters that needs optimization
def yEQ(x, params):
	return [(params['theta0'] + (i * params['theta1'])) for i in x]

from gradient import iterLearningRate, vanilla_gradient_descent

# # # formulate original 'y' i.e. y = 3 + 2*x (i.e. theta0 + theta1 * x)
x = range(1, 10)
y = yEQ(x, params = {'theta0' : 3, 'theta1' : 2})

print vanilla_gradient_descent(x, y, params = {'theta0' : 0, 'theta1' : 0}, derivatives = derivatives, force = True, learning_rate = 0.1, yEQ = yEQ, loss = loss, adaptive = True)
print iterLearningRate(x, y, params = {'theta0' : 0, 'theta1' : 0}, derivatives = derivatives, yEQ = yEQ, force = True, loss = loss, adaptive = True, plot_flag = False)
