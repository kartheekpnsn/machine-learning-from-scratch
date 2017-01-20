#!/usr/bin/env python
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#               File Owner - Kartheek Palepu            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import math

# # # Loss function = half-MSE # # #
# # parameters
# y = original values of target variable
# yhat = predicted values of target variable
def loss(y, yhat):
	term1 = [y[i] * math.log(yhat[i]) for i in range(len(y))]
	term2 = [(1 - y[i]) * math.log((1 - yhat[i])) for i in range(len(y))]
	term3 = sum([term1[i] + term2[i] for i in range(len(term1))])
	term4 = -term3/float(len(y))
	return term4

# # # derivative function of intercept # # #
# # parameters
# y = original values of target variable
# yhat = predicted values of target variable
def derivative1(y, yhat):
	d = sum([yhat[i] - y[i] for i in range(len(y))])
	d = d / float(len(y))
	return(d)

# # # derivative function of slopes i.e. theta1, theta2, theta3.... thetaN # # #
# # parameters
# y = original values of target variable
# yhat = predicted values of target variable
# x_n = list of values of nth feature in the data
def derivativeN(y, yhat, x_n):
	d = sum([((yhat[i] - y[i]) * x_n[i]) for i in range(len(y))])
	d = d / float(len(y))
	return(d)


# # # A function that calculates the derivative change with learning rate # # #
# # parameters
# x = list of lists of independent feature values in the data
# y = original values of target variable
# yhat = predicted values of target variable
# params = parameters that needs optimization
# learning_rate = learning_rate to proceed with the gradient
def derivatives(x, y, yhat, params, learning_rate):
	params['intercept'] = round(params['intercept'] - learning_rate * derivative1(y, yhat), 6)
	params['slopes'] = [round(params['slopes'][i] - learning_rate * derivativeN(y, yhat, x[i]), 6) for i in range(len(x))]
	return(params)

# # # A function (sigmoid function) that calculates y value given x and params # # #
# # parameters
# x = list of lists of independent feature values in the data
# params = parameters that needs optimization
def yEQ(x, params):
	mx = [[j * params['slopes'][i] for j in x[i]] for i in range(len(x))]
	mxc = [params['intercept'] + sum(i) for i in zip(*mx)]
	yhat = [(1/ (1 + math.exp(-i))) for i in mxc]
	yhat = [0.9999999999999999 if i == 1.0 else i for i in yhat]
	return yhat


# # import gradient (vanilla_gradient_descent) algorithm
from gradient import vanilla_gradient_descent

# # initializations
# # generate x
x = []
x1 = range(1, 11)
x2 = range(1, 20, 2)
x3 = range(0, 19, 2)
x.append(x1)
x.append(x2)
x.append(x3)

# # formulate original 'y' i.e. y = 1/(1 + exp(-(1 + x1 + 2*x2 + x3)))
slopes = [1, -2, 1]
intercept = 1
y = [round(i) for i in yEQ(x, params = {'intercept' : intercept, 'slopes' : slopes})]

# # run vanilla_gradient_descent
vanilla_gradient_descent(x, y, learning_rate = 0.01, print_every = 1000, threshold = 0.001, iters = 10000000, params = {'slopes' : [0] * len(x), 'intercept' : 0}, derivatives = derivatives, yEQ = yEQ, loss = loss)
