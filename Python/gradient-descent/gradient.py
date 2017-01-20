#!/usr/bin/env python
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#               File Owner - Kartheek Palepu            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # function to calculate vanilla gradient descent # # #
# # parameters
# x = input features (usually a list of lists)
# y = original target variable values
# params = parameters that needs to be optimized
# yEQ = A function that calculates the Yhat value with parameters x and params
# derivatives = A function that calculates the derivatives change for params
# loss = A function that calculates the loss value between original and predicted
# iters = number of iterations to execute
# learning_rate = speed of learning (high value converges faster but error will be more whereas low value converges slower with low error rate)
# adaptive = adaptive learning_rate configuration (the learning_rate adjusts based on error)
# threshold = threshold to conclude convergence
# change_threshold = if change of values are less than 10^-3 then declare convergence
# print_every = if print_every = 100, then print every 100th iteration
# plot_flag = to plot the error rate, gradient, fit equation
def vanilla_gradient_descent(x, y, params, yEQ, derivatives, loss, iters = 100000, learning_rate = 0.01, adaptive = False, threshold = 0.0001, change_threshold = 0.001, print_every = 100, plot_flag = True):
	# # start iterations
	for eachIter in range(iters):
		yhat = yEQ(x, params)
		if loss(y, yhat) <= threshold:
			print "=============================================================================="
			print "=============================================================================="
			print ">>>>> Converged at iteration # : " + str(eachIter) + " <<<<<"
			print "> Optimal Parameters :: " + " & ".join([' = '.join([str(j) for j in i]) for i in zip(params.keys(), params.values())])
			print "> Y values = " + ", ".join([str(k) for k in y])
			print "> Yhat values = " + ", ".join([str(k) for k in yhat])
			print "> Loss = " + str(loss(y, yhat))
			break;
		if eachIter % print_every == 0:
			print "> At Iteration " + str(eachIter) + " :: " + " & ".join([' = '.join([str(j) for j in i]) for i in zip(params.keys(), params.values())]) + " with Loss = " + str(loss(y, yhat))
		params = derivatives(x, y, yhat, params, learning_rate)
