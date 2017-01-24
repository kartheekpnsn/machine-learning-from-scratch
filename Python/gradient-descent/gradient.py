# # # function to calculate vanilla gradient descent # # #
# # parameters
# x = input features (usually a list of lists)
# y = original target variable values
# params = A dictionary of parameters that needs to be optimized with keys as names of the parameters and values as values of the parameters
# yEQ = A function that calculates the Yhat value with parameters x and params
# derivatives = A function that calculates the derivatives change for params
# loss = A function that calculates the loss value between original and predicted
# iters = number of iterations to execute
# learning_rate = speed of learning (high value converges faster but error will be more whereas low value converges slower with low error rate)
# adaptive = adaptive learning_rate configuration (the learning_rate adjusts based on error)
# threshold = threshold to conclude convergence
# change_threshold = if change of values are less than 10^-3 then declare convergence
# print_every = if print_every = 100, then print every 100th iteration
# force = flag to tell to force converge based on change in loss (if mean change in loss for past 100 iterations is less than 10^-4 then break)
# reg = flag to tell whether to add a regularization term or not
# scale = flag to tell whether to scale (normalize) the data or not
# plot_flag = to plot the error rate, gradient, fit equation
def vanilla_gradient_descent(x, y, params, yEQ, derivatives, loss, iters = 100000, learning_rate = 0.01, adaptive = False, threshold = 0.0001, change_threshold = 0.001, print_every = 100, force = True, reg = False, scale = False, plot_flag = True):
	iter_loss = []
	if scale:
		means = [sum(i)/float(len(i)) for i in x]
		sds = [sum([(j - means[i])**2 for j in x[i]])/(len(x[i])-1) for i in range(len(x))] # (1/(N-1)) * sigma((xi - mu)^2)
		x = [[(j - means[i])/float(sds[i]) for j in x[i]] for i in range(len(x))]

	# # start iterations
	break_flag = False
	for eachIter in range(iters):
		yhat = yEQ(x, params)
		iter_loss.append(loss(y, yhat))
		# adaptive learning rate if the present loss is increasing more than the previous loss then decrease the learning rate
		if adaptive and eachIter >= 2 and iter_loss[-1] > iter_loss[-2]:
			learning_rate = learning_rate/2
		if loss(y, yhat) <= threshold:
			print "=============================================================================="
			print ">>>>> Converged at iteration # : " + str(eachIter) + " with Learning rate = " + str(learning_rate) + " <<<<<"
			break_flag = True
			break;

		# check if loss is increasing for past 100 iterations
		if eachIter > 1000:
			temp_iter_loss = iter_loss[len(iter_loss):(len(iter_loss) - 100 - 1):-1]
			if all(temp_iter_loss[i] > temp_iter_loss[i+1] for i in xrange(len(temp_iter_loss)-1)):
				print "=============================================================================="
				print ">>>>> Cannot Converge with Learning rate = " + str(learning_rate) + " <<<<<"
				break_flag = True
				break;

		# check if the change in loss is less than 10^-4 if so break and declare convergence
		# or if the loss is not at all changing for past 100 iterations !
		if force:
			if len(iter_loss) >= 2000:
				temp_iter_loss = iter_loss[(len(iter_loss) - 1):(len(iter_loss) - 100):-1]
				if len(list(set(temp_iter_loss))) == 1:
					print "=============================================================================="
					print ">>>>> Force Converged at iteration # : " + str(eachIter) + " with Learning rate = " + str(learning_rate) + " <<<<<"
					break_flag = True
					break;
				# temp_iter_loss = [abs(temp_iter_loss[each] - temp_iter_loss[each + 1]) for each in range(len(temp_iter_loss) - 1)]
				# mean_loss = sum(temp_iter_loss)/float(len(temp_iter_loss))
				# if mean_loss < 0.0001:
				# 	print "=============================================================================="
				# 	print ">>>>> Force Converged at iteration # : " + str(eachIter) + " with Learning rate = " + str(learning_rate) + " <<<<<"
				# 	break_flag = True
				# 	break;
		
		if eachIter % print_every == 0:
			print "> At Iteration " + str(eachIter) + " :: " + " & ".join([' = '.join([str(j) for j in i]) for i in zip(params.keys(), params.values())]) + " with Loss = " + str(loss(y, yhat))
		params = derivatives(x, y, yhat, params, learning_rate)

	if break_flag:
		print "> Parameters :: " + " & ".join([' = '.join([str(j) for j in i]) for i in zip(params.keys(), params.values())])
		print "> Y values = " + ", ".join([str(k) for k in y])
		print "> Yhat values = " + ", ".join([str(k) for k in yhat])
		print "> Loss = " + str(loss(y, yhat))
	else:
		print "=============================================================================="
		print ">>>>> Ran Full Iterations for alpha = " + str(learning_rate) + " <<<<<"
		print "> End Parameters :: " + " & ".join([' = '.join([str(j) for j in i]) for i in zip(params.keys(), params.values())])
		print "> Y values = " + ", ".join([str(k) for k in y])
		print "> End Yhat values = " + ", ".join([str(k) for k in yhat])
		print "> End Loss = " + str(loss(y, yhat))

	if plot_flag:
		import matplotlib.pyplot as plt
		plt.plot(range(1, len(iter_loss) + 1), iter_loss, lw = 2)
		plt.axis([-len(iter_loss)/100, len(iter_loss) + 100, -max(iter_loss)/10, max(iter_loss) + 10])
		plt.title('For alpha = ' + str(learning_rate) + ' :: Loss at every iteration')
		plt.ylabel('Iterations')
		plt.xlabel('Loss')
		plt.show()
	return {'learning_rate' : learning_rate, 'params' : params, 'loss' : loss(y, yhat), 'iter_loss' : iter_loss}

# # # function to calculate vanilla gradient descent # # #
# # parameters
# x = input features (usually a list of lists)
# y = original target variable values
# params = A dictionary of parameters that needs to be optimized with keys as names of the parameters and values as values of the parameters
# yEQ = A function that calculates the Yhat value with parameters x and params
# derivatives = A function that calculates the derivatives change for params
# loss = A function that calculates the loss value between original and predicted
# iters = number of iterations to execute
# threshold = threshold to conclude convergence
# change_threshold = if change of values are less than 10^-3 then declare convergence
# print_every = if print_every = 100, then print every 100th iteration
# force = flag to tell to force converge based on change in loss (if mean change in loss for past 100 iterations is less than 10^-4 then break)
# reg = flag to tell whether to add a regularization term or not
# scale = flag to tell whether to scale (normalize) the data or not
# plot_flag = to plot the error rate, gradient, fit equation
def iterLearningRate(x, y, params, yEQ, derivatives, loss, iters = 100000, threshold = 0.0001, change_threshold = 0.001, print_every = 100, adaptive = True, force = True, reg = False, scale = False, plot_flag = True):
	alphas = [0.001, 0.01, 0.1, 1] # , 3, 5, 10]
	fParams = []
	for alpha in alphas:
		print ">>> For Learning Rate = " + str(alpha) + " <<<"
		temp = params.copy()
		try:
			vgd = vanilla_gradient_descent(x, y, temp, yEQ, derivatives, loss, learning_rate = alpha, iters = iters, threshold = threshold, adaptive = adaptive, change_threshold = change_threshold, print_every = print_every, force = force, reg = reg, scale = scale, plot_flag = plot_flag)
		except:
			print "# # # @ @ @ FAILED FOR ALPHA = " + str(alpha) + " @ @ @ # # #"
			vgd = {'learning_rate' : alpha, 'params' : params, 'loss' : float("inf")}
		print "=============================================================================="
		print "=============================================================================="
		fParams.append(vgd)
	loss = [i['loss'] for i in fParams]
	alphas = [i['learning_rate'] for i in fParams]
	fParams = [i['params'] for i in fParams]
	minLossId = [i for i in range(len(loss)) if loss[i] == min(loss)][0]
	return {'falpha' : alphas[minLossId], 'loss' : loss[minLossId], 'params' : fParams[minLossId]}
