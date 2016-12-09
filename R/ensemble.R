data = iris
target = ifelse(data$Species == "setosa", 1, 0)
data$Species = NULL
# So, we have 50 setosas and 100 others (i.e. 50 - 1's and 100 - 0's)
# model1 = very good model with almost all corrects (i.e. 50 points have probabilites 0.4 to 1 and 100 points have probabilities 0 to 0.4)
model1 = c(runif(50, 0.40001, 1), runif(100, 0, 0.4))
# model2 = good model with most corrects (i.e. 50 points have probabilites 0.3 to 1 and 100 points have probabilities 0 to 0.3)
model2 = c(runif(50, 0.30001, 1), runif(100, 0, 0.3))
# model3 = very bad model with almost no corrects (i.e. 50 points have probabilites 0.2 to 1 and 100 points have probabilities 0 to 0.2)
model3 = c(runif(50, 0.20001, 1), runif(100, 0, 0.2))
probabilities = data.frame(model1, model2, model3)

# # ENSEMBLE MODEL - 1
# MAXIMUM VOTING ALGORITHM
# calculate final probabilities
fProbabilities1 = rowSums(probabilities)/3
fProbabilities1 = ifelse(fProbabilities1 >= 0.5, 1, 0)
table(fProbabilities1, target)

# # # ENSEMBLE MODEL - 2
# # STACKING
# # row = each row of the testset
# # coefficients = coefficients obtained after building Logistic regression model
# predictLR = function(row, coefficients) {
# 	# formula: Yhat = beta0 + beta1 * X1 + beta2 * X2 + ..... betaN * XN
# 	beta0 = coefficients[1]
# 	betas = coefficients[2:length(coefficients)]
# 	yhat = beta0 + sum(row * betas)
# 	# return probability by using the below formula
# 	# yhat = e^(beta0 + beta1 * x1 + ...) / (1 + e^(beta0 + beta1 * x1 + ..))
# 	# the above formula can be simplified as
# 	# yhat = 1/(1.0 + e^(-(beta0 + beta1 * x1 + ...)))
# 	return(1/(1 + exp(-yhat)))
# }	

# # # Logistic regression using stochastic gradient descent
# # train = training data
# # target = target dependent variable (only numeric !!) (i.e. 0, 1)
# # learningRate = Used to limit the amount each coefficient is corrected each time it is updated.
# # (same learningRate concept as in Neural nets)
# # epochs = The number of times to run through the training data while updating the coefficients.
# # (same epochs concept as in Neural nets)
# # threshold = at what error rate we need to break the loop (lower the overfitted !)
# logisticRegSD = function(train, target, coefs = NULL, learningRate = 0.002, epochs = 200, threshold = 0.01) {
# 	if(is.null(coefs)) {
# 		coef = numeric(ncol(train) + 1)
# 	} else {
# 		coef = coefs
# 	}
# 	# run for each epoch
# 	for(eachEpoch in 1:epochs) {
# 		sumError = 0
# 		for(eachRow in 1:nrow(train)) {
# 			# calculate the probability for each row
# 			yhat = predictLR(train[eachRow, ], coef)
# 			# calculate the error
# 			error = target[eachRow] - yhat
# 			sumError = sumError + (error^2) # mean squared error
# 			# improve the coefficient beta0 by using learningRate, error and the predicted probability
# 			coef[1] = coef[1] + learningRate * error * yhat * (1 - yhat)
# 			for(i in 1:ncol(train)) {
# 				# improve the other coefficients (beta1, beta2, ..) by using learningRate, error, the predicted probability and the original row values
# 				coef[i + 1] = coef[i + 1] + learningRate * error * yhat * (1 - yhat) * train[eachRow, i]
# 			}
# 		}
# 		# print at each epoch !
# 		print(paste0("> > Epoch: ", eachEpoch, ", learningRate: ", learningRate, ", Error: ", sumError))
# 		# if threshold meets break the loop for godsake ! :-P
# 		if(sumError < threshold) { break; }
# 	}
# 	return(coef)
# }
# coefs = logisticRegSD(train = probabilities, target = target)

# fProbabilities2 = vector()
# for(i in 1:nrow(probabilities)){
# 	fProbabilities2 = c(fProbabilities2, predictLR(probabilities[i, ], coefs))
# }
# fProbabilities2 = ifelse(fProbabilities2 >= 0.5, 1, 0)
# table(fProbabilities2, target)
