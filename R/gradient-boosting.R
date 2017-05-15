# # # GRADIENT BOOSTING # # #
# # STEP: 1 (BUILDING)
	# # Build Decision Stump Classifier and Predict or Predict as mean value of train target value
	# # For each iteration
		# # Calculate gradient sigmoid(ypred) - yhat
		# # Cbind(X, gradient) - build decision stump regressor with maxdepth = 2 and minsplit = 2
		# # store the model in a list
		# # predict the gradient on X
		# # update the prediction using learning rate i.e. y_pred = y_pred - (learning_rate * predicted_gradient)
# # STEP: 2 (PREDICTION)
	# # For each tree
		# # predict the gradient
		# # update the gradient using learning rate i.e. gradient = gradient * learning_rate
		# # update the final prediction as:
			# # y_pred - gradient
	# # squash it using soft max

f_gradient_fit = function(data, target = 'Y', base_model = NULL, n_estimators = 100, learning_rate = 1, verbose = TRUE) {
	nclasses = length(unique(data[, target]))
	y = data[, target]
	model_list = list()
	library(rpart)
	for(eachClass in 1:nclasses) {
		print(paste("# # For Class:", eachClass - 1, "# #"))

		if(is.null(base_model)) {
			y_pred = rep(mean(y), nrow(data))
		} else {
			y_pred = predict(base_model, data, type = 'prob')[, eachClass]
		}

		fit_list = list()
		for(i in 1:n_estimators){
			gradient = -(y - y_pred)
			m_data = cbind(subset(data, select = setdiff(colnames(data), target)), Y = gradient)
			fit = rpart(Y ~ ., data = m_data, control = rpart.control(minsplit = 2, maxdepth = 2))
			fit_list[[i]] = fit
			update = predict(fit, m_data)
			y_pred = y_pred - (learning_rate * update)
			if(i %% 10 == 0) {
				progress = (i/n_estimators) * 100
				loss = (1/length(y_pred)) * sum((y - y_pred)**2)
				if(verbose) {
					print(paste("> Progress :", progress))
					print(paste("> Loss     :", loss))
				}
			}
		}
		model_list[[eachClass]] = fit_list
	}
	return(model_list)
}

f_gradient_predict = function(model_list, newdata, learning_rate = 1, validate = TRUE, target = 'Y', base_model = NULL) {
	y_pred = list()
	for(eachClass in 1:length(model_list)) {
		y_pred[[eachClass]] = rep(0, nrow(newdata))
		for(tree in model_list[[eachClass]]) {
			update = predict(tree, newdata = newdata)
			update = learning_rate * update
			y_pred[[eachClass]] = y_pred[[eachClass]] - update
		}
	}
	y_pred = do.call('cbind', y_pred)
	y_pred = data.frame(t(apply(y_pred, 1, function(x) 1 - (exp(x)/sum(exp(x))))))
	colnames(y_pred) = paste('class_', 1:length(model_list))
	if(validate) {
		source('https://raw.githubusercontent.com/kartheekpnsn/machine-learning-codes/master/R/functions.R')
		y_pred[, target] = newdata[, target]
		cutoff = getCutoff(probabilities = y_pred[, 2], original = newdata[, target], plotROC = FALSE, all = FALSE)
		print("# # Base model performance # #")
		print(performance(predicted = predict(base_model, newdata), original = newdata[, target]))
		print("# # Boosted model performance # #")
		print(performance(predicted = as.numeric(y_pred[, 2] >= cutoff), original = newdata[, target]))
		return(list(cutoff = cutoff, predicted = y_pred))
	} else {
		return(list(predicted = y_pred))
	}
}


# # # # # IMPLEMENTATION # # # # # 
	data = data.frame(A = sample(c('male', 'female', 'others'), 1000, replace = T), 
						B = sample(c('normal', 'lean', 'fit', 'fat'), 1000, replace = T))
	data$Y = 0
	data$Y[data$A == 'male' & data$B %in% c('fat', 'lean', 'normal')] = 1
	data$Y[data$A == 'female' & data$B %in% c('lean', 'fat')] = 1
	data$Y[data$A == 'others' & data$B == 'lean'] = 1

	test = data.frame(A = sample(c('male', 'female', 'others'), 100, replace = T), 
						B = sample(c('normal', 'lean', 'fit', 'fat'), 100, replace = T))
	test$Y = 0
	test$Y[test$A == 'male' & test$B %in% c('fat', 'lean', 'normal')] = 1
	test$Y[test$A == 'female' & test$B %in% c('lean', 'fat')] = 1
	test$Y[test$A == 'others' & test$B == 'lean'] = 1

	library(randomForest)
	base_model = randomForest(factor(Y) ~ ., data = data, ntree = 2, mtries = 1)
	table(predict(base_model, data), data$Y)

	model_list = f_gradient_fit(data = data, target = 'Y', base_model = base_model, n_estimators = 100)
	predicted = f_gradient_predict(model_list, newdata = test, target = 'Y', validate = TRUE, base_model = base_model)
