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

f_gradientfit = function(data, target, learning_rate = 0.5, n_estimators = 10, verbose = FALSE) {
	y_pred = rep(mean(data[, target]), nrow(data))
	y = data[, target]
	fit_list = list()
	library(rpart)
	for(i in 1:n_estimators) {
		gradient = (1 / (1 + exp(-y_pred))) - y
		m_data = cbind(subset(data, select = setdiff(colnames(data), target)), Y = gradient)
		sample_m_data = m_data[sample(rownames(m_data), round(nrow(m_data) * 0.7)), ]
		fit = rpart(Y ~ ., data = sample_m_data, control = rpart.control(minsplit = 2, maxdepth = 2))
		fit_list[[i]] = fit
		update = predict(fit, m_data)
		y_pred = y_pred - (learning_rate * update)
		progress = 100 * (i / n_estimators)
		if(progress %% 10 == 0) {
			print(paste('Progress at tree -', i, ':', progress, "%"))
			error = (1/length(y_pred)) * sum(((1 / (1 + exp(-y_pred))) - y)^2)
			print(paste0("Error at tree - ", i, ": ", error))
			if(verbose) {
				print(cbind(gradient = gradient, y_pred = y_pred))
			}
		}
	}
	return(fit_list)
}


f_predict = function(fit, newdata) {
	y_pred = rep(0, nrow(newdata))
	for(i in 1:n_estimators) {
		update = predict(fit[[i]], newdata)
		update = learning_rate * update
		y_pred = y_pred - upte
		print(rbind(update, y_pred))
	}
	y_pred = exp(y_pred)/sum(exp(y_pred))
	return(y_pred)
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # IMPLEMENTATION # # #
df = data.frame(A = runif(100, 10, 50), B = runif(100, 20, 100))
df$Y = c(0, 1, 1, 0)

fit = f_gradientfit(df, target = 'Y')

f_predict(fit, df)
