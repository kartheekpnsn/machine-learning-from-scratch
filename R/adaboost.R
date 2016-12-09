# # # Clear Memory and run garbage collector # # #
  
	rm(list = ls())
	gc()


# # # Necessary Functions # # #

	# # To build the model # #
	callMyModel = function(model_name, formula, data, weights, ...)
	{ 
		return(model_name(formula, data, weights = weights,...))
	}
	
	# # Adaboost algorithm # #
	adaboost = function(model, dataFile, outcomeColName = "", outcomeColNum = 0, rounds = 10, ...)
	{
		# # Required Packages
		library(caret)
	
		if(!is.data.frame(dataFile))
		{
			dataFile = data.frame(dataFile)
		}
	
		if(outcomeColName == "")
		{
			if(outcomeColNum == 0)
			{
				stop("Please provide either outcome column name or column number")
			} else {
				colnames(dataFile)[outcomeColNum] = "Y"
			}
		} else {
			colnames(dataFile)[which(colnames(dataFile) == outcomeColName)] = "Y"
		}
		
		dataFile$Y = as.factor(as.character(dataFile$Y))
	
		if(length(levels(dataFile$Y)) == 2)
		{
			levels(dataFile$Y) = c(-1, 1)
		}
	
		# # Initialize weights
		n = nrow(dataFile)
		w = rep(1/n, n)
	
		# # Initialize alpha vector and modelList
		falpha = vector()
		modelList = list()
		for(i in 1:rounds)
		{
			# # Build model on training
			# ~ fit = callMyModel(model, Y ~ ., data = dataFile, weights = w)
			fit = rpart(Y ~ ., data = dataFile, weights = w)

			# # Predict on testing
			yPred = predict(fit, dataFile, type = "class")
	
			# # Calculate error in the model
			misclass = abs(as.numeric(as.character(dataFile$Y)) - as.numeric(as.character(yPred)))/2
			error = t(w) %*% misclass
	
			if(error > 0.5)
				error = 1 - error
			
			# # if vote == Inf i.e. error = 0. That means we have used a strong model for classification
			if(sum(error) == 0)
			{
				print(paste("Stopped in round - ", i, " - as the base model is having error of 0", sep = ""))
				print(paste("EXITING AT ROUND - ", i, sep = ""))
				break;
			}
			
			# # Tune the final vote weight based on error
			alpha = log((1 - error) / error)
	
			# # Recalculate weights based on error
			w = w * exp(alpha * misclass)
			w  = w / sum(w)
	
			# # Alpha will be the final weight during the final voting
			falpha[i] = alpha
	
			# # Save the model in the below list
			modelList[[i]] = fit
		}
		res = list(alpha = falpha, modelList = modelList)
		class(res) <- 'adaboost'
		return(res)
	}

# # # Set the path # # #

	setwd("YOUR - PATH - FOR - DATASET")


# # # Load required libraries # # #

	library(rpart)
	library(caret)

# # # Data Preprocessing # # #

	# # Read the data
	dataFile = read.csv("YOUR-DATASET.csv")

	# # Make the dependent variable factor
	dataFile$Y = as.factor(dataFile$Y)

	trainIndex = createDataPartition(dataFile[, "Y"], p = .60, list = FALSE, times = 1) # for balanced sampling
	trainSet = dataFile[trainIndex, ]
	testSet = dataFile[-trainIndex, ]


# # # RUN ADABOOST ALGORITHM # # #

	rounds = 10

	# # Run adaboost for 10 rounds
	fit = adaboost(rpart, trainSet, "Y", rounds)

	# # get predicted value for each of 10 learners
	fyhat = list()
	
	for(i in 1:rounds)
	{
		fyhat[[i]] = as.numeric(as.character(predict(fit$modelList[[i]], testSet, type = "class")))
	}

	# # Calculate the overall prediction using alpha
	fyhat = matrix(unlist(fyhat), ncol = nrow(testSet), byrow=T)
	sum = sign(t(fit$alpha) %*% fyhat)

	# # Confusion matrix
	table(sum, testSet$Y)


# # # Compare with a weak learner # # #

	weakFit = rpart(Y~., data = trainSet)
	yhat = predict(weakFit, testSet, type = "class")
	# # Confusion matrix of weak learner
	table(yhat, testSet$Y)
