# row = each row of the testset
# coefficients = coefficients obtained after building Logistic regression model
predictLR = function(row, coefficients) {
    # formula: Yhat = beta0 + beta1 * X1 + beta2 * X2 + ..... betaN * XN
    beta0 = coefficients[1]
    betas = coefficients[2:length(coefficients)]
    yhat = beta0 + sum(row * betas)
    # return probability by using the below formula
    # yhat = e^(beta0 + beta1 * x1 + ...) / (1 + e^(beta0 + beta1 * x1 + ..))
    # the above formula can be simplified as
    # yhat = 1/(1.0 + e^(-(beta0 + beta1 * x1 + ...)))
    return(1/(1 + exp(-yhat)))
}    

# # Logistic regression using stochastic gradient descent
# train = training data
# target = target dependent variable (only numeric !!) (i.e. 0, 1)
# learningRate = Used to limit the amount each coefficient is corrected each time it is updated.
# (same learningRate concept as in Neural nets)
# epochs = The number of times to run through the training data while updating the coefficients.
# (same epochs concept as in Neural nets)
# threshold = at what error rate we need to break the loop (lower the overfitted !)
logisticRegSD = function(train, target, coefs = NULL, learningRate = 0.002, epochs = 200, threshold = 0.01) {
    if(is.null(coefs)) {
        coef = numeric(ncol(train) + 1)
    } else {
        coef = coefs
    }
    # run for each epoch
    for(eachEpoch in 1:epochs) {
        sumError = 0
        for(eachRow in 1:nrow(train)) {
            # calculate the probability for each row
            yhat = predictLR(train[eachRow, ], coef)
            # calculate the error
            error = target[eachRow] - yhat
            sumError = sumError + (error^2) # mean squared error
            # improve the coefficient beta0 by using learningRate, error and the predicted probability
            coef[1] = coef[1] + learningRate * error * yhat * (1 - yhat)
            for(i in 1:ncol(train)) {
                # improve the other coefficients (beta1, beta2, ..) by using learningRate, error, the predicted probability and the original row values
                coef[i + 1] = coef[i + 1] + learningRate * error * yhat * (1 - yhat) * train[eachRow, i]
            }
        }
        # print at each epoch !
        print(paste0("> > Epoch: ", eachEpoch, ", learningRate: ", learningRate, ", Error: ", sumError))
        # if threshold meets break the loop for godsake ! :-P
        if(round(sumError, 2) <= threshold) { break; }
    }
    return(coef)
}

# min max normalization
# normalizes vector x into values between 0 and 1
minmaxNorm = function(x) {
    (x - min(x)) / (max(x) - min(x))
}

data = iris
# normalize if needed !!
# for(i in 1:(ncol(data)-1)) {
# 	data[, i]  = minmaxNorm(data[, i])
# }

data$Species = as.character(data$Species)
# make the species as binary (coz i know binary only in logistic regression :-P ).. if setosa then 1 else 0
data$Species = ifelse(data$Species == "setosa", 1, 0)

# to avoid randomness
set.seed(294056)
# sample 5 rows from iris dataset for testing
index = as.numeric(sample(rownames(data), 5))
# store training data with out the above 5 rows
train = data[-index, ]
# store target of training in trainTarget object
trainTarget = data$Species[-index]
# remove target from training as we have already stored it seperately
train$Species = NULL
# store the sampled 5 rows in test set
test = data[index, ]
# remove target from test also as we store it seperately
test$Species = NULL
# store target of test set in testTarget object
testTarget = data$Species[index]

# ~ ~ coefs = coef = c(5.6881452, 6.3293433, 4.9284260, -21.1255730, -0.3772764)
# default coefs will be NULL or all zeroes (to start by)
coefs = NULL
coefs = logisticRegSD(train, trainTarget, coefs = coefs, epochs = 5000, learningRate = 0.3)

# calculate probability for each row in test set
probs = vector()
for(i in 1:nrow(test)) {
	probs = c(probs, predictLR(test[i, ], coefs))
}
# confusion matrix - 100% ACCURACY {HOLAAAAAA !!}
print(table(probs >= 0.5, testTarget))
