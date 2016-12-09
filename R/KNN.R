# http://www.learnbymarketing.com/tutorials/k-nearest-neighbors-in-r-example/

# # Function to calculate euclidean distance
euclideanDistance = function(a, b) {
	sqrt(sum((a - b)^2))
}

# # k-nearest-neighbors-in-r-function
# train = training set
# target = target variable (in case of iris it is Species)
# test = test data to get the prediction
# k = no. of neighbors to be evaluated
knn = function(train, target, test, k) {
	# n = number of rows in training
	n = nrow(train)
	if(n != length(target)) {
		stop("target should be of same length as train")
	}
	if (n <= k){ 
		stop("k can not be more than n-1")
	}
	# calcuate euclidean distance and store it in the eucDist vector (vector is kind off array)
	eucDist = vector()
	for(i in 1:n) {
		eucDist = c(eucDist, euclideanDistance(train[i, ], test))
	}
	# sort the distances in ascending order and get the first K distance indexes
	firstK = order(eucDist)[1:k]
	# using the first K distance indexes get the target values
	fTarget = target[firstK]
	# now sort the fTarget based on frequency and get the highest frequent target name
	fTarget = names(sort(table(fTarget), decreasing = TRUE))[1]
	return(fTarget)
}

# to avoid randomness
set.seed(294056)
# sample 5 rows from iris dataset for testing
index = as.numeric(sample(rownames(iris), 5))
# store training data with out the above 5 rows
training = iris[-index, ]
# remove target from training as we store it seperately
training$Species = NULL
# store target of training in target object
target = as.character(iris$Species[-index])
# store the sampled 5 rows in test set
test = iris[index, ]
# remove target from test also as we store it seperately
test$Species = NULL
# store target of test set in testTarget object
testTarget = as.character(iris$Species[index])

# predictedTarget stores all the predictions
predictedTarget = vector()
# iterate through test rows to get predictions for each row
for(i in 1:nrow(test)) {
	predictedTarget = c(predictedTarget, knn(train = training, target = target, test = test[i,], k = 1))
}
# see the confusion matrix
print(table(testTarget, predictedTarget))
