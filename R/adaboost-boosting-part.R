# # To start with
# I assumed I have built 3 weak classifiers and got the output as below.
# Note: I haven't done the classifier-building step but assumed some values as outputs as shown below

# original values
o = sample(c(1, -1), 100, replace = T)

# classifier - 1 (very bad classifier) (just gets 2 1's correct)
p1 = -sign(o)
p1[p1==-1][1:2] = 1

# classifier - 2 (very bad classifier) (just gets 2 -1's correct)
p2 = -sign(o)
p2[p2==1][1:2] = -1

# classifier - 3 (very bad classifier) (just gets one -1 and one 1 correct)
p3 = -sign(o)
p3[p3==1][3] = -1
p3[p3==-1][3] = 1


# function to calculate alpha value
# alpha = (1/2)*log((1-error)/error)
# where error = misclassification rate = (no.of.points misclassified)/(total no.of.points)
fn_alpha = function(o, p) {
	error = (length(which(o==-1 & p==1)) + length(which(o==1 & p==-1)))/length(p)
	return(0.5 * log((1-error)/error))
}

# calculate weights for each classifier
alpha = sapply(list(p1, p2, p3), fn_alpha, o = o)

# now build a linear combination of all 3 classifiers with their weights
preds = data.frame(p1, p2, p3)
# final_classifier = sign(sigma(alpha*preds))
fPreds = sign(rowSums(as.matrix(preds) %*% diag(alpha)))

# get accuracy
library(caret)
confusionMatrix(fPreds, o) # highest than all the 3 classifiers - 100% Accuracy
