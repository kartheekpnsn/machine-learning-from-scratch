
% Logistic regression using stochastic gradient descent
% train = training data
% target = target dependent variable (only numeric !!) (i.e. 0, 1)
% learningRate = Used to limit the amount each coefficient is corrected each time it is updated.
% (same learningRate concept as in Neural nets)
% epochs = The number of times to run through the training data while updating the coefficients.
% (same epochs concept as in Neural nets)
% threshold = at what error rate we need to break the loop (lower the overfitted !)

function [ coef ] = logisticRegSD_fn (train, target, coefs, epochs,learningRate)
threshold = 0.01;
    if(isempty(coefs))
        coef = zeros(1,size(train,2)+ 1);
    else
        coef = coefs;
    end
%  for each epoch
for eachepoch = 1:1:epochs
    sumError = 0;
    for eachrow = 1:1:size(train,1)
%   calculate the probability for each row 
        yhat = predictLR_fn( train(eachrow,:),coef);
%   calculate the error
        error = target(eachrow) - yhat;
%   mean squared error
        sumError = sumError + (error^2);
%   improve the coefficient beta0 by using learningRate, error and the predicted probability
    coef(1) = coef(1) + learningRate * error * yhat * (1 - yhat);
        for i = 1:1:size(train,2)
%       improve the other coefficients (beta1, beta2, ..) by using learningRate, error, the predicted probability and the original row values
            coef(i + 1) = coef(i + 1) + learningRate * error * yhat * (1 - yhat) * train(eachrow, i);
        end
    end
%   display at each epoch !
    fprintf('\n the current epoch is %d and error %d',eachepoch,sumError);
%   if threshold meets break the loop;
    if(sumError < threshold)
        break
    end
end
end

%%%% LOAD THE DATA
data=load('iris.csv');

totalIndex = 1:length(data);

%%%% To get 5 random rows from data
n=length(totalIndex);
nr=5;
[ ~,idx]=sort(rand(n,1));
out=totalIndex(idx(1:nr));

%%%% Keep the 5 selected rows as test and others as train
testIndex = out;
trainIndex = setdiff(totalIndex, testIndex);

train = data(trainIndex, :);
trainTarget = train(:, 5);
train = train(:, 1:4);
test = data(testIndex, :);
testTarget = test(:, 5);
test = test(:, 1:4);

%%%% convert trainTarget into binary
for i=1:1:length(trainTarget)
    if trainTarget(i) != 1
        trainTarget(i) = 0
    end
end

%%%% convert testTarget into binary
for i=1:1:length(testTarget)
    if testTarget(i) != 1
        testTarget(i) = 0
    end
end

%%%% run Logistic regression
coefs = logisticRegSD_fn(train, trainTarget, coefs = [0,0,0,0,0], epochs = 5000, learningRate = 0.3);

%%%% get probability scores for the 5 test rows
probs = []
for i=1:1:length(testTarget)
    probs = [probs, predictLR_fn(test(i, :), coefs)];
end

%%%% make the probability scores to binary
for i=1:1:length(probs)
    if probs(i) >= 0.5
        probs(i) = 1
    else
        probs(i) = 0
    end
end

%%%% Calculate TP (True Positives), TN (True Negatives), FP (False Positives), FN (False Negatives) and then Accuracy
tp = 0
tn = 0
fp = 0
fn = 0

for i=1:1:length(probs)
    if probs(i) == 0 & testTarget(i) == 0
        tn = tn + 1;
    elseif probs(i) == 1 & testTarget(i) == 1
        tp = tp + 1;
    elseif probs(i) == 1 & testTarget(i) == 0
        fp = fp + 1;
    else
        fn = fn + 1;
    end
end

accuracy = ((tp + tn)/length(probs))*100
fprintf("\nThe Accuracy is: %d Percentage", accuracy);
