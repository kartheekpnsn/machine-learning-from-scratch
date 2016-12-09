function [probs] = ensemble(probMatrix)
	probs = []
	for i=1:1:length(probMatrix)
		probs = [probs, sum(probMatrix(i, :))/length(probMatrix)]
	end
	for i=1:1:length(probs)
		if probs(i) >= 0.5
			probs(i) = 1
		else
			probs(i) = 0
		end
	end
end


original = [1, 0, 1];
predicted = [0.3, 0.6, 0.6; 0.2, 0.2, 0.7; 0.8, 0.1, 0.6];

probs = ensemble(predicted);

tp = 0;
tn = 0;
fp = 0;
fn = 0;

for i=1:1:length(probs)
	if probs(i) == 0 & original(i) == 0
		tn = tn + 1;
	elseif probs(i) == 1 & original(i) == 1
		tp = tp + 1;
	elseif probs(i) == 1 & original(i) == 0
		fp = fp + 1;
	else
		fn = fn + 1;
	end
end

accuracy = ((tp + tn)/length(probs))*100
fprintf("\nThe Accuracy is: %d Percentage", accuracy);
