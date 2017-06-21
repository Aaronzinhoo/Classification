%function
function [test_labels] = KNN(train,test,labels,k)
% create distance matrix of the test/train data
[row,col] = size(test);
[r,c] = size(train);
D = zeros(row,r);
% calc dist matrix
for m=1:row
    for n=1:r
        D(m,n) = norm(test(m,:)-train(n,:)); 
    end
end
[B,In] = sort(D,2);
%reduce the size of the matrix to get elem to k
Index = In(:,k);
test_labels = zeros(row,3);
for i=1:row
    % labels(index(i,:),:) grabs the rows of labels at row Index(i,k)
    % we take the sum of all of these rows
    % then find which element of the row is the max
    % set the index of the max elem to class
    %set the test point label to the correct class 
    [trash, class] = max(sum(labels(Index(i,:),:),1)); 
    test_labels(i,class) = 1;
end
end