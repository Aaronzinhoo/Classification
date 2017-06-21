%%
%1 Generate the data 
mu1 = [1,2];
mu2 = [2 4];
sigma1 = [.1 .05;.05 .2];
sigma2 = [.2 -0.1;-0.1 .3];
r1 = mvnrnd(mu1,sigma1,50);
r2 = mvnrnd(mu2,sigma2,50);
%%
%2 Plot the data
figure
hold on
scatter(r1(:,1),r1(:,2),'+');
scatter(r2(:,1),r2(:,2),'^');
legend('N(\mu_1,\Sigma_1)', 'N(\mu_2,\Sigma_2)')
xlabel('Data x')
ylabel('Data y')
title('Plot of Data #2')
hold off
%%
% Compute least squares to find W
T = [ones(50,1),zeros(50,1);zeros(50,1), ones(50,1)];
X = horzcat(ones(100,1),[r1;r2]);
W = inv((X'*X))* X'*T;
X1 = X*W(:,1);
X2 = X*W(:,2);
figure
hold on
plot(X1, 'o')
plot(X2, '*')
plot(T(:,1),'^')
plot(T(:,2), '+')
title('T Compared to Predicted Values XW')
xlabel('Data Input')
ylabel('Approx. Classification')
legend('XW_1','XW_2','T_1','T_2')
hold off
w_o = W(1,2) - W(1,1);
x1 = -(w_o)/(W(2,2) - W(2,1));
x2 = -(w_o)/(W(3,2) - W(3,1));
%%
%4 Plot line between points
figure
hold on
plot([0,x1],[x2,0])
scatter(r1(:,1),r1(:,2),'+');
scatter(r2(:,1),r2(:,2),'^');
legend('N_1/N_2','N(\mu_1,\Sigma_1)', 'N(\mu_2,\Sigma_2)')
xlabel('Data x')
ylabel('Data y')
title('Data with Boundary')
hold off
%%
%5 Generate N=50 2D data 
mu1 = [2,2];
mu2 = [2 4];
mu3 = [3 3];
sigma1 = [.2 .05;.05 .3];
sigma2 = [.4 -0.1;-0.1 .3];
sigma3 = [.5 -0.3;-0.3 .4];
r1 = mvnrnd(mu1,sigma1,50);
r2 = mvnrnd(mu2,sigma2,50);
r3 = mvnrnd(mu3,sigma3,50); 
figure
hold on
scatter(r1(:,1),r1(:,2),'+');
scatter(r2(:,1),r2(:,2),'^');
scatter(r3(:,1),r3(:,2),'x');
title('Plot of Data for #5')
xlabel('Data x')
ylabel('Data y')
legend('N(\mu_1,\Sigma_1)', 'N(\mu_2,\Sigma_2)', 'N(\mu_3,\Sigma_3)')
hold off
%%
%6 Class boundaries/Coeff
T_Train = [ones(50,1),zeros(50,1),zeros(50,1);zeros(50,1), ones(50,1),zeros(50,1);zeros(50,1),zeros(50,1), ones(50,1)];

X_Train = horzcat(ones(150,1),[r1;r2;r3]);
W = inv((X_Train'*X_Train))* X_Train'*T_Train;
%%
x = 0:5; 
b1 =  (W(1,2)-W(1,1))/(W(3,1)-W(3,2));
m1 = (W(2,2)-W(2,1))/(W(3,1)-W(3,2));
b2 = -(W(1,3)-W(1,1))/(W(3,3)-W(3,1));
m2 = -(W(2,3)-W(2,1))/(W(3,3)-W(3,1));
b3 = -(W(1,3) - W(1,2))/(W(3,3) - W(3,2));
m3 = -(W(2,3) - W(2,2))/(W(3,3)-W(3,2));
figure
hold on
scatter(r1(:,1),r1(:,2),'+');
scatter(r2(:,1),r2(:,2),'^');
scatter(r3(:,1),r3(:,2),'x');
plot(x,m1*x+b1,'--')
plot(x,m2*x+b2, ':')
plot(x,m3*x + b3)
title('Plot of Data W/ Boundaries #5')
xlabel('Data x')
ylabel('Data y')
legend('N(\mu_1,\Sigma_1)', 'N(\mu_2,\Sigma_2)', 'N(\mu_3,\Sigma_3)','N_1/N_2', 'N_1/N_3', 'N_2/N_3')
hold off
    
%%
%7
T_Test = [ones(100,1),zeros(100,1),zeros(100,1);zeros(100,1), ones(100,1),zeros(100,1);zeros(100,1),zeros(100,1), ones(100,1)];
r1 = mvnrnd(mu1,sigma1,100);
r2 = mvnrnd(mu2,sigma2,100);
r3 = mvnrnd(mu3,sigma3,100);
X_Test = horzcat(ones(300,1),[r1;r2;r3]);
LS_time = cputime;
%W = inv((X_Test'*X_Test))* X_Test'*T_Test;
C = W'*(X_Test');
Time_LS = cputime - LS_time;
[M,I] = max(C);
Test(1:100) = 1;
Test(101:200) = 2;
Test(201:300) = 3;
Comp = I==Test;
R = sum(Comp,2);
c = 100* R/(3*100);

%The value for c is going to be less than 100 because of the
%fact that our data points over lap. So there is a chance for
%misclassification if a data point is far from the cluster of its group
%which is the case here. Also Because this is least squares which assume a
%unimodal gaussian distribution while the data is not necessarily unimodal
%this may also cause problems with classification.

%%
%8 Implement KNN
%Already done as a fucntion
class_err = [];
times = [];
TLabel(1:100) = 1;
TLabel(101:200) = 2;
TLabel(201:300) = 3;
%%
%9 test code for various K

for K=1:60
    d = cputime;
    label = KNN(X_Train,X_Test,T_Train,K);
    times(K) = cputime-d;
    [M,	I9] = max(label,[],2);
    Comp = I9'==TLabel;
    R9 = sum(Comp);
    class_err(K) = 100* R9/(3*100);
end
disp('done')
figure
plot(class_err)
xlabel('K Value')
ylabel('Error in %')
title('Error of KNN')
figure
plot(times)
xlabel('K Value')
ylabel('Time in cputime')
title('Time of KNN')

% In this case, it seems that LS method actually seems to work better than
% KNN by a very little margin, approx ~5% which to me is suprising, but
% this could be because the data overlaps so much, KNN gets a lot of
% misclassification due to there existing clusters from both groups on each
% in certain regions. In terms of speed both seem to perform significantly
% fast, but this is primarily due to the low number of data points. Had
% this number been larger, KNN would do significantly worst since it
% iterates through all the training and testing data. For values of K 
% between 1 and 60 we see a decrease in the accuracy of classification of
% data. We see at K=1 (which is geenrally over-fitting) we hit the highest
% accuracy which isnt necessarily a good thing since it implies overfitting
% and after that we see the accuracy diminish. This justifies not showing K
% for values greater than 60. The max is aorund 75% for this run. Meaning
% for this case LS is better since it faster and classifies around the same
% and generally a little better.

%%
%10 Repeat 5-9 for 4 sets of data
mu1 = [2,2];
mu2 = [3 3];
mu3 = [4 4];
sigma1 = [.1 0;0 .1];
sigma2 = [.2 -0.1;-0.1 .3];
sigma3 = [.4 -0.3;-0.3 .3];
r1 = mvnrnd(mu1,sigma1,50);
r2 = mvnrnd(mu2,sigma2,50);
r3 = mvnrnd(mu3,sigma3,50); 
%%
figure
hold on
scatter(r1(:,1),r1(:,2),'+');
scatter(r2(:,1),r2(:,2),'^');
scatter(r3(:,1),r3(:,2),'x');
title('Prob 10 Data')
legend('N(\mu_1,\Sigma_1)', 'N(\mu_2,\Sigma_2)', 'N(\mu_3,\Sigma_3)')
xlabel('x')
ylabel('y')
hold off
%%
%number 6 again
X_Data = horzcat(ones(150,1),[r1;r2;r3]);
W = inv((X_Data'*X_Data))*X_Data'*T_Train;
x = 0:6; 
b1 =  (W(1,2)-W(1,1))/(W(3,1)-W(3,2));
m1 = (W(2,2)-W(2,1))/(W(3,1)-W(3,2));
b2 = -(W(1,3)-W(1,1))/(W(3,3)-W(3,1));
m2 = -(W(2,3)-W(2,1))/(W(3,3)-W(3,1));
b3 = -(W(1,3) - W(1,2))/(W(3,3) - W(3,2));
m3 = -(W(2,3) - W(2,2))/(W(3,3)-W(3,2));
figure
hold on
scatter(r1(:,1),r1(:,2),'+');
scatter(r2(:,1),r2(:,2),'^');
scatter(r3(:,1),r3(:,2),'x');
plot(x,m1*x+b1,'--')
plot(x,m2*x+b2, ':')
plot(x,m3*x + b3)
xlabel('x')
ylabel('y')
title('Prob 10 Data w/ Boundaries')
legend('N(\mu_1,\Sigma_1)', 'N(\mu_2,\Sigma_2)', 'N(\mu_3,\Sigma_3)','N_1/N_2', 'N_1/N_3', 'N_2/N_3')
hold off

%%
% 7 again
T_Test = [ones(100,1),zeros(100,1),zeros(100,1);zeros(100,1), ones(100,1),zeros(100,1);zeros(100,1),zeros(100,1), ones(100,1)];
r1 = mvnrnd(mu1,sigma1,100);
r2 = mvnrnd(mu2,sigma2,100);
r3 = mvnrnd(mu3,sigma3,100);
X_Test = horzcat(ones(300,1),[r1;r2;r3]);
LS_time = cputime;
C = W'*(X_Test');
Time_LS = cputime - LS_time;
[M,I] = max(C);
Test(1:100) = 1;
Test(101:200) = 2;
Test(201:300) = 3;
Comp = I==Test;
%overall class classification error
R10 = sum(Comp,2);
c10 = 100* R10/(3*100)

%%
% per class accuracy
R1 = sum(Comp(1:100),2);
R2 = sum(Comp(101:200),2);
R3 = sum(Comp(201:300),2);
c1 = 100* R1/(100);   
c2 = 100* R2/(100);   
c3 = 100* R3/(100); 

%%
%8 Implement KNN
%Already done as a fucntion
class_err = [];
times = [];
TLabel(1:100) = 1;
TLabel(101:200) = 2;
TLabel(201:300) = 3;
%%
%9 test code for various K
class_err = [];
times = [];
c_1 = [];
c_2 = [];
c_3 = [];
Comp_arr = zeros(60,300);
for K=1:60
    d = cputime;
    label = KNN(X_Data,X_Test,T_Train,K);
    times(K) = cputime-d;
    [M,	I10] = max(label,[],2);
    Comp_arr(K,:) = I10'==TLabel;
    R9 = sum(Comp_arr(K,:));
    class_err(K) = 100* R9/(3*100);
    R_1 = sum(Comp_arr(K,1:100));
    R_2 = sum(Comp_arr(K,101:200));
    R_3 = sum(Comp_arr(K,201:300));
    c_1(K) = 100* R_1/(100);  
    c_2(K) = 100* R_2/(100);   
    c_3(K) = 100* R_3/(100);
end
disp('done')
figure
plot(class_err)
xlabel('K Value')
ylabel('Classification Error in %')
title('Overall Class Error of KNN Prob 10')

figure
hold on
%plot(Time_LS)
plot(times)
xlabel('K Value')
ylabel('Time in cputime')
title('Time of KNN Prob 10')
hold off

figure
hold on
plot(c_1)
plot(c_2)
plot(c_3)
xlabel('K Value')
ylabel('Individual Class Acc (%)')
title('Plot of Individual Class Acc')
legend('Class 1','Class 2','Class 3')
hold off
%%
% 10 The classes are much more linearly seperable here than before actually they are l
% linearly sperable in this case. The class by class accuracies for LS in this case
% are poor only for the second class. As for the KNN trials though the class accuracies
% are almost 100% for each class with K from ~1-10, which means we are not 
% overfitting here and are acccuarately classifying the data. The decision boundaries
% for the LS classification are actually split across the middle data cluster 
% which is why the class is being poorly classified. The intersection splits 
% the middle dataset across either the first or second class and leaves only 
% a little triangle shaped area for the middle set classificaion 
% Here KNN is actually better than LS. The data is a lot better classified
% and the time isnt that big of a factor for such a difference in accuracy 
% The reason for this is because the data is seperable, most of the points 
% neighbors will actually be from the same group. LS assumes a unimodal
% gaussian distrubution that definitely isn the case here and it performs
% poorly compared to KNN since it misses the middle data set. This provides
% an other reason. The reason for any misclassification in KNN occurs when
% we have an outlier point near an other cluster which does occur. Overall
% KNN is better for linearly seperable data with multiple data points since
% it classifies based on the neighbors. Since the data is seperable we
% suspect for points of the same group to be mre likely closer to each
% other. Error for LS 66.67
 
    
    
    