clear; 
clc;
rng(0);

% a. Generate two data sets X1 and X1'of N = 200 two-dimensional vectors
% each. The first half of the vectors stem from the normal distribution with
% m1 = [�5, 0]T and S1 = I, while the second half of the vectors stem
% from the normal distribution with m1 = [5, 0]T and S1 = I,where I is the
% identity 2 by 2 matrix. Append each vector of both X1 and X1' by inserting
% an additional coordinate,which is set equal to 1.

% b. Apply the perceptron algorithm,the sum of error squares classifier,and the
% LMS algorithm on the previous data set, using various initial values for the
% parameter vector (where necessary).

% c. Measure the performance of each one of the above methods on both X1 and X1'�.

% d. Plot the data sets X1 and X1' as well as the line corresponding to the
% parameter vector w.

% SOLUTION


% 变量定义-----------------------------------------------------------------
l=2; %二维
c=2; %三个类别
N=200; %200条数据
% 方差矩阵
m=[[-5 5];
   [0 0]];
% 协方差矩阵都是4I
S=ones(2,2,2);
S(:,:,1)=eye(2,2);
S(:,:,2)=eye(2,2);
% 等概率分布
P=[0.5,0.5];

% a------------------------------------------------------------------------
% 产生分布
[X,y]=generate_gauss_classes(m,S,P,N);
% 获取X1、X2
X1=X(:,1:N/2,:); X2=X(:,N/2+1:N);
% 更改y，使其为+-1
y(y>1)=-1;
% 维度增加，且增加的维度的值都是1
new_row=ones(1,N);
X=[X;new_row];

% b------------------------------------------------------------------------
w_ini=[0.5 -1 1]';
% 感知器算法
w_perce=perce(X,y,w_ini);
% 最小化误差平方和分类器
w_SSEr=SSErr(X,y);
% LMS分类器
w_LMSalg=LMSalg(X,y,w_ini);

% c------------------------------------------------------------------------
result_perce=sign(w_perce'*X);
err_N1_perce=1-2*sum(result_perce(1:N/2)==y(1:N/2))/N; fprintf('感知器算法在I1类中分类误差：%6.4f\n',err_N1_perce);
err_N2_perce=1-2*sum(result_perce(N/2+1:N)==y(N/2+1:N))/N; fprintf('感知器算法在I2类中分类误差：%6.4f\n',err_N2_perce);

result_SSEr=sign(w_SSEr'*X);
err_N1_SSEr=1-2*sum(result_SSEr(1:N/2)==y(1:N/2))/N; fprintf('最小化误差平方和算法在I1类中分类误差：%6.4f\n',err_N1_SSEr);
err_N2_SSEr=1-2*sum(result_SSEr(N/2+1:N)==y(N/2+1:N))/N; fprintf('最小化误差平方和算法在I2类中分类误差：%6.4f\n',err_N2_SSEr);

result_LMSalg=sign(w_LMSalg'*X);
err_N1_LMSalg=1-2*sum(result_LMSalg(1:N/2)==y(1:N/2))/N; fprintf('LMS算法在I1类中分类误差：%6.4f\n',err_N1_LMSalg);
err_N2_LMSalg=1-2*sum(result_LMSalg(N/2+1:N)==y(N/2+1:N))/N; fprintf('LMS算法在I2类中分类误差：%6.4f\n',err_N2_LMSalg);
% 可以看出，在线性可分情况下，感知器算法和最小化误差平方和算法的错误概率都是0
% LMS算法则存在一定的错误分类（取决于初始值的选取）

% d------------------------------------------------------------------------
scatter(X(1,1:N/2),X(2,1:N/2),[],[1 0 0],"filled");
hold on;
scatter(X(1,N/2+1:N),X(2,N/2+1:N),[],[0 0 1],"filled");
hold on;
% w1*x+w2*y+w3=0;
% y=-w1/w2*x-w3/w2;
l_perce=refline(-w_perce(1)/(w_perce(2)+10^-6),-w_perce(3)/(w_perce(2)+10^-6));
l_perce.Color='k'; l_perce.DisplayName='感知器算法超平面'; l_perce.LineWidth=0.8;

l_SSEr=refline(-w_SSEr(1)/(w_SSEr(2)+10^-6),-w_SSEr(3)/(w_SSEr(2)+10^-6));
l_SSEr.Color='y'; l_SSEr.DisplayName='最小化误差平方和算法超平面'; l_SSEr.LineWidth=0.8;

l_LMSalg=refline(-w_LMSalg(1)/(w_LMSalg(2)+10^-6),-w_LMSalg(3)/(w_LMSalg(2)+10^-6));
l_LMSalg.Color='g'; l_LMSalg.DisplayName='LMS算法超平面'; l_LMSalg.LineWidth=0.8;

grid on;
axis([-10 10 -10 10]);
legend;

% 函数定义-----------------------------------------------------------------
% 子函数，产生高斯分布
function [X,y]=generate_gauss_classes(m,S,P,N)
    [~,c]=size(m);
    X=[];
    y=[];
    for j=1:c
        % Generating the [p(j)*N)] vectors from each distribution
        t=mvnrnd(m(:,j),S(:,:,j),fix(P(j)*N))';
        % The total number of points may be slightly less than N
        % due to the fix operator
        X=[X t];
        y=[y ones(1,fix(P(j)*N))*j];
    end
end

% 感知器算法
function w=perce(X,y,w_ini)
    [l,N]=size(X);
    max_iter=10000;
    % Maximum allowable number of iterations
    rho=0.05;
    % Learning rate
    w=w_ini;
    % Initialization of the parameter vector
    iter=0;
    % Iteration counter
    mis_clas=N;
    % Number of misclassified vectors
    while (mis_clas>0) && (iter<max_iter)
        iter=iter+1;
        mis_clas=0;
        gradi=zeros(l,1);% Computation of the "gradient"
        % term
        for i=1:N
            if((X(:,i)'*w)*y(i)<0)
                mis_clas=mis_clas+1;
                gradi=gradi+rho*(-y(i)*X(:,i));
            end
        end
    w=w-rho*gradi; % Updating the parameter vector
    end
end

% 最小化误差平方和分类器
function w=SSErr(X,y)
    w=inv(X*X')*(X*y');
end

% LMS分类器
function w=LMSalg(X,y,w_ini)
    [~,N]=size(X);
    rho=0.1;
    % Learning rate initialization
    w=w_ini;
    % Initialization of the parameter vector
        for i=1:N
        w=w+(rho/i)*(y(i)-X(:,i)'*w)*X(:,i);
        end
end
