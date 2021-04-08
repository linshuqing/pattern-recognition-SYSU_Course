% QUESTION
% a. Generate a data set X1 of N=1,000 two-dimensional vectors that stem
% from three equiprobable classes modeled by normal distributions with
% mean vectors m1=[1, 1] , m2=[12, 8] , m3=[16,1]T and covariance
% matrices S1=S2=S3=4I,where I is the 2 by 2 identity matrix.

% b. Apply the Bayesian,the Euclidean,and the Mahalanobis classifiers on X1.

% c. Compute the classification error for each classifier.

% 变量定义-----------------------------------------------------------------
l=2; %二维
c=3; %三个类别
N=10000; %1000条数据
% 方差矩阵
m=[[1 12 16];
   [1 8 1]];
% 协方差矩阵都是4I
S=ones(2,2,3);
S(:,:,1)=4*eye(2,2);
S(:,:,2)=4*eye(2,2);
S(:,:,3)=4*eye(2,2);
% 等概率分布
P=[1/3,1/3,1/3];

% 问题回答-----------------------------------------------------------------
% 问题a:产生要求的分布-----------------------------------------------------
% X:数据，y:真实类别
[X,y]=generate_gauss_classes(m,S,P,N);
% 问题b:应用三种分类器-----------------------------------------------------
% 贝叶斯分类器分类结果
z_bayes=bayes_classifier(m,S,P,X);
% 欧几里得距离分类结果
z_euclidean=euclidean_classifier(m,X);
% 马氏距离分类结果
z_mahalanobis=mahalanobis_classifier(m,S,X);
% 问题c:计算分类误差-------------------------------------------------------
[~,number_of_data]=size(y);
% 贝叶斯分类器误差
err_bayes=1-sum(z_bayes==y)/number_of_data;
fprintf('贝叶斯分类器误差：%6.4f\n',err_bayes);
% 欧几里得距离分类器误差
err_euclidean=1-sum(z_euclidean==y)/number_of_data;
fprintf('欧几里得距离分类器误差：%6.4f\n',err_euclidean);
% 马氏距离分类器误差
err_mahalanobis=1-sum(z_mahalanobis==y)/number_of_data;
fprintf('马氏距离分类器误差：%6.4f\n',err_mahalanobis);



% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
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
% 贝叶斯分类器
function z=bayes_classifier(m,S,P,X)
    [~,c]=size(m); % l=dimensionality, c=no. of classes
    [~,N]=size(X); % N=no. of vectors
    for i=1:N
        for j=1:c
            t(j)=P(j)*comp_gauss_dens_val(m(:,j),S(:,:,j),X(:,i));
        end
        % Determining the maximum quantity Pi*p(x|wi)
        [~,z(i)]=max(t); % 返回的是类别1，2，3...
    end
end

%  computes the value of the Gaussian distribution N(m, S)
function z=comp_gauss_dens_val(m,S,x)
    [l,~]=size(m);
    % l=dimensionality
    z=(1/((2*pi)^(l/2)*det(S)^0.5))...
    *exp(-0.5*(x-m)'*inv(S)*(x-m));
end

function z=euclidean_classifier(m,X)
    [~,c]=size(m); % l=dimensionality, c=no. of classes
    [~,N]=size(X); % N=no. of vectors
    for i=1:N
        for j=1:c
            t(j)=sqrt((X(:,i)-m(:,j))'*(X(:,i)-m(:,j))); % 计算欧几里得距离
        end
        % 返回欧几里得距离最小的类别
        [~,z(i)]=min(t); % 返回的是类别1，2，3...
    end
end

function z=mahalanobis_classifier(m,S,X)
    [~,c]=size(m); % l=dimensionality, c=no. of classes
    [~,N]=size(X); % N=no. of vectors
    for i=1:N
        for j=1:c
            t(j)=sqrt((X(:,i)-m(:,j))'*inv(S(:,:,j))*(X(:,i)-m(:,j))); % 计算马氏距离
        end
        % 返回马氏距离最小的类别
        [~,z(i)]=min(t); % 返回的是类别1，2，3...
    end
end
