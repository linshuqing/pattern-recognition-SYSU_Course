% Write a MATLAB function that will take as inputs:
% (a) the mean vectors,and (b) a matrix X containing column vectors that stem
% from the above classes. It will give as output an N-dimensional vector whose
% ith component contains the class where the corresponding vector is assigned,
% according to the minimum Euclidean distance classifier.


l=2;
c=3;
N=2000;
m=[[0 1 2];
   [0 1 2]];
S=ones(2,2,3);
S(:,:,1)=eye(2,2);
S(:,:,2)=eye(2,2);
S(:,:,3)=eye(2,2);
% 对应类别1，2，3...
P=[0.3,0.5,0.2];
% 产生3类，二维，均值分别为0 0，1 1，2 2，方差相同的高斯分布
[X,y]=generate_gauss_classes(m,S,P,N);
z=euclidean_classifier(m,X);

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