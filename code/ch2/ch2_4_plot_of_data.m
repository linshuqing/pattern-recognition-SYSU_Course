% Write a MATLAB function that takes as inputs: (a) a matrix X and
% a vector y defined as in the previous function, (b) the mean vectors of c class
% distributions. It plots: (a) the data vectors of X using a different color for each
% class,(b) the mean vectors of the class distributions. It is assumed that the data
% live in the two-dimensional space.

% SOLUTION
% CAUTION: This function can handle up to
% six different classes
l=2;
c=3;
N=2000;
m=[[0 1 2];
   [0 1 2]];
S=ones(2,2,3);
S(:,:,1)=eye(2,2);
S(:,:,2)=eye(2,2);
S(:,:,3)=eye(2,2);
P=[0.3,0.5,0.2];
% 产生3类，二维，均值分别为0 0，1 1，2 2，方差相同的高斯分布
[X,y]=generate_gauss_classes(m,S,P,N);
% 画图
plot_data(X,y,m);

function plot_data(X,y,m)
    [~,N]=size(X); % N=no. of data vectors, l=dimensionality
    [l,c]=size(m); % c=no. of classes
    if(l ~=2 )
        fprintf('NO PLOT CAN BE GENERATED\n')
        return
    else
        pale=['r.'; 'g.'; 'b.'; 'y.'; 'm.'; 'c.'];
        figure(1)
        % Plot of the data vectors
        hold on
        for i=1:N
            plot(X(1,i),X(2,i),pale(y(i),:))
        end
        % Plot of the class means
        for j=1:c
            plot(m(1,j),m(2,j),'k+')
        end
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