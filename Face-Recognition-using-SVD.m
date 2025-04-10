clear all, close all, clc

load ../codefile/allFaces.mat
%% 

allPersons = zeros(n*6,m*6);
count = 1;
for i=1:6
    for j=1:6

        allPersons(1+(i-1)*n:i*n,1+(j-1)*m:j*m) ...
            = reshape(faces(:,1+sum(nfaces(1:count-1))),n,m);
        count = count + 1;
    end
end

figure(1), axes('position',[0  0  1  1]), axis off
imagesc(allPersons), colormap gray
%% 

for person = 1:length(nfaces)
    subset = faces(:,1+sum(nfaces(1:person-1)):sum(nfaces(1:person)));
    allFaces = zeros(n*8,m*8);
    
    count = 1;
    for i=1:8
        for j=1:8
            if(count<=nfaces(person)) 
                allFaces(1+(i-1)*n:i*n,1+(j-1)*m:j*m) ...
                    = reshape(subset(:,count),n,m);
                count = count + 1;
            end
        end
    end
    
    imagesc(allFaces), colormap gray    
end
%% 

% We use the first 36 people for training data
trainingFaces = faces(:,1:sum(nfaces(1:36)));
avgFace = mean(trainingFaces,2);  % size n*m by 1;
%% 



% Compute eigenfaces on mean-subtracted training data
X = trainingFaces-avgFace*ones(1,size(trainingFaces,2));
[U,S,V] = svd(X,'econ');
%% 


figure, axes('Position',[0 0 1 1]), axis off
imagesc(reshape(avgFace,n,m)),
colormap gray;

EigenFaces = zeros(n*8,m*8);
count = 1;
for i=1:8
    for j=1:8
        EigenFaces(1+(i-1)*n:i*n,1+(j-1)*m:j*m) ...
            = reshape(U(:,count),n,m);
        count = count + 1;
    end
end
figure(1), axes('Position',[0 0 1 1]), axis off
imagesc(EigenFaces),
colormap;
%% 


imagesc(reshape(avgFace,n,m)) % Plot avg face
imagesc(reshape(U(:,18),n,m))  % Plot first eigenface

%% 


testFace = faces(:,2+sum(nfaces(1:36))); % First face of person 37
subplot(2,4,1)
imagesc(reshape(testFace,n,m)) 
colormap gray
count = 1;
testFaceMS = testFace - avgFace;
for r=[4 10 20 100 200 800 1600 ]
    count = count+1;
    subplot(2,4,count)
    reconFace = avgFace + (U(:,1:r)*(U(:,1:r)'*testFaceMS));
    % reconFace =  (U(:,1:r)*(U(:,1:r)'*testFaceMS));
    imagesc(reshape(reconFace,n,m))
    title(['r',num2str(r,'%d')]);
end

%% Project person 2 and 7 onto PC5 and PC6

P1num = 2;  % person number 2
P2num = 7;  % person number 7
P1 = faces(:,1+sum(nfaces(1:P1num-1)):sum(nfaces(1:P1num)));
P2 = faces(:,1+sum(nfaces(1:P2num-1)):sum(nfaces(1:P2num)));
P1 = P1 - avgFace*ones(1,size(P1,2));
P2 = P2 - avgFace*ones(1,size(P2,2));

figure 
subplot(1,2,1), imagesc(reshape(P1(:,1),n,m)); colormap gray, axis off
title('person2')
subplot(1,2,2), imagesc(reshape(P2(:,1),n,m)); colormap gray, axis off
title('person7')
% project onto PCA modes 5 and 6
PCAmodes = [5 6];
PCACoordsP1 = U(:,PCAmodes)'*P1;
PCACoordsP2 = U(:,PCAmodes)'*P2;

figure
plot(PCACoordsP1(1,:),PCACoordsP1(2,:),'kd','MarkerFaceColor','k')
title('person2')
axis([-4000 4000 -4000 4000]), hold on, grid on
plot(PCACoordsP2(1,:),PCACoordsP2(2,:),'r^','MarkerFaceColor','r')
title('person2 (black) , person7 (red)')
set(gca,'XTick',[0],'YTick',[0]);
%% 

% Compute Recognition Accuracy
num_train = sum(nfaces(1:36)); % Training data
num_test = sum(nfaces(37:end)); % Testing data

correct_count = 0;
total_count = num_test;
for i = 37:length(nfaces)  % Test persons
    for j = 1:nfaces(i)    % Each face of test person
        testFace = faces(:,sum(nfaces(1:i-1)) + j);
        testFaceMS = testFace - avgFace;
        projectedTest = U(:,1:100)' * testFaceMS; % Use top 100 eigenfaces

        minDist = inf;
        predicted_label = -1;
        for k = 1:36 % Compare with training persons
            refFaces = faces(:,sum(nfaces(1:k-1)) + (1:nfaces(k)));
            refProj = U(:,1:100)' * (refFaces - avgFace * ones(1, nfaces(k)));

            % Nearest neighbor classification (Euclidean distance)
            dists = sum((refProj - projectedTest).^2, 1);
            [minD, minIdx] = min(dists);
            
            if minD < minDist
                minDist = minD;
                predicted_label = k;
            end
        end
        
        % Check if correctly classified
        if predicted_label == i
            correct_count = correct_count + 1;
        end
    end
end


% Compute Reconstruction RMSE for different r values
r_values = [4, 10, 20, 100, 200, 800, 1600];
rmse_values = zeros(size(r_values));

for idx = 1:length(r_values)
    r = r_values(idx);
    reconFace = avgFace + (U(:,1:r) * (U(:,1:r)' * testFaceMS));
    rmse_values(idx) = sqrt(mean((testFace - reconFace).^2));
end

% Display Performance Table
fprintf('Performance Table:\n');
fprintf('r-value | RMSE\n');
fprintf('-----------------------------\n');
for i = 1:length(r_values)
    fprintf('%6d | %.4f\n', r_values(i) , rmse_values(i));
end

%% 

% Function to generate title with true and predicted labels
function titleStr = generateTitle(predLabel, trueLabel)
    titleStr = sprintf('Predicted: %s\nTrue: %s', predLabel, trueLabel);
end

% Select Person 2 and Person 7 for comparison
P1num = 2;  % Person 2
P2num = 7;  % Person 7
P1 = faces(:,1+sum(nfaces(1:P1num-1)):sum(nfaces(1:P1num)));
P2 = faces(:,1+sum(nfaces(1:P2num-1)):sum(nfaces(1:P2num)));

% Mean subtraction
P1 = P1 - avgFace * ones(1, size(P1, 2));
P2 = P2 - avgFace * ones(1, size(P2, 2));

% Project onto PCA modes 5 and 6
PCAmodes = [5 6];
PCACoordsP1 = U(:, PCAmodes)' * P1;
PCACoordsP2 = U(:, PCAmodes)' * P2;

% Reconstruct faces using only PC5 and PC6
reconP1 = U(:, PCAmodes) * PCACoordsP1 + avgFace;
reconP2 = U(:, PCAmodes) * PCACoordsP2 + avgFace;

% Generate Titles for Display
trueLabels = {'Person 2', 'Person 7'};
predictedLabels = {'Person 2 (SVD)', 'Person 7 (SVD)'};

titleP1 = generateTitle(predictedLabels{1}, trueLabels{1});
titleP2 = generateTitle(predictedLabels{2}, trueLabels{2});

% Visualization: Original vs. Reconstructed
figure;
subplot(2,2,1), imagesc(reshape(P1(:,1) + avgFace, n, m)), colormap gray, axis off;
title(titleP1, 'FontSize', 10);

subplot(2,2,2), imagesc(reshape(reconP1(:,1), n, m)), colormap gray, axis off;
title('Reconstructed by SVD', 'FontSize', 10);

subplot(2,2,3), imagesc(reshape(P2(:,1) + avgFace, n, m)), colormap gray, axis off;
title(titleP2, 'FontSize', 10);

subplot(2,2,4), imagesc(reshape(reconP2(:,1), n, m)), colormap gray, axis off;
title('Reconstructed by SVD', 'FontSize', 10);
