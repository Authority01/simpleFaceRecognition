clear ; close all; clc
load('ORL_32x32.mat');
[m, n] = size(fea); % m is the number of data set , n is the number of features

% randomly split the data set into trainning set and test set
sel = randperm(m); 
train_X = fea(sel(1: round(0.7 * m)), :);
train_y = gnd(sel(1: round(0.7 * m)));
test_X = fea(sel(round(0.7 * m) + 1 : end), :);
test_y = gnd(sel(round(0.7 * m) + 1 : end));

% show some face pictures
figure;
for i = 1:9
    subplot(3,3,i);
    displayFace(train_X(i, :),sqrt(n));
end

% PCA dimension reduce, in order to speed up
[train_X, mu, sigma] = featureNormalize(train_X);
[train_X, U] = pca(train_X); 
test_X = (test_X - mu) ./ sigma;
test_X = test_X * U';
X_rec = train_X * U;

% neural network parameters
input_layer_size  = size(U,1);
hidden_layer_size = 50;
num_labels = gnd(end);
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% max iteration times
options = optimset('MaxIter', 50);

% regularation parameter
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, train_X, train_y, lambda);

% costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% prediction and accuracy
pred = predict(Theta1, Theta2, train_X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == train_y)) * 100);
pred = predict(Theta1, Theta2, test_X);
fprintf('\nprediction  test_y');
[pred, test_y]
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == test_y)) * 100);

