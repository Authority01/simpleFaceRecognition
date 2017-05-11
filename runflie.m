clear ; close all; clc
%load('ORL_32x32.mat');
load('ORL_64x64/StTrainFile10.txt');
load('ORL_64x64/StTestFile10.txt');
%[m, n] = size(fea); % m is the number of data set , n is the number of features
X_train_raw = StTrainFile10(:, 1:end - 1);
y_train = StTrainFile10(:, end);
X_test_raw = StTestFile10(:, 1:end - 1);
y_test = StTestFile10(:, end);
n = size(X_train_raw, 2);
% randomly split the data set into trainning set and test set
% sel = randperm(m); 
% X_train_raw = fea(sel(1: round(0.7 * m)), :);
% y_train = gnd(sel(1: round(0.7 * m)));
% X_test_raw = fea(sel(round(0.7 * m) + 1 : end), :);
% y_test = gnd(sel(round(0.7 * m) + 1 : end));

% show some face pictures
% figure;
% for i = 1:9
%     subplot(3,3,i);
%     displayFace(X_train_raw(i, :),sqrt(n));
% end

% PCA dimension reduce, in order to speed up
[X_train, mu, sigma] = featureNormalize(X_train_raw);
[X_train, U] = pca(X_train); 
X_test = (X_test_raw - mu) ./ sigma;
X_test = X_test * U';

% neural network parameters
input_layer_size  = size(U,1);
num_labels = y_train(end);
hidden_layer_size = 50;
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
    num_labels, X_train, y_train, lambda);

% costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% prediction and accuracy
pred = predict(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);
pred = predict(Theta1, Theta2, X_test);
fprintf('\nprediction  test_y');
[pred, y_test] % show prediction and truth
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

%Visualize the prediction
for i = 1:length(pred)
    index = find(y_train == pred(i));
    figure;
    subplot(2, length(index), ceil(length(index)/2));
    displayFace(X_test_raw(i, :),sqrt(n))
    title('Who is he/she?')
    for j = 1:length(index)
        subplot(2, length(index), length(index) + j);
        displayFace(X_train_raw(index(j), :),sqrt(n));
        if j == ceil(length(index)/2)
            title('May be he/she? Right?')
        end
    end
    fprintf('Program paused. Press any key to continue! ctrl + C to quit.\n');
    pause;
end

