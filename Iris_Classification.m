% Iris Classification using Perceptron and Bayesian Classifiers
% The task is to classify Iris Setosa and Iris Versicolor based on Petal Length and Petal Width
clc
clearvars


%% DATA PREPROCESSING

% Load the Iris dataset
load fisheriris;
% Select only Setosa and Versicolor classes
X = meas(1:100, 3:4); % Petal length and Petal width
Y = species(1:100);

% Encode labels: Setosa = 0, Versicolor = 1
Y = strcmp(Y, 'versicolor');

% Split the dataset into training and testing sets (70% training, 30% testing)
rng(1); % For reproducibility
cv = cvpartition(Y, 'HoldOut', 0.3);
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_test = X(test(cv), :);
Y_test = Y(test(cv));

%% PERCEPTRON IMPLEMENTATION

max_iter = 1000;  % Maximum number of iterations
learning_rate = 0.01;  % Learning rate
w = zeros(size(X_train, 2) + 1, 1); % Initialize weights

% Training the Perceptron
for iter = 1:max_iter
    for i = 1:length(Y_train)
        xi = [1; X_train(i, :)']; % Add bias term
        if (Y_train(i) == 1 && w' * xi <= 0) || (Y_train(i) == 0 && w' * xi > 0)
            w = w + learning_rate * (2*Y_train(i)-1) * xi;
        end
    end
end

% Evaluate Perceptron on the test set
Y_pred_perceptron = double([ones(size(X_test, 1), 1) X_test] * w >= 0);

%% BAYESIAN CLASSIFICATION

% Estimate the mean and covariance matrix for each class
mu_0 = mean(X_train(Y_train == 0, :)); % Mean of Setosa
mu_1 = mean(X_train(Y_train == 1, :)); % Mean of Versicolor
cov_0 = cov(X_train(Y_train == 0, :)); % Covariance of Setosa
cov_1 = cov(X_train(Y_train == 1, :)); % Covariance of Versicolor

% Calculate the prior probabilities
prior_0 = mean(Y_train == 0);
prior_1 = mean(Y_train == 1);

% Evaluate Bayesian Classifier on the test set
Y_pred_bayes = zeros(size(Y_test));
for i = 1:length(Y_test)
    p0 = mvnpdf(X_test(i, :), mu_0, cov_0) * prior_0;
    p1 = mvnpdf(X_test(i, :), mu_1, cov_1) * prior_1;
    Y_pred_bayes(i) = p1 > p0;
end

%% PERFORMANCE EVALUATION

% Perceptron Metrics
[accuracy_p, precision_p, recall_p, F1_p] = evaluate_metrics(Y_test, Y_pred_perceptron);

% Bayesian Metrics
[accuracy_b, precision_b, recall_b, F1_b] = evaluate_metrics(Y_test, Y_pred_bayes);

%% VISUALIZATION

figure;
hold on;
gscatter(X(:, 1), X(:, 2), Y, 'rb', 'os');
legend('Setosa', 'Versicolor');

% Decision boundary for Perceptron
x_vals = linspace(min(X(:,1)), max(X(:,1)), 100);
y_vals = -(w(2)/w(3)) * x_vals - (w(1)/w(3));
plot(x_vals, y_vals, 'k-', 'LineWidth', 2);

% Decision boundary for Bayesian
syms x y;
f = (1/sqrt(det(cov_0))) * exp(-0.5 * ([x y] - mu_0) * inv(cov_0) * ([x y] - mu_0)') * prior_0 ...
    == (1/sqrt(det(cov_1))) * exp(-0.5 * ([x y] - mu_1) * inv(cov_1) * ([x y] - mu_1)') * prior_1;
fimplicit(f, [min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))], 'b-', 'LineWidth', 2);

% Title box
% Adding a legend with plot symbols and boundaries
legend('Setosa', 'Versicolor', 'Perceptron Boundary', 'Bayesian Boundary', 'Location', 'Best');

%% METRICS DISPLAY
disp('Perceptron Classifier Performance:');
disp(['Accuracy: ', num2str(accuracy_p)]);
disp(['Precision: ', num2str(precision_p)]);
disp(['Recall: ', num2str(recall_p)]);
disp(['F1 Score: ', num2str(F1_p)]);

disp('Bayesian Classifier Performance:');
disp(['Accuracy: ', num2str(accuracy_b)]);
disp(['Precision: ', num2str(precision_b)]);
disp(['Recall: ', num2str(recall_b)]);
disp(['F1 Score: ', num2str(F1_b)]);

%% Function to calculate evaluation metrics

function [accuracy, precision, recall, F1] = evaluate_metrics(y_true, y_pred)
    TP = sum((y_pred == 1) & (y_true == 1));
    TN = sum((y_pred == 0) & (y_true == 0));
    FP = sum((y_pred == 1) & (y_true == 0));
    FN = sum((y_pred == 0) & (y_true == 1));
    
    accuracy = (TP + TN) / length(y_true);
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    F1 = 2 * (precision * recall) / (precision + recall);
end
