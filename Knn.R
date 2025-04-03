# Load libraries
library(caret)
library(class)
library(glmnet)
library(pROC)
library(ROSE)
library(data.table)
library(e1071)
library(ROCR)


# Function to calculate classification metrics
calculate_classification_metrics <- function(true_labels, predicted_labels) {
  confusion_matrix <- confusionMatrix(predicted_labels, true_labels)
  precision <- confusion_matrix$byClass['Pos Pred Value']
  recall <- confusion_matrix$byClass['Sensitivity']
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  tp <- confusion_matrix$table[2, 2]
  tn <- confusion_matrix$table[1, 1]
  fp <- confusion_matrix$table[1, 2]
  fn <- confusion_matrix$table[2, 1]
  
  cat("True Positives:", tp, "\n")
  cat("True Negatives:", tn, "\n")
  cat("False Positives:", fp, "\n")
  cat("False Negatives:", fn, "\n")
  
  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1 Score:", f1_score, "\n")
  
  return(list(
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    tp = tp,
    tn = tn,
    fp = fp,
    fn = fn
  ))
}

# Function to plot confusion matrix
plot_confusion_matrix <- function(conf_matrix, title) {
  # Plot the confusion matrix
  image(conf_matrix, main = title, col = c("red", "Blue"),
        xlab = "Predicted", ylab = "Actual")
}


####################### KNN ###########################################################

# Part 1: KNN with Date and Location
df_knn <- df_dat_enc

# Feature Engineering
X_features <- df_knn[, !names(df_knn) %in% c("RainTomorrow")]
df_knn$RainTomorrow <- as.factor(df_knn$RainTomorrow)

# Split the data into training and testing sets
set.seed(42)
splitIndex <- createDataPartition(df_knn$RainTomorrow, p = 0.8, list = FALSE)
X_train <- X_features[splitIndex, ]
X_test <- X_features[-splitIndex, ]
y_train <- df_knn$RainTomorrow[splitIndex]
y_test <- df_knn$RainTomorrow[-splitIndex]

# Initialize the KNN classifier with default parameters
knn_model_date_loc <- knn(train = X_train, test = X_test, cl = y_train, k = 5)

# Calculate accuracy
accuracy_date_loc <- sum(knn_model_date_loc == y_test) / length(y_test)
cat("Accuracy with default parameters (Date and Location):", accuracy_date_loc, "\n")

# Confusion Matrix
confusion_matrix_date_loc <- confusionMatrix(knn_model_date_loc, y_test)

# Extract Precision, Recall, and F1 Score
precision_date_loc <- confusion_matrix_date_loc$byClass['Pos Pred Value']
recall_date_loc <- confusion_matrix_date_loc$byClass['Sensitivity']
f1_score_date_loc <- 2 * (precision_date_loc * recall_date_loc) / (precision_date_loc + recall_date_loc)

cat("Precision (Date and Location):", precision_date_loc, "\n")
cat("Recall (Date and Location):", recall_date_loc, "\n")
cat("F1 Score (Date and Location):", f1_score_date_loc, "\n")

# Calculate AUC-ROC
auc_roc_date_loc <- calculate_auc_roc(knn_model_date_loc, as.numeric(y_test))
cat("AUC-ROC (Date and Location):", auc_roc_date_loc, "\n")


# Part 2: KNN without Date and Location - Basic Model
df_knn_basic <- df_loc_dat

# Feature Engineering
X_features_basic <- df_knn_basic[, !names(df_knn_basic) %in% c("RainTomorrow")]
df_knn_basic$RainTomorrow <- as.factor(df_knn_basic$RainTomorrow)

# Split the data into training and testing sets
set.seed(42)
splitIndex_basic <- createDataPartition(df_knn_basic$RainTomorrow, p = 0.8, list = FALSE)
X_train_basic <- X_features_basic[splitIndex_basic, ]
X_test_basic <- X_features_basic[-splitIndex_basic, ]
y_train_basic <- df_knn_basic$RainTomorrow[splitIndex_basic]
y_test_basic <- df_knn_basic$RainTomorrow[-splitIndex_basic]

# Initialize the KNN classifier with default parameters
knn_model_basic <- knn(train = X_train_basic, test = X_test_basic, cl = y_train_basic, k = 5)

# Calculate accuracy
accuracy_basic <- sum(knn_model_basic == y_test_basic) / length(y_test_basic)
cat("Accuracy with default parameters (Basic Model):", accuracy_basic, "\n")

# Confusion Matrix
confusion_matrix_basic <- confusionMatrix(knn_model_basic, y_test_basic)

# Extract Precision, Recall, and F1 Score
precision_basic <- confusion_matrix_basic$byClass['Pos Pred Value']
recall_basic <- confusion_matrix_basic$byClass['Sensitivity']
f1_score_basic <- 2 * (precision_basic * recall_basic) / (precision_basic + recall_basic)

cat("Precision (Basic Model):", precision_basic, "\n")
cat("Recall (Basic Model):", recall_basic, "\n")
cat("F1 Score (Basic Model):", f1_score_basic, "\n")

# Calculate AUC-ROC
auc_roc_basic <- calculate_auc_roc(knn_model_basic, as.numeric(y_test_basic))
cat("AUC-ROC (Basic Model):", auc_roc_basic, "\n")

# Part 3: Hyperparameter Tuning
param_grid_hyperparam <- expand.grid(k = c(3, 5, 7, 9, 11))
ctrl_hyperparam <- trainControl(method = "cv", number = 5)
knn_tune_hyperparam <- train(x = X_train, y = y_train, method = "knn", trControl = ctrl_hyperparam, tuneGrid = param_grid_hyperparam)

# Get the best parameters and the best accuracy
best_params_hyperparam <- knn_tune_hyperparam$bestTune
best_accuracy_hyperparam <- knn_tune_hyperparam$results$Accuracy[which.max(knn_tune_hyperparam$results$Accuracy)]
best_k_hyperparam <- best_params_hyperparam$k

cat("Best Accuracy after Hyperparameter Tuning:", best_accuracy_hyperparam, "\n")

# Initialize the KNN classifier with the best 'k' parameter
knn_model_hyperparam <- knn(train = X_train, test = X_test, cl = y_train, k = best_k_hyperparam)

# Calculate the accuracy of the KNN model with the best parameters
accuracy_hyperparam <- sum(knn_model_hyperparam == y_test) / length(y_test)
cat("Accuracy with the best 'k' parameter after Hyperparameter Tuning:", accuracy_hyperparam, "\n")

# Confusion Matrix after Hyperparameter Tuning
confusion_matrix_hyperparam <- confusionMatrix(knn_model_hyperparam, y_test)

# Extract Precision, Recall, and F1 Score after Hyperparameter Tuning
precision_hyperparam <- confusion_matrix_hyperparam$byClass['Pos Pred Value']
recall_hyperparam <- confusion_matrix_hyperparam$byClass['Sensitivity']
f1_score_hyperparam <- 2 * (precision_hyperparam * recall_hyperparam) / (precision_hyperparam + recall_hyperparam)

cat("Precision after Hyperparameter Tuning:", precision_hyperparam, "\n")
cat("Recall after Hyperparameter Tuning:", recall_hyperparam, "\n")
cat("F1 Score after Hyperparameter Tuning:", f1_score_hyperparam, "\n")

# Calculate AUC-ROC after Hyperparameter Tuning
auc_roc_hyperparam <- calculate_auc_roc(knn_model_hyperparam, as.numeric(y_test))
cat("AUC-ROC after Hyperparameter Tuning:", auc_roc_hyperparam, "\n")



# Part 4: KNN without Date and Location - Log Transform - Hyperparameter Tuning - All Features
df_knn_all <- df_loc_dat
df_knn_all$constant <- 1e-6
skewed_vars_all <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')

# Applying log transformation to skewed variables
df_knn_all[paste0(skewed_vars_all, '_log')] <- lapply(df_knn_all[skewed_vars_all], function(x) log(x + df_knn_all$constant))

# Selecting features and target variable
X_all <- df_knn_all[, !names(df_knn_all) %in% c("RainTomorrow")]
y_all <- df_knn_all$RainTomorrow

# Splitting the dataset into training and testing sets
set.seed(42)
splitIndex_all <- createDataPartition(y_all, p = 0.8, list = FALSE)
X_train_all <- X_all[splitIndex_all, ]
X_test_all <- X_all[-splitIndex_all, ]
y_train_all <- y_all[splitIndex_all]
y_test_all <- y_all[-splitIndex_all]

# Standardizing the features
scaler_all <- preProcess(X_train_all, method = c("center", "scale"))
X_train_all <- predict(scaler_all, X_train_all)
X_test_all <- predict(scaler_all, X_test_all)

# Hyperparameter Tuning
param_grid_all <- expand.grid(k = c(1, 3, 5, 7, 9))
ctrl_all <- trainControl(method = "cv", number = 5)

# Use train function for hyperparameter tuning
knn_tune_all <- train(
  x = X_train_all,
  y = y_train_all,
  method = "knn",
  trControl = ctrl_all,
  tuneGrid = param_grid_all
)

# Get the best parameters and the best accuracy
best_params_all <- knn_tune_all$bestTune
best_accuracy_all <- knn_tune_all$results$Accuracy[which.max(knn_tune_all$results$Accuracy)]
best_k_all <- best_params_all$k

cat("Best k (All Features):", best_k_all, "\n")
cat("Best Accuracy (All Features):", best_accuracy_all, "\n")

# Initialize the KNN classifier with the best 'k' parameter
knn_all_model <- knn(train = X_train_all, test = X_test_all, cl = y_train_all, k = best_k_all)

# Making predictions with the model using all features
y_pred_all <- as.factor(knn_all_model)

# Calculating accuracy
accuracy_all <- sum(y_pred_all == y_test_all) / length(y_test_all)
cat("Accuracy with all features (Hyperparameter Tuning):", accuracy_all, "\n")

# Check unique levels in y_pred_all and y_test_all
unique_levels_pred <- levels(y_pred_all)
unique_levels_test <- levels(y_test_all)

cat("Unique levels in y_pred_all:", unique_levels_pred, "\n")
cat("Unique levels in y_test_all:", unique_levels_test, "\n")


# Set levels of y_test_all based on unique levels in y_pred_all
y_test_all <- factor(y_test_all, levels = unique_levels_pred)

# Now calculate the confusion matrix
confusion_matrix_all <- confusionMatrix(y_pred_all, y_test_all)

# Extract Precision, Recall, and F1 Score
precision_all <- confusion_matrix_all$byClass['Pos Pred Value']
recall_all <- confusion_matrix_all$byClass['Sensitivity']
f1_score_all <- 2 * (precision_all * recall_all) / (precision_all + recall_all)

cat("Precision after Hyperparameter Tuning - All Features:", precision_all, "\n")
cat("Recall after Hyperparameter Tuning - All Features:", recall_all, "\n")
cat("F1 Score after Hyperparameter Tuning - All Features:", f1_score_all, "\n")

# Calculate AUC-ROC
auc_roc_all <- calculate_auc_roc(y_pred_all, as.numeric(y_test_all))
cat("AUC-ROC after Hyperparameter Tuning - All Features:", auc_roc_all, "\n")

# Part 5: KNN without Date and Location - Log transform - Hyperparameter Tuning - Selected Features
df_knn_selected <- df_loc_dat
df_knn_selected$constant <- 1e-6
skewed_vars_selected <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')

# Applying log transformation to skewed variables
df_knn_selected[paste0(skewed_vars_selected, '_log')] <- lapply(df_knn_selected[skewed_vars_selected], function(x) log(x + df_knn_selected$constant))

# Selecting features and target variable
selected_vars <- c('MinTemp', 'MaxTemp', 'Rainfall_log', 'Evaporation_log', 'Sunshine_log', 'WindGustSpeed_log', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm')
X_selected <- df_knn_selected[, selected_vars]
y_selected <- df_knn_selected$RainTomorrow

# Splitting the dataset into training and testing sets
set.seed(42)
splitIndex_selected <- createDataPartition(y_selected, p = 0.8, list = FALSE)
X_train_selected <- X_selected[splitIndex_selected, ]
X_test_selected <- X_selected[-splitIndex_selected, ]
y_train_selected <- y_selected[splitIndex_selected]
y_test_selected <- y_selected[-splitIndex_selected]

# Standardizing the features
scaler_selected <- preProcess(X_train_selected, method = c("center", "scale"))
X_train_selected <- predict(scaler_selected, X_train_selected)
X_test_selected <- predict(scaler_selected, X_test_selected)

# Convert outcome variable to factor
y_train_selected <- as.factor(y_train_selected)

# Hyperparameter Tuning - Selected Features
param_grid_selected <- expand.grid(k = c(1, 3, 5, 7, 9))
ctrl_selected <- trainControl(method = "cv", number = 5)
knn_tune_selected <- train(x = X_train_selected, y = y_train_selected, method = "knn", trControl = ctrl_selected, tuneGrid = param_grid_selected)


# Get the best parameters and the best accuracy
best_params_selected <- knn_tune_selected$bestTune
best_accuracy_selected <- knn_tune_selected$results$Accuracy[which.max(knn_tune_selected$results$Accuracy)]
best_k_selected <- best_params_selected$k

cat("Results after Hyperparameter Tuning - Selected Features:\n")
cat("Best k:", best_k_selected, "\n")
cat("Best Accuracy:", best_accuracy_selected, "\n")

# Initialize the KNN classifier with the best 'k' parameter
knn_selected_model <- knn(train = X_train_selected, test = X_test_selected, cl = y_train_selected, k = best_k_selected)

# Making predictions with the best model
y_pred_selected <- as.factor(knn_selected_model)
y_test_selected <- factor(y_test_selected, levels = unique_levels_pred)

# Confusion Matrix after Hyperparameter Tuning - Selected Features
confusion_matrix_selected <- confusionMatrix(y_pred_selected, y_test_selected)

# Extract Precision, Recall, and F1 Score after Hyperparameter Tuning - Selected Features
precision_selected <- confusion_matrix_selected$byClass['Pos Pred Value']
recall_selected <- confusion_matrix_selected$byClass['Sensitivity']
f1_score_selected <- 2 * (precision_selected * recall_selected) / (precision_selected + recall_selected)

cat("Precision after Hyperparameter Tuning - Selected Features:", precision_selected, "\n")
cat("Recall after Hyperparameter Tuning - Selected Features:", recall_selected, "\n")
cat("F1 Score after Hyperparameter Tuning - Selected Features:", f1_score_selected, "\n")

# Calculate AUC-ROC
auc_roc_selected <- calculate_auc_roc(y_pred_selected, as.numeric(y_test_selected))
cat("AUC-ROC after Hyperparameter Tuning - Selected Features:", auc_roc_selected, "\n")



# Part 6: KNN with SMOTE Balanced Dataset
# Creating a new dataframe with selected features
df_selected <- df_knn_selected[, c(selected_vars, "RainTomorrow")]

# Perform SMOTE oversampling on the dataset
df_SMOTE_balanced <- ovun.sample(RainTomorrow ~ . , data = df_selected, seed = 1)$data

# Selecting features and target variable
X_balanced <- df_SMOTE_balanced[, selected_vars]
y_balanced <- df_SMOTE_balanced$RainTomorrow

# Splitting the dataset into training and testing sets
set.seed(42)
splitIndex_balanced <- createDataPartition(y_balanced, p = 0.8, list = FALSE)
X_train_balanced <- X_balanced[splitIndex_balanced, ]
X_test_balanced <- X_balanced[-splitIndex_balanced, ]
y_train_balanced <- y_balanced[splitIndex_balanced]
y_test_balanced <- y_balanced[-splitIndex_balanced]

# Initialize the KNN classifier with default parameters
knn_model_balanced <- knn(train = X_train_balanced, test = X_test_balanced, cl = y_train_balanced, k = 5)

# Calculate the accuracy of the KNN model on the balanced dataset
accuracy_balanced <- sum(knn_model_balanced == y_test_balanced) / length(y_test_balanced)
cat("Accuracy on the balanced dataset:", accuracy_balanced, "\n")

y_test_balanced <- factor(y_test_balanced, levels = levels(as.factor(knn_model_balanced)))

# Confusion Matrix on the balanced dataset
confusion_matrix_balanced <- confusionMatrix(knn_model_balanced, y_test_balanced)

# Extract Precision, Recall, and F1 Score on the balanced dataset
precision_balanced <- confusion_matrix_balanced$byClass['Pos Pred Value']
recall_balanced <- confusion_matrix_balanced$byClass['Sensitivity']
f1_score_balanced <- 2 * (precision_balanced * recall_balanced) / (precision_balanced + recall_balanced)

cat("Precision on the balanced dataset:", precision_balanced, "\n")
cat("Recall on the balanced dataset:", recall_balanced, "\n")
cat("F1 Score on the balanced dataset:", f1_score_balanced, "\n")

# Calculate AUC-ROC on the balanced dataset
auc_roc_balanced <- calculate_auc_roc(knn_model_balanced, as.numeric(y_test_balanced))
cat("AUC-ROC on the balanced dataset:", auc_roc_balanced, "\n")
 
############################# LOGISTIC REGRESSION ###################################################

# Part 1 - Logistic Regression without date and location
df_part1 <- df_loc_dat
df_part1$RainTomorrow <- as.factor(df_part1$RainTomorrow)

set.seed(42)
split_index_part1 <- createDataPartition(df_part1$RainTomorrow, p = 0.8, list = FALSE)
X_train_part1 <- df_part1[split_index_part1, !(names(df_part1) %in% c("RainTomorrow", "Date", "Location"))]
X_test_part1 <- df_part1[-split_index_part1, !(names(df_part1) %in% c("RainTomorrow", "Date", "Location"))]
y_train_part1 <- df_part1$RainTomorrow[split_index_part1]
y_test_part1 <- df_part1$RainTomorrow[-split_index_part1]

# Train the logistic regression model
logistic_model_part1 <- glmnet(as.matrix(X_train_part1), as.factor(y_train_part1), family = "binomial", alpha = 1, lambda = 0)

# Predict on the test set
predicted_probabilities_part1 <- predict(logistic_model_part1, newx = as.matrix(X_test_part1), type = "response")
predicted_labels_part1 <- as.factor(ifelse(predicted_probabilities_part1 > 0.5, 1, 0))

accuracy_part1 <- sum(predicted_labels_part1 == y_test_part1) / length(y_test_part1)
cat("Accuracy with default parameters:", accuracy_part1, "\n")

metrics_part1 <- calculate_classification_metrics(y_test_part1, predicted_labels_part1)

# Calculate AUC-ROC for the logistic regression model in Part 1
roc_curve_part1 <- multiclass.roc(as.numeric(y_test_part1), as.numeric(predicted_probabilities_part1))
auc_roc_part1 <- auc(roc_curve_part1)
cat("AUC-ROC with default parameters:", auc_roc_part1, "\n")

# Part 2 - Logistic Regression without date and location with log transformation
df_part2 <- df_loc_dat
skewed_vars_part2 <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')
df_part2[paste0(skewed_vars_part2, '_log')] <- lapply(df_part2[skewed_vars_part2], function(x) log(x + 1))
df_part2$RainTomorrow <- as.factor(df_part2$RainTomorrow)

set.seed(42)
split_index_part2 <- createDataPartition(df_part2$RainTomorrow, p = 0.8, list = FALSE)
X_train_part2 <- df_part2[split_index_part2, !(names(df_part2) %in% c("RainTomorrow", "Date", "Location"))]
X_test_part2 <- df_part2[-split_index_part2, !(names(df_part2) %in% c("RainTomorrow", "Date", "Location"))]
y_train_part2 <- df_part2$RainTomorrow[split_index_part2]
y_test_part2 <- df_part2$RainTomorrow[-split_index_part2]

# Train the logistic regression model
logistic_model_part2 <- glmnet(as.matrix(X_train_part2), as.factor(y_train_part2), family = "binomial", alpha = 1, lambda = 0)

# Apply log transformation to skewed variables in the test set
X_test_part2[paste0(skewed_vars_part2, '_log')] <- lapply(X_test_part2[skewed_vars_part2], function(x) log(x + 1))

# Predict on the test set
predicted_probabilities_part2 <- predict(logistic_model_part2, newx = as.matrix(X_test_part2), type = "response")
predicted_labels_part2 <- as.factor(ifelse(predicted_probabilities_part2 > 0.5, 1, 0))

accuracy_part2 <- sum(predicted_labels_part2 == y_test_part2) / length(y_test_part2)
cat("Accuracy with log-transformed:", accuracy_part2, "\n")

metrics_part2 <- calculate_classification_metrics(y_test_part2, predicted_labels_part2)


# Calculate AUC-ROC for the logistic regression model in Part 2
roc_curve_part2 <- multiclass.roc(as.numeric(y_test_part2), as.numeric(predicted_probabilities_part2))
auc_roc_part2 <- auc(roc_curve_part2)
cat("AUC-ROC with default parameters:", auc_roc_part2, "\n")


# Part 3 - Logistic Regression without date and location with log transformation and hyperparameter tuning
df_part3 <- df_loc_dat

# Apply log transformation to skewed variables
skewed_vars_part3 <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')
df_part3[paste0(skewed_vars_part3, '_log')] <- lapply(df_part3[skewed_vars_part3], function(x) log(x + 1))

df_part3$RainTomorrow <- as.factor(df_part3$RainTomorrow)

# Define a grid of alpha and lambda values for hyperparameter tuning
param_grid <- expand.grid(
  alpha = seq(0, 1, length = 5),  # alpha values from 0 to 1
  lambda = seq(0.0001, 1, length = 10)  # lambda values from a small positive value to 1
)


# Splitting the data into features and target variable
X_part3 <- df_part3[, !(names(df_part3) %in% c("RainTomorrow", "Date", "Location"))]
y_part3 <- df_part3$RainTomorrow

# Splitting the data into training and testing sets
set.seed(42)
split_index_part3 <- createDataPartition(y_part3, p = 0.8, list = FALSE)
X_train_part3 <- X_part3[split_index_part3, ]
X_test_part3 <- X_part3[-split_index_part3, ]
y_train_part3 <- y_part3[split_index_part3]
y_test_part3 <- y_part3[-split_index_part3]

# Train the logistic regression model with hyperparameter tuning
ctrl_part3 <- trainControl(method = "cv", number = 5)
logistic_tune_part3 <- train(
  x = X_train_part3, 
  y = y_train_part3, 
  method = "glmnet", 
  trControl = ctrl_part3, 
  tuneGrid = param_grid,
  family = "binomial"
)

# Getting the best hyperparameters
best_params_part3 <- logistic_tune_part3$bestTune
best_alpha_part3 <- best_params_part3$alpha
best_lambda_part3 <- best_params_part3$lambda

# Train the logistic regression model with the best hyperparameters
final_logistic_model_part3 <- glmnet(as.matrix(X_train_part3), as.factor(y_train_part3), family = "binomial", alpha = best_alpha_part3, lambda = best_lambda_part3)

# Apply log transformation to skewed variables in the test set
X_test_part3[paste0(skewed_vars_part3, '_log')] <- lapply(X_test_part3[skewed_vars_part3], function(x) log(x + 1))

# Predict on the test set
predicted_probabilities_part3 <- predict(final_logistic_model_part3, newx = as.matrix(X_test_part3), type = "response")
predicted_labels_part3 <- as.factor(ifelse(predicted_probabilities_part3 > 0.5, 1, 0))

# Evaluate the model
accuracy_hyperparam_part3 <- sum(predicted_labels_part3 == y_test_part3) / length(y_test_part3)
cat("Accuracy on the test set:", accuracy_hyperparam_part3, "\n")

# Calculate additional metrics
metrics_part3 <- calculate_classification_metrics(y_test_part3, predicted_labels_part3)

# Calculate AUC-ROC for the logistic regression model
roc_curve_part3 <- multiclass.roc(y_test_part3, as.numeric(predicted_probabilities_part3))
auc_roc_part3 <- auc(roc_curve_part3)
cat("AUC-ROC with default parameters:", auc_roc_part3, "\n")

# Part 4 - Logistic Regression without date and location with log transformation, hyperparameter tuning, and SMOTE balancing
df_part4 <- df_loc_dat

# Apply log transformation to skewed variables
skewed_vars_part4 <- c('Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed')
df_part4[paste0(skewed_vars_part4, '_log')] <- lapply(df_part4[skewed_vars_part4], function(x) log(x + 1))

df_part4$RainTomorrow <- as.factor(df_part4$RainTomorrow)

# Splitting the data into features and target variable
X_part4 <- df_part4[, !(names(df_part4) %in% c("RainTomorrow", "Date", "Location"))]
y_part4 <- df_part4$RainTomorrow

# Splitting the data into training and testing sets
set.seed(42)
split_index_part4 <- createDataPartition(y_part4, p = 0.8, list = FALSE)
X_train_part4 <- X_part4[split_index_part4, ]
X_test_part4 <- X_part4[-split_index_part4, ]
y_train_part4 <- y_part4[split_index_part4]
y_test_part4 <- y_part4[-split_index_part4]

# Add SMOTE to balance the training set
df_SMOTE_balanced_part4 <- ovun.sample(RainTomorrow ~ . , data = df_part4, seed = 1)$data
X_train_part4 <- df_SMOTE_balanced_part4[, !(names(df_SMOTE_balanced_part4) %in% c("RainTomorrow"))]
y_train_part4 <- df_SMOTE_balanced_part4$RainTomorrow

# Train the logistic regression model with hyperparameter tuning
ctrl_part4 <- trainControl(method = "cv", number = 5)
logistic_tune_part4 <- train(
  x = X_train_part4, 
  y = y_train_part4, 
  method = "glmnet", 
  trControl = ctrl_part4, 
  tuneGrid = param_grid,
  family = "binomial"
)

# Getting the best hyperparameters
best_params_part4 <- logistic_tune_part4$bestTune
best_alpha_part4 <- best_params_part4$alpha
best_lambda_part4 <- best_params_part4$lambda

# Train the logistic regression model with the best hyperparameters
final_logistic_model_part4 <- glmnet(as.matrix(X_train_part4), as.factor(y_train_part4), family = "binomial", alpha = best_alpha_part4, lambda = best_lambda_part4)

# Apply log transformation to skewed variables in the test set
X_test_part4[paste0(skewed_vars_part4, '_log')] <- lapply(X_test_part4[skewed_vars_part4], function(x) log(x + 1))

# Predict on the test set
predicted_probabilities_part4 <- predict(final_logistic_model_part4, newx = as.matrix(X_test_part4), type = "response")
predicted_labels_part4 <- as.factor(ifelse(predicted_probabilities_part4 > 0.5, 1, 0))

# Evaluate the model
accuracy_hyperparam_part4 <- sum(predicted_labels_part4 == y_test_part4) / length(y_test_part4)
cat("Accuracy on the test set:", accuracy_hyperparam_part4, "\n")

# Calculate additional metrics
metrics_part4 <- calculate_classification_metrics(y_test_part4, predicted_labels_part4)

# Calculate AUC-ROC for the logistic regression model
roc_curve_part4 <- multiclass.roc(as.numeric(y_test_part4), as.numeric(predicted_probabilities_part4))
auc_roc_part4 <- auc(roc_curve_part4)
cat("AUC-ROC with default parameters:", auc_roc_part4, "\n")


# Part 5 - Logistic Regression without date and location with SMOTE balancing
df_part5 <- df_loc_dat
df_part5$RainTomorrow <- as.factor(df_part5$RainTomorrow)

set.seed(42)
split_index_part5 <- createDataPartition(df_part5$RainTomorrow, p = 0.8, list = FALSE)
X_train_part5 <- df_part5[split_index_part5, !(names(df_part5) %in% c("RainTomorrow", "Date", "Location"))]
X_test_part5 <- df_part5[-split_index_part5, !(names(df_part5) %in% c("RainTomorrow", "Date", "Location"))]
y_train_part5 <- df_part5$RainTomorrow[split_index_part5]
y_test_part5 <- df_part5$RainTomorrow[-split_index_part5]

# Add SMOTE to balance the training set
df_SMOTE_balanced_part5 <- ovun.sample(RainTomorrow ~ . , data = df_part5, seed = 1)$data
X_train_part5 <- df_SMOTE_balanced_part5[, !(names(df_SMOTE_balanced_part5) %in% c("RainTomorrow"))]
y_train_part5 <- as.factor(df_SMOTE_balanced_part5$RainTomorrow)

# Apply SMOTE to the test set to match the number of variables
df_SMOTE_test_part5 <- ovun.sample(RainTomorrow ~ . , data = df_part5[-split_index_part5, ], seed = 1)$data
X_test_part5 <- df_SMOTE_test_part5[, !(names(df_SMOTE_test_part5) %in% c("RainTomorrow"))]
y_test_part5 <- as.factor(df_SMOTE_test_part5$RainTomorrow)

# Train the logistic regression model
logistic_model_part5 <- glmnet(as.matrix(X_train_part5), y_train_part5, family = "binomial", alpha = 1, lambda = 0)

# Predict on the test set
predicted_probabilities_part5 <- predict(logistic_model_part5, newx = as.matrix(X_test_part5), type = "response")
predicted_labels_part5 <- as.factor(ifelse(predicted_probabilities_part5 > 0.5, 1, 0))

accuracy_part5 <- sum(predicted_labels_part5 == y_test_part5) / length(y_test_part5)
cat("Accuracy with default parameters:", accuracy_part5, "\n")

metrics_part5 <- calculate_classification_metrics(y_test_part5, predicted_labels_part5)

# Calculate AUC-ROC for the logistic regression model
roc_curve_part5 <- multiclass.roc(as.numeric(y_test_part5), as.numeric(predicted_probabilities_part5))
auc_roc_part5 <- auc(roc_curve_part5)
cat("AUC-ROC with default parameters:", auc_roc_part5, "\n")





