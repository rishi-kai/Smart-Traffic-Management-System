# Load necessary libraries
library(caret)
library(dplyr)
library(readr)

# Load the dataset
data <- read_csv('traffic_data.csv')

# Select features and target variable
features <- c('time_of_day', 'day_of_week', 'weather_condition')  # Add your relevant features here
target <- 'traffic_volume'

# Split the data into features (X) and target (y)
X <- data %>% select(all_of(features))
y <- data[[target]]

# Preprocessing: Handle categorical and numerical features
preprocess_params <- preProcess(X, method = c('center', 'scale', 'dummyVars'))

# Apply preprocessing
X_processed <- predict(preprocess_params, X)

# Split the data into training and testing sets
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_processed[train_index, ]
X_test <- X_processed[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Train the model using cross-validation
train_control <- trainControl(method = 'cv', number = 5)
model <- train(X_train, y_train, method = 'lm', trControl = train_control)

# Print cross-validation results
print(model)

# Make predictions on the test set
y_pred <- predict(model, X_test)

# Evaluate the model
mse <- mean((y_test - y_pred)^2)
r2 <- cor(y_test, y_pred)^2
mae <- mean(abs(y_test - y_pred))

cat('Mean Squared Error:', mse, '\n')
cat('R-squared:', r2, '\n')
cat('Mean Absolute Error:', mae, '\n')

# Optional: Print model coefficients
cat('Coefficients:', coef(model$finalModel), '\n')
cat('Intercept:', coef(model$finalModel)[1], '\n')
