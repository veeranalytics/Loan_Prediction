# Load Libraries
library(shiny)
library(shinydashboard)
library(shinyjs)
library(ggplot2)
library(ggthemes)
library(plotly)
library(curl)
library(RCurl)
library(DT)
library(caret)
library(plyr)
library(dplyr)
library(caret)
library(randomForest)
library(stringr)

# Get Data
loan_df <- read.csv("https://raw.githubusercontent.com/veeranalytics/Credit-Risk-Model/master/loan_data.csv")

set.seed(123)

# Take a glimpse of data
str(loan_df)

# Convert Numerical Factor Variables to numbers
loan_df$annual_inc <- as.numeric(loan_df$annual_inc)
loan_df$loan_amnt <- as.numeric(loan_df$loan_amnt)
loan_df$int_rate <- as.numeric(loan_df$int_rate)

# Look at missing values
sapply(loan_df, function(x) sum(is.na(x)))

# Missing Value Imputation using median
loan_df$emp_length[which(is.na(loan_df$emp_length))] <- median(loan_df$emp_length, na.rm = TRUE)

# create dataset-- of current loan customers for credit risk modeling
loan_current <- loan_df %>%
  filter(loan_status %in% c("Current"))

# create dataset-- of current loan customers at warning for credit risk modeling
loan_warning <- loan_df %>%
  filter(loan_status %in% c("In Grace Period","Late (16-30 days)","Late (31-120 days)"))

# Create a copy of loan dataset for modeling and remove loan_status are
# "Current" and Warnings ("In Grace Period","Late (16-30 days)","Late (31-120 days)") 
# And ID variables
loan_data <- loan_df %>%
  filter(!(loan_status %in% c("Current","In Grace Period","Late (16-30 days)","Late (31-120 days)")))

# Create 02 categories for loan status
loan_data$loan_status2 <- rep(NA, length(loan_data$loan_status))
loan_data$loan_status2[which(loan_data$loan_status == "Charged Off")] <- "Yes"
loan_data$loan_status2[which(loan_data$loan_status == "Default")] <- "Yes"
loan_data$loan_status2[which(loan_data$loan_status == "Fully Paid")] <- "No"
loan_data$loan_status <- NULL
colnames(loan_data)[14] <- c("loan_status")
loan_data$loan_status <- as.factor(loan_data$loan_status)

# Removing loan_id and member id columns-- as will not pbe useful for the modeling process
model_data <- loan_data %>%
  select(-loan_id, -member_id)

# Look at the loan_status 
n_default <- summary(model_data$loan_status)[2]
model_data$index <- as.numeric(row.names(model_data))

# The data is unbalanced-- Performing Undersampling
# Create sample dataset
not_default <- model_data %>%
  filter(loan_status == "No") %>%
  sample_n(n_default)

is_default <- model_data %>%
  filter(loan_status == "Yes")

full_sample <- rbind(not_default, is_default) %>%
  arrange(index)

full_sample$index <- NULL
str(full_sample)

#Remove result column and categorical columns
default_numeric <- full_sample %>%
  select(-home_ownership, -term, -marital, -job, -loan_status)

# Find any highly correlated predictors and remove
# Highly correlated predictors create instability in the model so one of the two is removed.
high_cor_cols <- findCorrelation(cor(default_numeric), cutoff = .1, verbose = TRUE, 
                                 names = TRUE, exact = TRUE)
high_cor_cols # no predictors are highly co-related, so will not drop any variable

# Pre-processing the full dataset for modelling
preproc_model <- preProcess(full_sample[, -1], 
                            method = c("center", "scale", "nzv"))

fraud_preproc <- predict(preproc_model, newdata = full_sample[, -1])

# Bind the results to?the pre-processed data
fraud_pp_w_result <- cbind(loan_status = full_sample$loan_status, fraud_preproc)

# Split sample into train and test sets for Random Forest
in_train_rf <- createDataPartition(y = fraud_pp_w_result$loan_status, p = .75, 
                                list = FALSE) 
train_rf <- full_sample[in_train_rf, ] 
test_rf <- full_sample[-in_train_rf, ]


# Split sample into train and test sets for Decision Trees
in_train <- createDataPartition(y = full_sample$loan_status, p = .75, 
                                list = FALSE) 
train <- full_sample[in_train, ] 
test <- full_sample[-in_train, ] 


# Modeling Part
# Set general parameters
# Random Forest Model
set.seed(123)

control <- trainControl(method = "repeatedcv",
                        number = 10,
                        repeats = 3,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
grid <- expand.grid(.mtry = 5, .ntree = seq(25, 150, by = 25))

start_time <- Sys.time()
rf_model <- train(loan_status ~ ., 
                  data = train_rf, 
                  method="rf", 
                  metric = "ROC", 
                  TuneGrid = grid, 
                  trControl=control)
end_time <- Sys.time()
end_time - start_time

# Print and Plot the model
print(rf_model$finalModel)
plot(rf_model$finalModel)

# Plot variable importance
varImpPlot(rf_model$finalModel)

# Predict on Training set
rf_train_pred <- predict(rf_model, train)
confusionMatrix(train$loan_status, rf_train_pred, positive = "Yes")

# Predict on Test set
rf_test_pred <- predict(rf_model, test)
confusionMatrix(test$loan_status, rf_test_pred, positive = "Yes")

# Plot the ROC curve
rf_probs <- predict(rf_model, test, type = "prob")
rf_ROC <- roc(response = test$loan_status, 
              predictor = rf_probs$Yes, 
              levels = levels(test$loan_status))
plot(rf_ROC)
auc(rf_ROC)