
library(caret)
Data_train <- read.csv("D:/Priyanka/Coursera/8.MachineLearning/pml-training.csv")
inTrain <- createDataPartition(Data_train$classe, p=0.7, list = FALSE)
TrainSet <- Data_train[inTrain,]
TestSet <- Data_train[-inTrain,]
dim(TrainSet)
dim(TestSet)

nzv <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[,-nzv]
TestSet <- TestSet[,-nzv]
dim(TrainSet)
dim(TestSet)

AllNA <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA == FALSE]
TestSet <- TestSet[,AllNA == FALSE]
dim(TrainSet)
dim(TestSet)

TrainSet <- TrainSet[,-(1:4)]
TestSet <- TestSet[,-(1:4)]

library(randomForest)
ControlRF <- trainControl(method = "cv", number = 3)
m_rf <- train(classe ~., data = TrainSet, method = "rf", trControl = ControlRF)
m_rf$finalModel

pred_rf <- predict(m_rf, TestSet)
conf_mat_rf <- confusionMatrix(pred_rf, TestSet$classe)
conf_mat_rf

library(rpart)
library(rpart.plot)
library(rattle)
m_dt <- rpart(classe ~., data = TrainSet, method = "class")
fancyRpartPlot(m_dt)

pred_dt <- predict(m_dt, TestSet, type ="class")
conf_mat_dt <- confusionMatrix(pred_dt, TestSet$classe)
conf_mat_dt

ControlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
m_gbm <- train(classe ~., data = TrainSet, method = "gbm", trControl = ControlGBM)
m_gbm$finalModel

pred_gbm <- predict(m_gbm, TestSet)
conf_mat_gbm <- confusionMatrix(pred_gbm, TestSet$classe)
conf_mat_gbm


Data_test <- read.csv("D:/Priyanka/Coursera/8.MachineLearning/pml-testing.csv")
Pred_test <- predict(m_rf, Data_test)
