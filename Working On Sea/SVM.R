setwd("E:/Megha/BE project")
dt <- read.csv("sea.txt")
library('caret')
library('e1071')
index <- createDataPartition(dt$X6.677259, p=0.75, list=FALSE)
trainSet <- dt[ index,]
testSet <- dt[-index,]
x_train <- trainSet[,c(1,2,3)] 
y_train <- trainSet[,4]
x_test <- testSet[,c(1,2,3)]
y_test <- testSet[,4]
x <- cbind(x_train,y_train)
# Train the model using the training sets and check score
fit <- svm(y_train ~ ., data = x)
summary(fit)
#Predict Output
predicted= predict(fit,x_test) 
library(ROCR)
pred <- prediction(predicted,y_test)
f.perf <- performance(pred,"f")
plot(f.perf)
confusionMatrix(data=as.numeric(predicted>0.5),reference = y_test, positive="1")
