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
y_test<-as.factor(y_test)
x <- cbind(x_train,y_train)
# Train the model using the training sets and check score
fit <- naiveBayes(y_train ~ ., data = x)
summary(fit)
#Predict Output
predicted= predict(fit,x_test) 
precision <- posPredValue(predicted, y_test, positive="1")
recall <- sensitivity(predicted, y_test, positive="1")

F1 <- (2 * precision * recall) / (precision + recall)
library(ROCR)
pred <- prediction(predicted,y_test)
f.perf <- performance(pred,"f")
plot(f.perf)


