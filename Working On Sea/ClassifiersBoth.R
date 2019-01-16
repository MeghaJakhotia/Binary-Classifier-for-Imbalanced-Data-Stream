library("RMOA")

sea_data <- read.csv("sea.txt", sep= ",",col.names = c("V1","V2","V3","class"))
sea_data$class <- as.factor(sea_data$class)
sea_datastream <- datastream_dataframe(data=sea_data)

trainset <- sea_datastream$get_points(sea_datastream, n = 5000, outofpoints = c("stop", "warn", "ignore"))
trainset <- datastream_dataframe(data=trainset)

ctrl<-MOAoptions(model="NaiveBayes")
mymodel<-NaiveBayes(control=ctrl)
naive_model <- trainMOA(model = mymodel,class ~ ., data = trainset)
naive_model

testset <- sea_datastream$get_points(sea_datastream, n = 5000, outofpoints = c("stop", "warn", "ignore"))

scores <- predict(naive_model, newdata=testset[, c("V1","V2","V3")], type="response")
str(scores)
table(scores, testset$class)

scores2 <- predict(naive_model, newdata=testset, type="response")
head(scores2)

tmodel <- trainMOA(model = mymodel,trace = TRUE, formula = class ~ V1+V2+V3, data = trainset,options = list(maxruntime=0.1))
scores <- predict(tmodel, newdata=testset[, c("V1","V2","V3")], type="response")
str(scores)

library('caret')
library('e1071')
Confuse <- confusionMatrix(scores,testset$class)

#Ensemble
emodel1<-OzaBoost(baseLearner="trees.HoeffdingTree",ensembleSize=30)
ozaboost_model <- trainMOA(model = emodel1,class ~ ., data = trainset)
ozaboost_model
testset2 <- sea_datastream$get_points(sea_datastream, n = 5000, outofpoints = c("stop", "warn", "ignore"))
scores2 <- predict(ozaboost_model, newdata=testset2[, c("V1","V2","V3")], type="response")
str(scores2)
Confuse_ozaboost<-confusionMatrix(scores2,testset2$class)
Confuse_ozaboost
