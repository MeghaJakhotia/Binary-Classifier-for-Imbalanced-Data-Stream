#Reading the input data set
dt <- read.csv("Input.txt",col.names = c("V1","v2", "V3", "V4","V5","V6","V7"))
library('caret')
library('e1071')
#We need to set the seed so that the index values generated for partition are constant on every run
set.seed(0)

#Dividing the input data sets into 2 parts (Train and Test) and storing it in separate files 
index <- createDataPartition(dt$V7, p=0.5, list=FALSE)
write.table(dt[index,],"Train.txt",sep=",",row.names=FALSE,col.names = TRUE)
write.table(dt[-index,],"Test.txt",sep=",",row.names=FALSE,col.names = TRUE)

#Storing the class labels seprately as they would be available to the clustering algorithm after a delay
write.table(dt[index,4],"ClassLabels.txt",sep=",",row.names = FALSE,col.names="class")


library(RMOA)
#--- BUILDING A CLASSIFIER MODEL ---
#Reading the data as a data frame
sea_data <- read.csv("Train.txt", sep= ",",col.names = c("V1","V2","V3","V4","V5","V6","V7"))
sea_data$V7<-as.factor(sea_data$V7)

#Converted to data streams
sea_datastream <- datastream_dataframe(data=sea_data)
#Forming the train set to be given as input to the classifier model
trainset <- sea_datastream$get_points(sea_datastream, n = 5592, outofpoints = c("stop", "warn", "ignore"))
trainset <- datastream_dataframe(data=trainset)

#Classifier Model
ctrl<-MOAoptions(model="NaiveBayes")
mymodel<-NaiveBayes(control=ctrl)
naive_model <- trainMOA(model = mymodel,V7 ~ ., data = trainset)
naive_model

#Test set and predictions
sea_test_data <- read.csv("Test.txt", sep= ",",col.names = c("V1","V2","V3","V4","V5","V6","V7"))
sea_test_data$V7<-as.factor(sea_test_data$V7)
sea_test_datastream <- datastream_dataframe(data=sea_test_data)

testset <- sea_test_datastream$get_points(sea_test_datastream, n = 5591, outofpoints = c("stop", "warn", "ignore"))
scores <- predict(naive_model, newdata=sea_test_data[, c("V1","V2","V3","V4","V5","V6")], type="response")
#str(scores)
#table(scores, testset$class)

Confuse<-confusionMatrix(data=scores,testset[,7], positive="1")
Confuse
tp<-Confuse$table[2,2] #Positive="1"
fp<-Confuse$table[1,2]
tn<-Confuse$table[1,1]

fn<-Confuse$table[2,1]
preci<-tp/(tp+fp)
recal<-tp/(tp+fn)
fscore<-2*((preci*recal)/(preci+recal))

#Ensemble model - OzaBoost
sea_test_datastream$reset()
trainset$reset()
#a$reset()
emodel1<-OzaBoost(baseLearner="trees.HoeffdingTree",ensembleSize=30)
ozaboost_model <- trainMOA(model = emodel1,V7 ~ ., data = trainset)
ozaboost_model
testset2 <- sea_test_datastream$get_points(sea_test_datastream, n = 5591, outofpoints = c("stop", "warn", "ignore"))
scores2 <- predict(ozaboost_model, newdata=testset2[, c("V1","V2","V3","V4","V5","V6")], type="response")
str(scores2)
Confuse_ozaboost<-confusionMatrix(scores2,testset2[,7], positive="1")
Confuse_ozaboost
Confuse_ozaboost$table
tp<-Confuse_ozaboost$table[2,2] #Positive="1"
fn<-Confuse_ozaboost$table[1,2]
tn<-Confuse_ozaboost$table[1,1]
fp<-Confuse_ozaboost$table[2,1]
preci<-tp/(tp+fp)
recal<-tp/(tp+fn)
fscore<-2*((preci*recal)/(preci+recal))
fscore

