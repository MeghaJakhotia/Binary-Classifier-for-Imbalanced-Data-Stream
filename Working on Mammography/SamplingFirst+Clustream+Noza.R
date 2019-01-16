setwd("E:/Megha/X")

Inp <- read.csv("Input.txt",col.names = c("V1","V2", "V3", "V4", "V5", "V6","V7"), sep=",")
Inp$V7<-as.factor(Inp$V7)
library('caret')
library('e1071')
library('DMwR')

set.seed(0)

newData<- SMOTE(V7~.,Inp,perc.over = 2000,perc.under = 100 )
newData$V7<- as.numeric(as.character(newData$V7))
write.csv(newData,file="SampledData.txt",row.names = FALSE)
write.csv(newData[,c(1:6)],file="SampleWO",row.names = FALSE)
write.csv(newData[,c(7)],file="SampleClass",row.names = FALSE)
#table(newData$V7)

#Dividing the input data sets into 2 parts (Train and Test) and storing it in separate files 
index <- createDataPartition(newData$V7, p=0.6, list=FALSE)
write.table(newData[index,c(1:6)],"Train.txt",sep=",",row.names=FALSE,col.names = TRUE)
write.table(newData[index,7],"TrainC.txt",sep=",",row.names=FALSE,col.names = TRUE)

write.table(newData[-index,c(1:6)],"Test.txt",sep=",",row.names=FALSE,col.names = TRUE)
write.table(newData[-index,7],"TestC.txt",sep=",",row.names=FALSE,col.names = TRUE)

#Storing the class labels seprately as they would be available to the clustering algorithm after a delay
#write.table(dt[index,4],"ClassLabels.txt",sep=",",row.names = FALSE,col.names="class")

#--- CLUSTERING ---
library("stream")
library("streamMOA")
#Read the data into a data stream using the "stream" package
train_stream <- DSD_ReadCSV("Train.txt",k=2,take=c(1:6),sep = ",",header=TRUE)
#The "train_stream" does not contain any stored class labels

#Clustream + K-means clustering algorithm using the DSC_TwoStage of "streamMOA" package
clustream1<-DSC_CluStream(m=100,k=2)
update(clustream1,train_stream,6396)
reset_stream(train_stream)
points1 <- get_points(train_stream,6396)

a<- get_assignment(clustream1, points1)
a <- as.data.frame(a)
reset_stream(train_stream)
plot(clustream1, assignment= TRUE, type="both")

v <- read.csv("TrainC.txt", sep = ",")
#tapply(v$V1,v$V7,mean)
#apply(v,v$v7,mean)
c <- get_centers(clustream1)

classlabel_stream<- DSD_ReadCSV("TrainC.txt",k=2,sep = ",",header=TRUE)
reset_stream(train_stream)
x = c(0,0,0,0)
for (i in 1:6396) {
  pnt <- get_points(classlabel_stream,n=1)
  
  if(a[i,1]==1 & pnt==0 )
  { x[1] = x[1] + 1 }
  else {
    if (a[i,1]==1 & pnt==1)
    { x[2]= x[2] + 1}
    
    else
    { if (a[i,1]==2 & pnt==0)
    {  x[3]= x[3]+ 1 }
      else {
        if (a[i,1]==2 & pnt==1)
        { x[4]= x[4]+1 }
      }
    }
  }
}
rep =c(0,0)
if (x[1]<x[2])
{ rep[1]= 1
} else {
  rep[1]=0
}
if (x[3]<x[4])
{ rep[2]= 1
} else {
  rep[2]=0}
for(i in 1:6395){
  if(a[i,1]==1)
  { a[i,1]<-rep[1]
  } else {
    if (a[i,1]==2)
    {
      a[i,1]<-rep[2]
    }}
}

#Storing the input to be given to the classifier
#reset_stream(train_stream)
#write_stream(train_stream, file="InputforClassifier.txt", n=6395,class=TRUE, append = FALSE, sep=",", header=TRUE, row.names=FALSE)

#InpC <- data.frame(data)

library(RMOA)
#--- BUILDING A CLASSIFIER MODEL ---
#Reading the data as a data frame
sea_data <- read.csv("Train.txt", sep= ",")

#Converting the data of the assigned classes by the Clustering Algorithm as factors
a$a<-as.factor(a$a)

#Combining the data points and corresponding assigned class labels
sea_data<-data.frame(sea_data,a)

#Converted to data streams
sea_datastream <- datastream_dataframe(data=sea_data)
#Forming the train set to be given as input to the classifier model
trainset <- sea_datastream$get_points(sea_datastream, n = 6396, outofpoints = c("stop", "warn", "ignore"))
trainset <- datastream_dataframe(data=trainset)
a<-datastream_dataframe(data=a)

#Classifier Model
ctrl<-MOAoptions(model="NaiveBayes")
mymodel<-NaiveBayes(control=ctrl)
naive_model <- trainMOA(model = mymodel,a ~ ., data = trainset)
naive_model

#Test set and predictions
#reset_stream(train_stream)
#get_points(train_stream,15000)
#write_stream(train_stream, file="TestInput.txt", n=15000,class=TRUE, append = FALSE, sep=",", header=TRUE, row.names=FALSE)
sea_test_data <- read.csv("Test.txt", sep= ",")
sea_test_datastream <- datastream_dataframe(data=sea_test_data)
testset <- sea_test_datastream$get_points(sea_test_datastream, n =4264, outofpoints = c("stop", "warn", "ignore"))
scores <- predict(naive_model, newdata=testset[, c("V1","V2","V3","V4","V5","V6")], type="response")
str(scores)

test<-read.csv("TestC.txt",sep=",")
table(scores, test$x)

Confuse<-confusionMatrix(data=as.numeric(scores>0.5),test$x, positive="1")
Confuse$table
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
ozaboost_model <- trainMOA(model = emodel1,a ~ ., data = trainset)
ozaboost_model
testset2 <- sea_test_datastream$get_points(sea_test_datastream, n = 4264, outofpoints = c("stop", "warn", "ignore"))
scores2 <- predict(ozaboost_model, newdata=testset[, c("V1","V2","V3","V4","V5","V6")], type="response")
str(scores2)


Confuse_ozaboost<-confusionMatrix(data=as.numeric(scores2>0.5),test$x, positive="1")
Confuse_ozaboost
Confuse_ozaboost$table
tp<-Confuse_ozaboost$table[2,2] #Positive="1"
fp<-Confuse_ozaboost$table[1,2]
tn<-Confuse_ozaboost$table[1,1]
fn<-Confuse_ozaboost$table[2,1]
preci<-tp/(tp+fp)
recal<-tp/(tp+fn)
fscore2<-2*((preci*recal)/(preci+recal))
fscore
