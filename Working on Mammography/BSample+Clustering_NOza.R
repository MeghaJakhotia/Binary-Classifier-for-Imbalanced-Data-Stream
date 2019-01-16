setwd("E:/Megha/X")
#Reading the input data set
Inp <- read.csv("Input.txt",col.names = c("V1","V2", "V3", "V4", "V5", "V6","V7"), sep=",")
library('caret')
library('e1071')
#We need to set the seed so that the index values generated for partition are constant on every run
set.seed(0)

#Dividing the input data sets into 2 parts (Train and Test) and storing it in separate files 
index <- createDataPartition(Inp$V7, p=0.5, list=FALSE)

write.table(Inp[index,],"TrainF.txt",sep=",",row.names=FALSE,col.names = TRUE)
write.table(Inp[-index,],"TestF.txt",sep=",",row.names=FALSE,col.names = TRUE)

#write.table(Inp[index,c(1:6)],"Train.txt",sep=",",row.names=FALSE,col.names = TRUE)
#write.table(Inp[index,7],"TrainC.txt",sep=",",row.names=FALSE,col.names = TRUE)
#write.table(Inp[-index,c(1:6)],"Test.txt",sep=",",row.names=FALSE,col.names = TRUE)
#write.table(Inp[-index,7],"TestC.txt",sep=",",row.names=FALSE,col.names = TRUE)

#--- CLUSTERING ---
library("stream")
library("streamMOA")
library("DMwR")
#Read the data into a data stream using the "stream" package
train_stream <- DSD_ReadCSV("TrainF.txt",k=2,take=c(1:7),sep = ",",header=TRUE)
points1 <- get_points(train_stream,5592)
points1$V7 <- as.factor(points1$V7)
newData<- SMOTE(V7~.,points1,perc.over = 2000,perc.under = 100 )
newData$V7<- as.numeric(as.character(newData$V7))
write.csv(newData,file="SampledData.txt",row.names = FALSE)
write.csv(newData[,c(1:6)],file="SampleWO.txt",row.names = FALSE)
write.csv(newData[,c(7)],file="SampleClass.txt",row.names = FALSE)

train_strm <- DSD_ReadCSV("SampleWO.txt",k=2,take=c(1:6),sep = ",",header=TRUE)
#The "train_stream" does not contain any stored class labels

points2 <- get_points(train_strm,5000)
#Clustream + K-means clustering algorithm using the DSC_TwoStage of "streamMOA" package

Windo <- DSC_TwoStage(micro=DSC_Window(100),macro=DSC_Kmeans(k=2))
#clustream1<-DSC_CluStream(m=500,k=2)
reset_stream(train_strm)
update(Windo,train_strm,5000)


#Storing the clusters assigned to the input data points to a
a<- get_assignment(Windo, points2)
a <- as.data.frame(a)
#The class labels are received with a delay and read in as a data stream
classlabel_stream<- DSD_ReadCSV("SampleClass.txt",k=2,sep = ",",header=TRUE)

#Cluster plot
plot(Windo, assignment= TRUE, type="both")

#Finding the true class labels of the clusters
x = c(0,0,0,0)
for (i in 1:5000) {
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
for(i in c(1:5000)){
  if(a[i,1]==1)
  { a[i,1]<-rep[1]
  } else {
    if (a[i,1]==2)
    {
      a[i,1]<-rep[2]
    }}
}

#Storing the input to be given to the classifier
reset_stream(train_strm)
write_stream(train_strm, file="InputforClassifier.txt", n=5000,class=FALSE, append = FALSE, sep=",", header=TRUE, row.names=FALSE)

library(RMOA)
#--- BUILDING A CLASSIFIER MODEL ---
#Reading the data as a data frame
sea_data <- read.csv("InputforClassifier.txt", sep= ",",col.names = c("V1","V2","V3","V4","V5","V6"))

#Converting the data of the assigned classes by the Clustering Algorithm as factors
a$a<-as.factor(a$a)

#Combining the data points and corresponding assigned class labels
sea_data<-data.frame(sea_data,a)
#sea_data$V7 <-as.factor(sea_data$V7)
#Converted to data streams
sea_datastream <- datastream_dataframe(data=sea_data)
#Forming the train set to be given as input to the classifier model
trainset <- sea_datastream$get_points(sea_datastream, n = 5592, outofpoints = c("stop", "warn", "ignore"))
trainset <- datastream_dataframe(data=trainset)
a<-datastream_dataframe(data=a)

#Classifier Model
ctrl<-MOAoptions(model="NaiveBayes")
mymodel<-NaiveBayes(control=ctrl)

naive_model <- trainMOA(model = mymodel,a ~ ., data = trainset)
naive_model

test_data <- read.csv("TestF.txt", sep= ",",col.names = c("V1","V2","V3","V4","V5","V6","V7"))
test_datastream <- datastream_dataframe(data=test_data)
points2 <- test_data
#points2 <- get_points(test_datastream,5591)
points2$V7 <- as.factor(points2$V7)
newData<- SMOTE(V7~.,points2,perc.over = 2000,perc.under = 100 )
newData$V7<- as.numeric(as.character(newData$V7))
write.csv(newData,file="SampledDataT.txt",row.names = FALSE)
write.csv(newData[,c(1:6)],file="SampleWOT.txt",row.names = FALSE)
write.csv(newData[,c(7)],file="SampleClassT.txt",row.names = FALSE)
#Test set and predictions
#reset_stream(train_stream)
#get_points(train_stream,5000)
#write_stream(train_stream, file="TestInput.txt", n=5000,class=TRUE, append = FALSE, sep=",", header=TRUE)

sea_test_data <- read.csv("SampledDataT.txt", sep= ",",col.names = c("V1","V2","V3","V4","V5","V6","V7"))
test_datastream <- datastream_dataframe(data=sea_test_data)

testset <- test_datastream$get_points(test_datastream, n = 4797, outofpoints = c("stop", "warn", "ignore"))
scores <- predict(naive_model, newdata=testset[, c("V1","V2","V3","V4","V5","V6")], type="response")
str(scores)
table(scores, testset$V7)
#x<- read.csv("TestC.txt",sep=",")

Confuse<-confusionMatrix(data=scores,testset[,7], positive="1")
Confuse
tp<-Confuse$table[2,2] #Positive="1"
fn<-Confuse$table[1,2]
tn<-Confuse$table[1,1]
fp<-Confuse$table[2,1]
preci<-tp/(tp+fp)
recal<-tp/(tp+fn)
fscore<-2*((preci*recal)/(preci+recal))

fscore
#Ensemble model - OzaBoost
sea_test_datastream$reset()
trainset$reset()
#a$reset()
emodel1<-OzaBoost(baseLearner="trees.HoeffdingTree",ensembleSize=30)
ozaboost_model <- trainMOA(model = emodel1,a ~ ., data = trainset)
ozaboost_model
testset2 <- sea_test_datastream$get_points(sea_test_datastream, n = 5592, outofpoints = c("stop", "warn", "ignore"))
scores2 <- predict(ozaboost_model, newdata=testset2[, c("V1","V2","V3","V4","V5","V6")], type="response")
str(scores2)

Confuse_ozaboost<-confusionMatrix(data=as.numeric(scores2>0.5),testset2[,7], positive="1")
Confuse_ozaboost

Confuse_ozaboost$table
tp<-Confuse_ozaboost$table[2,2] #Positive="1"
fp<-Confuse_ozaboost$table[1,2]
tn<-Confuse_ozaboost$table[1,1]
fn<-Confuse_ozaboost$table[2,1]
preci<-tp/(tp+fp)
recal<-tp/(tp+fn)
fscore<-2*((preci*recal)/(preci+recal))
fscore

