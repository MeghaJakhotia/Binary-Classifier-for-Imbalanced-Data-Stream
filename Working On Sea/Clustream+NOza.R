setwd("E:/Megha/BE project")
#Reading the input data set
dt <- read.csv("sea.txt",col.names = c("V1","v2", "V3", "class"))
library('caret')
library('e1071')
#We need to set the seed so that the index values generated for partition are constant on every run
set.seed(0)

#Dividing the input data sets into 2 parts (Train and Test) and storing it in separate files 
index <- createDataPartition(dt$V1, p=0.5, list=FALSE)
write.table(dt[index,],"Train.txt",sep=",",row.names=FALSE,col.names = TRUE)
write.table(dt[-index,],"Test.txt",sep=",",row.names=FALSE,col.names = TRUE)

#Storing the class labels seprately as they would be available to the clustering algorithm after a delay
write.table(dt[index,4],"ClassLabels.txt",sep=",",row.names = FALSE,col.names="class")

#--- CLUSTERING ---
library("stream")
library("streamMOA")
#Read the data into a data stream using the "stream" package
train_stream <- DSD_ReadCSV("Train.txt",k=2,take=c(1:4),class=4,sep = ",",header=TRUE)
#The "train_stream" does not contain any stored class labels

#Clustream + K-means clustering algorithm using the DSC_TwoStage of "streamMOA" package
clustream1<-DSC_CluStream(m=100,k=2)
update(clustream1,train_stream,15000)
reset_stream(train_stream)
points1 <- get_points(train_stream,15000)

#Storing the clusters assigned to the input data points to a
a<- get_assignment(clustream1, points1)
a <- as.data.frame(a)
#The class labels are received with a delay and read in as a data stream
classlabel_stream<- DSD_ReadCSV("ClassLabels.txt",k=2,sep = ",",header=TRUE)

#Cluster plot
plot(clustream1, train_stream, assignment= TRUE, type="both")

#Finding the true class labels of the clusters
x = c(0,0,0,0)
for (i in 1:15000) {
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
for(i in c(1:15000)){
  if(a[i,1]==1)
  { a[i,1]<-rep[1]
  } else {
    if (a[i,1]==2)
    {
      a[i,1]<-rep[2]
    }}
}

#Storing the input to be given to the classifier
reset_stream(train_stream)
write_stream(train_stream, file="InputforClassifier.txt", n=15000,class=FALSE, append = FALSE, sep=",", header=TRUE, row.names=FALSE)

library(RMOA)
#--- BUILDING A CLASSIFIER MODEL ---
#Reading the data as a data frame
sea_data <- read.csv("InputforClassifier.txt", sep= ",",col.names = c("V1","V2","V3"))

#Converting the data of the assigned classes by the Clustering Algorithm as factors
a$a<-as.factor(a$a)

#Combining the data points and corresponding assigned class labels
sea_data<-data.frame(sea_data,a)

#Converted to data streams
sea_datastream <- datastream_dataframe(data=sea_data)
#Forming the train set to be given as input to the classifier model
trainset <- sea_datastream$get_points(sea_datastream, n = 5000, outofpoints = c("stop", "warn", "ignore"))
trainset <- datastream_dataframe(data=trainset)
a<-datastream_dataframe(data=a)

#Classifier Model
ctrl<-MOAoptions(model="NaiveBayes")
mymodel<-NaiveBayes(control=ctrl)
naive_model <- trainMOA(model = mymodel,a ~ ., data = trainset)
naive_model

#Test set and predictions
testset <- sea_datastream$get_points(sea_datastream, n = 5000, outofpoints = c("stop", "warn", "ignore"))
scores <- predict(naive_model, newdata=testset[, c("V1","V2","V3")], type="response")
str(scores)
table(scores, testset$a)

Confuse<-confusionMatrix(data=as.numeric(scores>0.5),testset[,4], positive="1")
#Evaluation metrics
Confuse$table
tp<-Confuse$table[2,2] #Positive="1"
fp<-Confuse$table[1,2]
tn<-Confuse$table[1,1]
fn<-Confuse$table[2,1]
preci<-tp/(tp+fp)
recal<-tp/(tp+fn)
fscore<-2*((preci*recal)/(preci+recal))

#Ensemble model - OzaBoost
sea_datastream$reset()
trainset$reset()
a$reset()
emodel1<-OzaBoost(baseLearner="trees.HoeffdingTree",ensembleSize=30)
ozaboost_model <- trainMOA(model = emodel1,a ~ ., data = trainset)
ozaboost_model
testset2 <- sea_datastream$get_points(sea_datastream, n = 5000, outofpoints = c("stop", "warn", "ignore"))
scores2 <- predict(ozaboost_model, newdata=testset2[, c("V1","V2","V3")], type="response")
str(scores2)
Confuse_ozaboost<-confusionMatrix(data=as.numeric(scores2>0.5),testset2[,4], positive="1")
Confuse_ozaboost

