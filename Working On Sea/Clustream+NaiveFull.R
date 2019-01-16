setwd("E:/Megha/BE project")
dt <- read.csv("sea.txt",col.names = c("V1","v2", "V3", "class"))
library('caret')
library('e1071')
set.seed(0)
index <- createDataPartition(dt$V1, p=0.5, list=FALSE)
write.table(dt[index,],"Train.txt",sep=",",row.names=FALSE,col.names = TRUE)
write.table(dt[-index,],"Test.txt",sep=",",row.names=FALSE,col.names = TRUE)
#dt1 <- read.csv("Train.txt",col.names = c("V1","v2", "V3", "class"))
write.table(dt[index,4],"ClassLabels.txt",sep=",",row.names = FALSE,col.names="class")

library("stream")
library("streamMOA")
train_stream <- DSD_ReadCSV("Train.txt",k=2,take=c(1:4),class=4,sep = ",",header=TRUE)
clustream1<-DSC_CluStream(m=100,k=2)
update(clustream1,train_stream,15000)
reset_stream(train_stream)
points1 <- get_points(train_stream,15000)

a<- get_assignment(clustream1, points1)
a <- as.data.frame(a)
classlabel_stream<- DSD_ReadCSV("ClassLabels.txt",k=2,sep = ",",header=TRUE)

plot(clustream1, train_stream, assignment= TRUE, type="both")

#reset_stream(train_stream)

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

reset_stream(train_stream)
write_stream(train_stream, file="InputforClassifier.txt", n=15000,class=FALSE, append = FALSE, sep=",", header=TRUE, row.names=FALSE)

library(RMOA)
sea_data <- read.csv("InputforClassifier.txt", sep= ",",col.names = c("V1","V2","V3"))
a$a<-as.factor(a$a)
sea_data<-data.frame(sea_data,a)
sea_datastream <- datastream_dataframe(data=sea_data)
trainset <- sea_datastream$get_points(sea_datastream, n = 5000, outofpoints = c("stop", "warn", "ignore"))
trainset <- datastream_dataframe(data=trainset)
a<-datastream_dataframe(data=a)
ctrl<-MOAoptions(model="NaiveBayes")
mymodel<-NaiveBayes(control=ctrl)
naive_model <- trainMOA(model = mymodel,a ~ ., data = trainset)
naive_model
testset <- sea_datastream$get_points(sea_datastream, n = 5000, outofpoints = c("stop", "warn", "ignore"))
scores <- predict(naive_model, newdata=testset[, c("V1","V2","V3")], type="response")
str(scores)
table(scores, testset$a)
Confuse <- confusionMatrix(scores,testset$a)
Confuse

