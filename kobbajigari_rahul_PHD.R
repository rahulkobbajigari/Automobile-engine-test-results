  
  #clear environment
  
  rm(list=ls())
  
  #load libraries
  
  
  
  #devtools::install_github("rstudio/keras")
  
  library(keras)
  #install_keras()
  
  library(dplyr)
  library(MLmetrics)
  library(lubridate)
  library(dummies)
  library(missForest)
  library(xgboost)
  library(data.table)
  library(randomForest)
  library(DMwR)
  library(nnet)
  library(caret)
  library(keras)
  library(tidyr)
  library(ROSE)
  library(smotefamily)
  library(rpart.plot)
  library(rpart)
  library(RColorBrewer)
  library(ROCR)
  library(C50)
  library(CHAID)
  library(woe)
  library(Information)
  library(InformationValue)
  library(gridExtra)
  library(devtools)
  library(e1071)
  library(MASS)
  library(car)
  library(psych)
  library(corrplot)
  library(zoo)
  library(h2o)
  library(sqldf)
  library(ggplot2)
  
  
  #Reading the data in R 
  
train<-read.csv("Train.csv")
test<-read.csv("Test.csv")
test_additional<-read.csv("Test_AdditionalData.csv")
engine_additionalData <- read.csv("Train_AdditionalData.csv", header = T)


#basic data understanding
#train data consists of (3156)attributes and (22)columns
dim(train)
#3156 X 22

#train_additional data This data is related to other similar but not identical test performed on the same configuration. Data is provided for two tests (TestA and TestB). Values under TestA and TestB corresponds to Engine ID in your primary data. 
#If a particular Engine ID is present under TestA, it has passed that particular test

dim(engine_additionalData)
#2026 X 2

#dimension of test  and test additonal data

dim(test)
#1053 X 21

dim(test_additional)
#686 X 2


## checking missing values in data 

sum(is.na(train))
#3160

#train additional conists of 39 missing points 
sum(is.na(engine_additionalData))
#39

#omitting the data present in the train_additional because it is unique id's
engine_additionalData<-na.omit(engine_additionalData)

#test data consists of 1060 missing points
sum(is.na(test))

#test_additional data consists of 13 missing values
sum(is.na(test_additional))

#omitting the data present in the test_additional because it is unique id's
test_additional<-na.omit(test_additional)


#train additional  data consists of two columns with respect to id 
#so i seperated two columns 
engine_additionalData1<-as.data.frame(engine_additionalData$TestA)

#changing the name of a column
names(engine_additionalData1)[1]<-paste("TestA")

engine_additionalData2<-as.data.frame(engine_additionalData$TestB)

##changing the name of a column
names(engine_additionalData2)[1]<-paste("TestB")

#added two columns in each data with (1) value
engine_additionalData1$testapassed <- 1
engine_additionalData2$testbpassed <- 1

#using the left_join to merge the data with train with respect to id's and testa,testb
train_join<-train%>%left_join(engine_additionalData1,c("ID"="TestA"))

train_join<-train_join%>%left_join(engine_additionalData2,c("ID"="TestB"))

#while merging the data if the values are not present in additional data then it converts into na's
#the missing values present in the testapassed and testbpassed are assumed as 0
#value(1)<-pass and value(0)<-fail in the test
train_join["testapassed"][is.na(train_join["testapassed"])] <- 0
train_join["testbpassed"][is.na(train_join["testbpassed"])] <- 0

#changing the column names 
names(train_join)[23]<-paste("TestA")
names(train_join)[24]<-paste("TestB")

#converting the testA and testB in to factors
train_join$TestA <- as.factor(train_join$TestA)
train_join$TestB <- as.factor(train_join$TestB)




#checking the missing values in each column after merging the data

#20 rows consists of 158 missing values
colSums(is.na(train_join))

#checking the unique values in train data
apply(train,function(x){length(unique(x))})

#checking the missing values in each row highest is 3 in a row
rowSums(is.na(train_join))

#Percentage of rows having missing values(64.3)
sum(!complete.cases(train_join))/nrow(train_join)




#converting target level to 1 and 0
#train_join$y <- ifelse(train_join$y == "pass", 1, 0)




train_join$Number.of.Cylinders<-as.factor(train_join$Number.of.Cylinders)


####
#handling the na's(KNN imputation)
train_imputation<-knnImputation(train_join,k=3)

#removing the ID column
train_imputation$ID<-NULL


####
#testdata

#
#train additional data consists of two columns with respect to id 
#so i seperated two columns 
test_additionalData1<-as.data.frame(test_additional$TestA)

#changing the column name
names(test_additionalData1)[1]<-paste("TestA")

test_additionalData2<-as.data.frame(test_additional$TestB)

#changing the column name
names(test_additionalData2)[1]<-paste("TestB")


#added two columns in each data with (1) value
test_additionalData1$testapassed <- 1
test_additionalData2$testbpassed <- 1


#using the left_join to merge the data with test with respect to id's and testa,testb

test_join<-test%>%left_join(test_additionalData1,c("ID"="TestA"))

test_join<-test_join%>%left_join(test_additionalData2,c("ID"="TestB"))


#while merging the data if the values are not present in additional data then it converts into na's
#the missing values present in the testapassed and testbpassed are assumed as 0
#value(1)<-pass and value(0)<-fail in the test
test_join["testapassed"][is.na(test_join["testapassed"])] <- 0
test_join["testbpassed"][is.na(test_join["testbpassed"])] <- 0
head(test_join)
str(test_join)

#changing the column name
names(test_join)[22]<-paste("TestA")
names(test_join)[23]<-paste("TestB")

#converting testa and testb in to factors
test_join$TestA <- as.factor(test_join$TestA)
test_join$TestB <- as.factor(test_join$TestB)

#converting number of cylinder into factor
test_join$Number.of.Cylinders<-as.factor(test_join$Number.of.Cylinders)

#checking the missing value in each column
#53 values in 20 columns 
colSums(is.na(test_join))



#Percentage of rows having missing values(0.63)
sum(!complete.cases(test_join))/nrow(test_join)

str(train_join)


####
#handling the na's
test_imputation<-knnImputation(test_join,k=3)
test_imputation$ID<-NULL


Train<-train_imputation
Train1<-test_imputation



#chi_square(Number.of.Cylinders,Lubrication,Cylinder.arragement,Turbocharger,Varaible.Valve.Timing..VVT.,main.bearing.type,displacement,Crankshaft.Design are not dependent to y)

allnames= names(Train)
for(i in 1:ncol(Train)){
  for(j in 1:ncol(Train)){
    chisq= chisq.test(Train[,i],Train[,j], correct=FALSE)
    if(chisq$p.value>0.05){
      cat(allnames[i],"and",allnames[j],"are not dependant on each other.","\n")
    }
    
    
  }
}


##feature engineering
#if the testa and testb are same then it is kept as one or zero

train_imputation$TESTAB4<-ifelse(Train$TestA==1&Train$TestB==1,'1-1',
                      ifelse(Train$TestA==0&Train$TestB==1,'0-1',
                             ifelse(Train$TestA==1&Train$TestB==0,'1-0','0-0')))

test_imputation$TESTAB4<-ifelse(Train1$TestA==1&Train1$TestB==1,'1-1',
                                 ifelse(Train1$TestA==0&Train1$TestB==1,'0-1',
                                        ifelse(Train1$TestA==1&Train1$TestB==0,'1-0','0-0')))



#converting in to factor
train_imputation$TESTAB4<-as.factor(train_imputation$TESTAB4)
test_imputation$TESTAB4<-as.factor(test_imputation$TESTAB4)

#checking the levels in each column
sapply(train_imputation,levels)
sapply(test_imputation,levels)



# Data Exploration, Variable Understanding and Data PreProcessing 

#1 target variable(y)
#categorical

#visualization and insights
 
#when
ggplot(train_imputation, aes(y, group = material.grade)) + 
  geom_bar(aes(y = ..prop.., fill = factor(material.grade)), stat="count") + 
  scale_y_continuous(labels=scales::percent) + 
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  ylab("relative frequencies") +
  facet_grid(~material.grade) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="material.grade across y") 


ggplot(train_imputation, aes(y, group = Number.of.Cylinders)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Number.of.Cylinders)), stat="count") + 
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  scale_y_continuous(labels=scales::percent) +
  ylab("relative frequencies") +
  facet_grid(~Number.of.Cylinders) +   
  labs(title="Barchart on Categorical Variable", 
       subtitle="Number.of.Cylinders across y") 


ggplot(train_imputation, aes(y, group = Lubrication)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Lubrication)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
    ylab("relative frequencies") +
  facet_grid(~Lubrication) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="Lubrication across y") 

ggplot(train_imputation, aes(y, group = Valve.Type)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Valve.Type)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
    ylab("relative frequencies") +
  facet_grid(~Valve.Type) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="Valve.Type across y") 


ggplot(train_imputation, aes(y, group = Bearing.Vendor)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Bearing.Vendor)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
    ylab("relative frequencies") +
  facet_grid(~Bearing.Vendor) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="Bearing.Vendor across y") 


ggplot(train_imputation, aes(y, group = Fuel.Type)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Fuel.Type)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  ylab("relative frequencies") +
  facet_grid(~Fuel.Type) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="Fuel.Type across y") 

ggplot(train_imputation, aes(y, group = Compression.ratio)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
    ylab("relative frequencies") +
  facet_grid(~Compression.ratio) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="Turbocharger across y") 
## no turbocharger then fail

ggplot(train, aes(y, group = cam.arrangement)) + 
  geom_bar(aes(y = ..prop.., fill = factor(cam.arrangement)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
    ylab("relative frequencies") +
  facet_grid(~cam.arrangement) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="cam.arrangement across y") 
## no turbocharger then fail

ggplot(train_imputation, aes(y, group = Cylinder.arragement)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
    ylab("relative frequencies") +
  facet_grid(~Cylinder.arragement) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title= "Barchart on Categorical Variable", 
       subtitle="cylinder.arrangement across y") 
## no turbocharger then fail

ggplot(train_imputation, aes(y, group = Turbocharger)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
    ylab("relative frequencies") +
  facet_grid(~Turbocharger) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="Turbocharger across y") 
## no turbocharger then fail

ggplot(train_imputation, aes(y, group = Varaible.Valve.Timing..VVT.)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
    ylab("relative frequencies") +
  facet_grid(~Varaible.Valve.Timing..VVT.)+
 labs(title="Barchart on Categorical Variable", 
       subtitle="Varaible.Valve.Timing..VVT. across y") 
#3Varaible.Valve.Timing..VVT. -No then pass

ggplot(train_imputation, aes(y, group = Cylinder.deactivation)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Cylinder.deactivation)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  ylab("relative frequencies") +
  facet_grid(~Cylinder.deactivation)+
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  labs(title="Barchart on Categorical Variable", 
       subtitle="cylinder.deactivation across y") 
#3Varaible.Valve.Timing..VVT. -No then pass

table(train$Cylinder.deactivation)

ggplot(train_imputation, aes(y, group = Direct.injection)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  facet_grid(~Direct.injection)+
    ylab("relative frequencies") +  labs(title="Histogram on Categorical Variable", 
       subtitle="Direct.injection across y") 

ggplot(train_imputation, aes(y, group = piston.type)) + 
  geom_bar(aes(y = ..prop.., fill = factor(piston.type)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  facet_grid(~piston.type)+
    ylab("relative frequencies") +
  labs(title="Barchart on Categorical Variable", 
       subtitle="piston.type across y") 



ggplot(train_imputation, aes(y, group = Peak.Power)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Peak.Power)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  facet_grid(~Peak.Power)+
  ylab("relative frequencies") +
  labs(title="Barchart on Categorical Variable", 
       subtitle="Peak.Power across y") 










ggplot(train_imputation, aes(y, group = TESTAB4)) + 
  geom_bar(aes(y = ..prop.., fill = factor(TESTAB4)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  ylab("relative frequencies") +
  facet_grid(~TESTAB4)+
  labs(title="Barchart on Categorical Variable", 
       subtitle="TestAB across y") 


ggplot(train_imputation, aes(y, group =TestA)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  ylab("relative frequencies") +
  facet_grid(~TestA)+
  labs(title="Barchart on Categorical Variable", 
       subtitle="TestA across y") 


ggplot(train_imputation, aes(y, group =TestB)) + 
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  ylab("relative frequencies") +
  facet_grid(~TestB)+
  labs(title="Barchart on Categorical Variable", 
       subtitle="TestB across y") 


#independent variable across TESTAB



ggplot(train_imputation, aes(TESTAB4, group = Peak.Power)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Peak.Power)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  facet_grid(~Peak.Power)+
  ylab("relative frequencies") +
  labs(title="Barchart on Categorical Variable", 
       subtitle="Peak.Power across testAB") 





ggplot(train_imputation, aes(TESTAB4, group = Cylinder.deactivation)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Cylinder.deactivation)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  ylab("relative frequencies") +
  facet_grid(~Cylinder.deactivation)+
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  labs(title="Barchart on Categorical Variable", 
       subtitle="cylinder.deactivation across TESTAB")





ggplot(train_imputation, aes(TESTAB4, group = Bearing.Vendor)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Bearing.Vendor)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  ylab("relative frequencies") +
  facet_grid(~Bearing.Vendor) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="Bearing.Vendor across testAB")




ggplot(train_imputation, aes(TESTAB4, group = Fuel.Type)) + 
  geom_bar(aes(y = ..prop.., fill = factor(Fuel.Type)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  ylab("relative frequencies") +
  facet_grid(~Fuel.Type) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="Fuel.Type across TESTAB") 




ggplot(train_imputation, aes(TESTAB4, group = piston.type)) + 
  geom_bar(aes(y = ..prop.., fill = factor(piston.type)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  facet_grid(~piston.type)+
  ylab("relative frequencies") +
  labs(title="Barchart on Categorical Variable", 
       subtitle="piston.type across TESTAB4") 





ggplot(train, aes(train_imputation$TESTAB4, group = cam.arrangement)) + 
  geom_bar(aes(y = ..prop.., fill = factor(cam.arrangement)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  geom_text(aes( label = scales::percent(..prop..),y= ..prop.. ), stat= "count", vjust = -.5) +
  ylab("relative frequencies") +
  facet_grid(~cam.arrangement) +  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Barchart on Categorical Variable", 
       subtitle="cam.arrangement across y")



#converting target level to 1 and 0
train_imputation$y <- ifelse(train_imputation$y == "pass", 1, 0)

train_imputation$y<-as.factor(train_imputation$y)





###logistic regression
glm.model<-glm(train_imputation$y ~  Lubrication + Bearing.Vendor + Fuel.Type + 
                 Cylinder.deactivation + displacement + piston.type + Max..Torque + 
                 Peak.Power +TESTAB4, family = "binomial", 
               data = train_imputation)

#summary for logistic regression
summary(glm.model)

#running the stepAIC

stepAIC(glm.model)


#predict the values on train
prob_train<-predict(glm.model,type = "response")
p<-predict(glm.model,train_imputation,type = "response")

ifelse(p>0.5,"1","0")

#table for train
tab<-table(p>0.5,train_imputation$y)
tab

#accuracy
acc<-sum(diag(tab))/sum(tab)
acc

#area under the curve
pred <- prediction(prob_train,train_imputation$y)
perf <- performance(pred, measure="tpr", x.measure="fpr")


# Plot the ROC curve 
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))

# Use the performance function to get the AUC score
perf_auc <- performance(pred, measure="auc")
auc <- perf_auc@y.values[[1]]
print(auc)

# For different cutoffs identifying the tpr and fpr
cutoffs <- data.frame(cut= perf@alpha.values[[1]], fpr= perf@x.values[[1]], 
                      tpr=perf@y.values[[1]])

# Sorting the data frame in the decreasing order based on tpr
cutoffs <- cutoffs[order(cutoffs$tpr, decreasing=TRUE),]




#predicting on test
testp<-predict(glm.model,test_imputation,type = "response")
ppp<-ifelse(testp>0.5,"pass","fail")
table(ppp)


y<-as.data.frame(ppp)
names(y)[1]<-paste("y")


Submission<-cbind("ID"=test$ID,y)
write.csv(Submission,file = "Submission.csv",row.names = F)


#partitioning the datain to train and validation
set.seed(785)
rows<-createDataPartition(train_imputation$y,times=1,p=0.8,list = F)
train_part<-train_imputation[rows,]
validation_part<-train_imputation[-rows,]

summary(train_imputation)


train_part$TestA<-NULL
train_part$TestB<-NULL



#h20
library(h2o)
h2o.init(nthreads = -1, max_mem_size = "4g")

train.hex <- as.h2o(x = train_part, destination_frame = "train.hex")

test.hex <- as.h2o(x = test_imputation, destination_frame = "test.hex")

validation.hex <- as.h2o(x = validation_part , destination_frame = "validation.hex")
# Prepare the parameters for the for H2O gbm grid search

ntrees_opt <- c(30,100 ,200)

maxdepth_opt <- c(2,5,10)


hyper_parameters <- list(ntrees = ntrees_opt,
                         
                         max_depth = maxdepth_opt

                         )
my_rf <- h2o.randomForest(x = setdiff(names(train.hex), "y"),y =  "y",
                          
                          training_frame = train.hex,
                          
                          nfolds = 15,fold_assignment = "Modulo",
                          
                          keep_cross_validation_predictions = TRUE,seed = 1)


my_rf1 <- h2o.grid("randomForest", x = setdiff(names(train.hex), "y"), y = "y", training_frame = train.hex ,
                   
                   hyper_params = list(ntrees = c(100,300,500), mtries = c(2,5,10), max_depth = c(3,4,5)), seed = 1122)
#h2o.varimp(my_rf)

h2o.varimp_plot(my_rf)


predict_rf_train.hex = h2o.predict(my_rf, newdata = train.hex[,setdiff(names(train.hex), "y")])

predict_rf.hex = h2o.predict(my_rf, newdata = test.hex[,setdiff(names(test.hex), "y")])

predict_rf_validattion.hex = h2o.predict(my_rf, newdata = validation.hex[,setdiff(names(validation.hex), "y")])

h2o.shutdown()
y

table(train$cam.arrangement,train$y)

#removing the insignificant columns

train_data<-train_imputation
train_data$Number.of.Cylinders<-NULL
train_data$Compression.ratio<-NULL
train_data$cam.arrangement<-NULL
train_data$Cylinder.arragement<-NULL
train_data$Varaible.Valve.Timing..VVT.<-NULL
train_data$Direct.injection<-NULL
train_data$Turbocharger<-NULL
train_data$main.bearing.type<-NULL
train_data$displacement<-NULL
train_data$piston.type<-NULL
train_data$Max..Torque<-NULL
train_data$Crankshaft.Design<-NULL
train_data$Liner.Design.<-NULL
#test_data<-test_imputation["testAB","Peak.Power","Cylinder.deactivation","Bearing.Vendor","material.grade","Lubrication","Valve.Type","testB","Fuel.Type","TestA"]



#dummifing the categorical variables
dummies<-dummyVars(y~.,data = train_data)
xtrain_data1<-data.frame(predict(dummies,train_data))
ytrain_data1<-train_imputation$y
ytrain_data1<-as.factor(ytrain_data1)
#xtrain_data1<-subset(xtrain_data1,ytrain_data1)
test_data<-test_imputation
test_data$y<-0


#dummifing the testdata
xtest_data1<-data.frame(predict(dummies,test_data))


ind_Attr = setdiff(names(xtrain_data1),"y")

dtrain = xgb.DMatrix(data = as.matrix(xtrain_data1[,ind_Attr]),label = xtrain_data1$y )



####xgboost####extra gradient boosting with various parameter tuning
xgb.ctrl <- trainControl(method = "repeatedcv", number = 10,
                         search='random')

####xgboost####extra gradient boosting with tune length 10
#set.seed(123)
xgb.tune <-train(
                 x = xtrain_data1,y=ytrain_data1,method="xgbTree",metric = "Accuracy",
                 trControl=xgb.ctrl,
                 tuneLength=11)

preds_xgb_train <- predict(xgb.tune,xtrain_data1)

confusionMatrix(preds_xgb_train,ytrain_data1)

preds_xgb_test <- data.frame(predict(xgb.tune, xtest_data1))

preds_xgb_test<-ifelse(preds_xgb_test$predict.xgb.tune..xtest_data1.==1,"pass","fail")


preds_xgb_test1<-as.data.frame(preds_xgb_test)
names(preds_xgb_test1)[1]<-paste("y")



Submission1<-cbind("ID"=test$ID,preds_xgb_test1)
write.csv(Submission1,file = "Submission1.csv",row.names = F)


##random forest
train_data$y<-as.factor(train_data$y)
train_part$main.bearing.type<-NULL
#BUILDING A RANDOM FOREST. FIRST, FIND THE OPTIMAL 'mtry' VALUE
tuneRF(train_data[,-1], train_data[,1], ntreeTry = 150)
# mtry = '4'

#USING 'randomForest' FUNCTION FROM THE RANDOM FOREST PACKAGE
model3 = randomForest(y~., data = train_data, ntree = 150, mtry = 2)
plot(model3)
varImpPlot(model3)


#PREDICTING ON VALIDATION


sed = predict(model3, validation_part)
table(sed)

df = confusionMatrix(sed, validation_part$y)
df

train_imputation1<-train_imputation
train_imputation1$y<-NULL
test_imputation1<-test_imputation
  

#checking if the extra levels present in test and then convert in to na's
for (i in 1:23) {
  test_imputation1[,i]<-factor(test_imputation1[,i],levels = levels(train_imputation1[,i]))
  
}

sum(is.na(test_imputation1))

sed1 = predict(model3, test_data)  


preds_rf_test <- data.frame(predict(model3, test_data))

preds_xgb_test<-ifelse(preds_rf_test$predict.model3..test_data.==1,"pass","fail")

preds_rf_test1<-as.data.frame(preds_rf_test)

names(preds_rf_test1)[1]<-paste("y")

Submission2<-cbind("ID"=test$ID,preds_rf_test1)
write.csv(Submission1,file = "Submission2.csv",row.names = F)



## 1.
## CART -> uses GINI Index -> Binary split
library(rpart)
### grow tree.
rpart.model = rpart(train_part$y ~ ., data = train_part, method = "class")
summary(rpart.model)
print(rpart.model)
predict.train.rpart = predict(rpart.model, newdata = train_part, type = "vector")
predict.train.rpart.classes = ifelse(predict.train.rpart > 1, "1", "0")
unique(predict.train.rpart)
predict.validation.rpart = predict(rpart.model, newdata = validation_part, type = "vector")
predict.rpart.validation.classes = ifelse(predict.validation.rpart > 1, "1", "0")

confusionMatrix(predict.train.rpart.classes, train_part$y, positive = "1")
confusionMatrix(predict.rpart.validation.classes, validation_part$y, positive = "1")

predict.test.rpart = predict(rpart.model, newdata = test_data, type = "vector")

predict.rpart.test.classes = ifelse(predict.test.rpart > 1, "1", "0")

preds_xgb_test<-ifelse(preds_rf_test$predict.model3..test_data.==1,"pass","fail")
preds_rf_test1<-as.data.frame(preds_rf_test)
names(preds_rf_test1)[1]<-paste("y")

Submission2<-cbind("ID"=test$ID,preds_rf_test1)
write.csv(Submission1,file = "Submission2.csv",row.names = F)




#cross validation
##################################
Train<-train_imputation

control <- trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(7)
 fit.glm <- caret::train(y~., data=Train, method="glm", trControl=control)
set.seed(7)
fit.cart <- caret::train(y~., data=Train, method="rpart", trControl=control)
set.seed(7)
fit.rf <- caret::train(y~., data=Train, method="rf", trControl=control)
set.seed(7)
fit.gbm1 <- caret::train(y~., data=Train, method="gbm", trControl=control,verbose = FALSE)
set.seed(7)
fit.xgb <- caret::train(y~., data=Train, method="xgbTree", trControl=control,verbose = FALSE)
set.seed(7)
fit.svm <- caret::train(y~., data=Train, method="svmRadial", trControl=control,verbose = FALSE)
set.seed(7)
fit.naive_bayes <- caret::train(y~., data=Train, method="naive_bayes", trControl=control,verbose = FALSE)
names(getModelInfo())
results <- resamples(list(GLM=fit.glm,CART=fit.cart, GBM=fit.gbm1, RF=fit.rf, XGB=fit.xgb,svm=fit.svm))
summary(results)
results<-as.data.frame(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)




########### cross validation #############
trainctrl = trainControl(method = "repeatedcv", repeats = 3, number = 3,
                         search='random',
                         allowParallel=T)

svm_Radial = train(y ~ .,data = Train,trControl = trainctrl,method = "svmRadial",metric = "Accuracy",tuneLength=30)

svm_Radial
preds_train=predict(svm_Radial,Train)
preds=predict(svm_Radial,test_imputation)

confusionMatrix((Train$y),preds_train)



y_pred_final=as.data.frame(predict(svm_Radial,newdata = (test_imputation)))
names(y_pred_final)[1]<-paste("y")

#y_pred_final<-ifelse(y_pred_final$y==1,"pass","fail")




Submission2<-cbind("ID"=test$ID,y_pred_final)
#names(Submission2)[1]<-paste("y")
write.csv(Submission2,file = "submission_SVM.csv",row.names = F)



#stacking
### probabilities ###

pred_rf_prob<-predict(glm.model,train_imputation,type = "prob")

pred_glm_prob<-predict(model_glm,dft1[,predictors],type = "prob")

pred_knn_prob<-predict(xgb.tune,test_imputation,type = "prob")


weighted_avg<-(y$y*0.6)+(preds_xgb_test1$y*0.05)+(y_pred_final$y*0.35)

weighted_avg<-as.factor(ifelse(weighted_avg>0.4,"Yes","No"))

weighted_avg<-as.data.frame(weighted_avg)



t_w<-table(weighted_avg$weighted_avg,val.s$Churn)

acc_w<-sum(diag(t_w))/sum(t_w)

acc_w



Recall(weighted_avg$weighted_avg,val.s$Churn,positive = "Yes")

table(weighted_avg$weighted_avg)



