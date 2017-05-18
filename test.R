# Author: Tarun, +91-8884534666, tarunadityab@gmail.com, skype: tarunaditya27
# Requirement is to build a 'cost sensitive algorithm' that minimizes False negatives or in other words have high recall rate 
# thus being accurate on predicting if someone is tuned in... The time to train & predict is 1 day thus removing the burden of capacity tunings in algo
# The model building is typically iterative process by adding new features, modifying model params etc, so simplicity sake limited features will be used
# however on actual data more feature engineering can be done

# data load
df<-read.csv('/home/tarun/Desktop/Test.csv',header = TRUE)

#Extract a few features
require(lubridate)
# creating 48 slots based on the timestamp
df$slot<-paste(hour(a),'_',ceiling(minute(a)/30))

#labeling it as primeslot
df$primeslot[hour(a)%in%c(8,9,10,20,21,22)] <- 1
df$primeslot[hour(a)%in%c(1:7,11:19,23)] <- 0

#clustering households based on channel preference, frequency of tunedins
#supplement data about the channel programs would give more info about the interests of household, eg households who particularly watch only news as eg
#assuming all the programs in these channels are classifed into 1)sports, 2)news, 3)movie

#aggregating household frequency behavior per program level ... this feature would help us understand a) not only the tune in probability but also 
#   b) type of programs they are tuned to... 

require(dplyr)
require(tidyr)
prog_aggdf<-df%>%group_by(housename,program)%>%summarize(freq=sum(tunedin))%>%spread(program,freq,fill=0)
head(aggdf)

#aggregating household's channel variety preference ... this feature would help us see if single channel loyal ones tune in most or multichannel 
#viewers are addicts & thus tune in most

channel_aggdf<-head(df%>%group_by(housename,channel)%>%summarize(freq=sum(tunedin))%>%spread(channel,freq,fill=0))

#merging both prog & channel behavior
merged<-merge(channel_aggdf,prog_aggdf,by='housename')

# clustering HH, note that this can be enchanced by adding geography details, demographic data etc
m<-kmeans(merged[,-1],centers = 3)
merged$cluster<-m$cluster

#using the household pref lables back to the tunein data & removing all other data
df1<-merge(df,merged,by='housename')
df1<-df1[,c(4,6:14)] 
 
#making sure that cluster,slot is treated as factor & not numeric
library(caret)
library(klaR)
df1$cluster<-as.factor(df1$cluster)
df1$slot<-as.factor(df1$slot)

split=0.80
trainIndex <- createDataPartition(df1$tunedin, p=split, list=FALSE)
data_train <- df1[ trainIndex,]
data_test <- df1[-trainIndex,]
# train a naive bayes model
model <- NaiveBayes(tunedin~., data=data_train)
# make predictions
x_test <- data_test[,2:10]
y_test <- data_test[,1]
predictions <- predict(model, data_test)
# summarize results
confusionMatrix(predictions$class, y_test)

# now for the real deal... building algorthim... first with neural nets & deep learning
require(nnet)
nnet_multinom<-multinom(tunedin~.,data_train)
# weights:  16 (15 variable)
#initial  value 9.704061 
#iter  10 value 0.087832
#iter  20 value 0.000210
#final  value 0.000052 
#converged
predictions <- predict(nnet_multinom, data_test)
#gave better confusion matrix
confusionMatrix(predictions, y_test)

# last benchmark: decision tree
library(rpart)
dtree<-rpart(tunedin~.,data=data_train)
#prediction
pred <- predict(dtree, newdata=data_test)
accuracy.meas(data_test$tunedin,pred)

#precision: 1.000
#recall: 0.200
#F: 0.167
roc.curve(data_test$tunedin, pred, plotit = F)
#0.56

#handling imbalances by modifying the loss function in classes to increase Recall / minimize FN
library(ROSE)
#over sampling
 data_balanced_over <- ovun.sample(tunedin ~ ., data = data_train, method = "over",N = 16)$data
#under sampling
 data_balanced_under <- ovun.sample(tunedin ~ ., data = data_train, method = "under",N = 6)$data
 
 table(data_balanced_over$tunedin)
 table(data_balanced_under$tunedin)
 data.rose <- ROSE(tunedin ~ ., data = data_train, seed = 1)$data
 table(data.rose$tunedin) # well balanced
 
 #build decision tree models
 tree.rose <- rpart(tunedin ~ ., data = data.rose)
 tree.over <- rpart(tunedin ~ ., data = data_balanced_over)
 tree.under <- rpart(tunedin ~ ., data = data_balanced_under)
 
 
 #make predictions on unseen data
 pred.tree.rose <- predict(tree.rose, newdata = data_test)
 pred.tree.over <- predict(tree.over, newdata = data_test)
 pred.tree.under <- predict(tree.under, newdata = data_test)
 
 #AUC ROSE
 roc.curve(data_test$tunedin, pred.tree.rose[,2])
 #Area under the curve (AUC): 0.989
 
 #AUC Oversampling
 roc.curve(data_test$tunedin, pred.tree.over[,2])
 #Area under the curve (AUC): 0.46
 
 #AUC Undersampling
 roc.curve(data_test$tunedin, pred.tree.under[,2])
 #Area under the curve (AUC): 0.867
 