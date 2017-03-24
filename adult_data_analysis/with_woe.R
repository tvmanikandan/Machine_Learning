rm(list=ls())
library(openxlsx)
library(caret)
library(e1071)
library(car)

set.seed(11)

input_df=read.xlsx("adult.xlsx",sheet = "input_data_with_woe")

str(input_df)

for (col in names(input_df)) { 
  if(col!="income") {
    input_df[, col]=scale(input_df[, col])
    print(col)
  }
}

# run logistic regression 80:20 split
part_index=createDataPartition(input_df$income,p=0.8,list=FALSE)
part_index

train_data=input_df[part_index,]
test_data=input_df[-part_index,]

x_train=subset(train_data,select=-c(income))
y_train=train_data$income

x_test=subset(test_data,select=-c(income))
y_test=test_data$income

rm(train_data)
rm(test_data)
gc()


model_lr_1=glm(factor(y_train) ~.,family=binomial(link='logit'),data=subset(x_train,
                                                      select=c(age,occupation,education,relationship,capital_gain,hours_per_week,workclass,capital_loss)),maxit = 500)
summary(model_lr_1)
vif(model_lr_1)
lr_1_pred=predict(model_lr_1,newdata = subset(x_test,select=c(age,occupation,education,relationship,capital_gain,hours_per_week,workclass,capital_loss)),type="response")
#prediction
confusionMatrix(y_test,ifelse(lr_1_pred>0.5,1,0)) # Accuracy = 0.8209



library(e1071)
model_svm_1 = svm (x_train, factor(y_train), type='C', kernel='linear')
svm_1_pred=predict(model_svm_1,newdata = x_test)
confusionMatrix(y_test,svm_1_pred) # Accuracy = 0.8246

model_svm_2 = svm (x_train, factor(y_train), type='C', kernel='polynomial', degree=2)
svm_2_pred = predict (model_svm_2, newdata = x_test)
confusionMatrix(y_test,svm_2_pred) # Accuracy = 0.7799
 
model_svm_3 = svm (x_train, factor(y_train), type='C', kernel='radial', gamma=0.1)
svm_3_pred = predict (model_svm_3, newdata=x_test)
confusionMatrix(y_test,svm_3_pred) # Accuracy = 0.8172

model_svm_4 = svm (x_train, factor(y_train), type='C', kernel='radial', gamma=0.1, cost=10,cross = 10)
svm_4_pred = predict (model_svm_4, newdata=x_test)
confusionMatrix(y_test,svm_4_pred) # Accuracy = 0.8246

library(randomForest)
help(svm)

dim(x_train)
table(y_train)
model_rf_1=randomForest(x_train,factor(y_train),ntree=500)
rf_1_pred = predict (model_rf_1, newdata=x_test)
confusionMatrix(y_test,rf_1_pred) # Accuracy = 0.8396

model_rf_2=randomForest(x_train,factor(y_train),ntree=1000)
rf_2_pred = predict (model_rf_2, newdata=x_test)
confusionMatrix(y_test,rf_2_pred) # Accuracy = 0.8433

varImpPlot(model_rf_2,sort = TRUE)

