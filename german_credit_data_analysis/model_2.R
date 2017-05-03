rm(list=ls())
library(openxlsx)
library(caret)
library(car)
library(neuralnet)
library(e1071)
library(glmnet)
library(corrplot)
library(ROCR)

set.seed(11)
df=read.xlsx("german_credit_data.xlsx",sheet = "Input_Data_Encoded")

df$target=ifelse(df$good_or_bad==1,0,1)
table(df$target)
head(df)
cols_to_drop=c("good_or_bad","status_of_existing_checking_account","credit_history","purpose","savings_account_or_bonds","present_employment_since","personal_status_and_sex","other_debtors_or_gurantors","property","other_installment_plans","housing","job","phone","foreign_worker")

df=subset(df, select = !(names(df) %in% cols_to_drop))
str(df)
names(df)

#check missing values
apply(df,2,function(x) sum(is.na(x)))

target=df$target
df$target=NULL

maxs=apply(df,2,max)
mins=apply(df,2,min)
scaled_df=as.data.frame(scale(df,center = mins,scale = maxs-mins))
head(scaled_df)
scaled_df$target=target
head(scaled_df)
str(scaled_df)

corrplot(cor(scaled_df))

# create train/validation/test data
part_index=createDataPartition(scaled_df$target,p=0.8,list=FALSE)
part_index

train_data=scaled_df[part_index,]
test_data=scaled_df[-part_index,]

idx2=createDataPartition(train_data$target,p=0.8,list=FALSE)
validation_data=train_data[-idx2,]
train_data=train_data[idx2,]



boxplot(scaled_df$age)
histogram(scaled_df$age)
summary(scaled_df$age)
table(scaled_df$age)

boxplot(scaled_df$status_of_existing_checking_account_encoded)
histogram(scaled_df$status_of_existing_checking_account_encoded)
summary(scaled_df$status_of_existing_checking_account_encoded)
table(scaled_df$status_of_existing_checking_account_encoded)

boxplot(scaled_df$duration_in_month)
histogram(scaled_df$duration_in_month)
summary(scaled_df$duration_in_month)
table(scaled_df$duration_in_month)


boxplot(scaled_df$credit_history_encoded)
histogram(scaled_df$credit_history_encoded)
summary(scaled_df$credit_history_encoded)
table(scaled_df$credit_history_encoded)

boxplot(scaled_df$purpose_encoded)
histogram(scaled_df$purpose_encoded)
summary(scaled_df$purpose_encoded)
table(scaled_df$purpose_encoded)

boxplot(scaled_df$present_employment_since_encoded)
histogram(scaled_df$present_employment_since_encoded)
summary(scaled_df$present_employment_since_encoded)
table(scaled_df$present_employment_since_encoded)


boxplot(scaled_df$other_debtors_or_gurantors_encoded)
histogram(scaled_df$other_debtors_or_gurantors_encoded)
summary(scaled_df$other_debtors_or_gurantors_encoded)
table(scaled_df$other_debtors_or_gurantors_encoded)

boxplot(scaled_df$housing_encoded)
histogram(scaled_df$housing_encoded)
summary(scaled_df$housing_encoded)
table(scaled_df$housing_encoded)


# Logistic regression - start
lr_model_1=glm(target ~.,family = binomial(link = 'logit'),data=train_data)
summary(lr_model_1)
vif(lr_model_1)

varImp(lr_model_1)

lr_model_2=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=c(target,status_of_existing_checking_account_encoded,credit_history_encoded,savings_account_or_bonds_encoded,purpose_encoded,present_employment_since_encoded,installment_rate,duration_in_month,other_debtors_or_gurantors_encoded,foreign_worker_encoded,housing_encoded,credit_amount,personal_status_and_sex_encoded,phone_encoded)))
summary(lr_model_2)
vif(lr_model_2)


lr_model_3=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=c(target,status_of_existing_checking_account_encoded,credit_history_encoded,savings_account_or_bonds_encoded,purpose_encoded,present_employment_since_encoded,installment_rate,duration_in_month,other_debtors_or_gurantors_encoded,foreign_worker_encoded,housing_encoded,credit_amount,personal_status_and_sex_encoded)))
summary(lr_model_3)
vif(lr_model_3)


lr_model_4=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=c(target,status_of_existing_checking_account_encoded,credit_history_encoded,savings_account_or_bonds_encoded,purpose_encoded,present_employment_since_encoded,installment_rate,duration_in_month,other_debtors_or_gurantors_encoded,foreign_worker_encoded,housing_encoded,credit_amount)))
summary(lr_model_4)
vif(lr_model_4)


lr_model_5=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=c(target,status_of_existing_checking_account_encoded,credit_history_encoded,savings_account_or_bonds_encoded,purpose_encoded,present_employment_since_encoded,installment_rate,duration_in_month,other_debtors_or_gurantors_encoded,foreign_worker_encoded,housing_encoded)))
summary(lr_model_5)
vif(lr_model_5)

lr_model_6=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=c(target,status_of_existing_checking_account_encoded,credit_history_encoded,savings_account_or_bonds_encoded,purpose_encoded,present_employment_since_encoded,installment_rate,duration_in_month,other_debtors_or_gurantors_encoded,foreign_worker_encoded)))
summary(lr_model_6)
vif(lr_model_6)

lr_model_7=glm(as.factor(target) ~.,family = binomial(link = 'logit'),data=subset(train_data,select=c(target,status_of_existing_checking_account_encoded,credit_history_encoded,savings_account_or_bonds_encoded,purpose_encoded,present_employment_since_encoded,installment_rate,duration_in_month,other_debtors_or_gurantors_encoded)))
summary(lr_model_7)
vif(lr_model_7)

pred_lr=predict(lr_model_7,newdata=
                         subset(validation_data,select=c(status_of_existing_checking_account_encoded,credit_history_encoded,savings_account_or_bonds_encoded,purpose_encoded,present_employment_since_encoded,installment_rate,duration_in_month,other_debtors_or_gurantors_encoded)),
                         type="response")
head(pred_lr)
#confusionMatrix(validation_data$target,ifelse(pred_lr>0.5,1,0))
caret::confusionMatrix(ifelse(pred_lr>0.5,1,0),validation_data$target,positive="0")

# Accuracy : 0.7625          
# 95% CI : (0.6889, 0.8261)
# No Information Rate : 0.7062          
# P-Value [Acc > NIR] : 0.06793         
# 
# Kappa : 0.4054          
# Mcnemar's Test P-Value : 0.41730         
#                                           
#             Sensitivity : 0.8584          
#             Specificity : 0.5319          
#          Pos Pred Value : 0.8151          
#          Neg Pred Value : 0.6098

pred_lr_test=predict(lr_model_7,newdata=
                  subset(test_data,select=c(status_of_existing_checking_account_encoded,credit_history_encoded,savings_account_or_bonds_encoded,purpose_encoded,present_employment_since_encoded,installment_rate,duration_in_month,other_debtors_or_gurantors_encoded)),
                type="response")
head(pred_lr_test)
caret::confusionMatrix(ifelse(pred_lr_test>0.5,1,0),test_data$target,positive="0")
#Metrics::auc(test_data$target,ifelse(pred_lr_test>0.5,1,0))
MLmetrics::LogLoss(pred_lr_test,test_data$target)
# Accuracy : 0.775           
# 95% CI : (0.7108, 0.8309)
# No Information Rate : 0.715           
# P-Value [Acc > NIR] : 0.0336          
# 
# Kappa : 0.4139          
# Mcnemar's Test P-Value : 0.1360          
#                                           
#             Sensitivity : 0.8811          
#             Specificity : 0.5088          
#          Pos Pred Value : 0.8182          
#          Neg Pred Value : 0.6304

###########################################################
# Function OptimisedConc : for concordance, discordance, ties
# The function returns Concordance, discordance, and ties
# by taking a glm binomial model result as input.
# Although it still uses two-for loops, it optimises the code
# by creating initial zero matrices
###########################################################
OptimisedConc=function(model)
{
  Data = cbind(model$y, model$fitted.values) 
  ones = Data[Data[,1] == 1,]
  zeros = Data[Data[,1] == 0,]
  conc=matrix(0, dim(zeros)[1], dim(ones)[1])
  disc=matrix(0, dim(zeros)[1], dim(ones)[1])
  ties=matrix(0, dim(zeros)[1], dim(ones)[1])
  for (j in 1:dim(zeros)[1])
  {
    for (i in 1:dim(ones)[1])
    {
      if (ones[i,2]>zeros[j,2])
      {conc[j,i]=1}
      else if (ones[i,2]<zeros[j,2])
      {disc[j,i]=1}
      else if (ones[i,2]==zeros[j,2])
      {ties[j,i]=1}
    }
  }
  Pairs=dim(zeros)[1]*dim(ones)[1]
  PercentConcordance=(sum(conc)/Pairs)*100
  PercentDiscordance=(sum(disc)/Pairs)*100
  PercentTied=(sum(ties)/Pairs)*100
  return(list("Percent Concordance"=PercentConcordance,"Percent Discordance"=PercentDiscordance,"Percent Tied"=PercentTied,"Pairs"=Pairs))
}

OptimisedConc(lr_model_7)

library(glmnet)
lasso_model=cv.glmnet(data.matrix(subset(train_data,select = c(status_of_existing_checking_account_encoded,
                                                                credit_history_encoded,
                                                                savings_account_or_bonds_encoded,
                                                                purpose_encoded,present_employment_since_encoded,
                                                                installment_rate,duration_in_month,
                                                                other_debtors_or_gurantors_encoded))),
                               train_data$target,family="binomial",alpha=1,standardize=F)

pred_lasso_validation=predict(lasso_model,newx=
                       subset(data.matrix(subset(validation_data,
                            select=c(status_of_existing_checking_account_encoded,
                                     credit_history_encoded,savings_account_or_bonds_encoded,
                                     purpose_encoded,present_employment_since_encoded,installment_rate,
                                     duration_in_month,other_debtors_or_gurantors_encoded)))),
                     type="response")
head(pred_lasso_validation)
caret::confusionMatrix(ifelse(pred_lasso_validation>0.5,1,0),validation_data$target,positive="0")
MLmetrics::LogLoss(pred_lasso_validation,validation_data$target)
# Accuracy : 0.7688          
# 95% CI : (0.6956, 0.8317)
# No Information Rate : 0.7062          
# P-Value [Acc > NIR] : 0.047206        
# 
# Kappa : 0.3681          
# Mcnemar's Test P-Value : 0.003085        
# 
# Sensitivity : 0.9204          
# Specificity : 0.4043          
# Pos Pred Value : 0.7879          
# Neg Pred Value : 0.6786

pred_lasso_test=predict(lasso_model,newx=
                                subset(data.matrix(subset(test_data,
                                                          select=c(status_of_existing_checking_account_encoded,
                                                                   credit_history_encoded,savings_account_or_bonds_encoded,
                                                                   purpose_encoded,present_employment_since_encoded,installment_rate,
                                                                   duration_in_month,other_debtors_or_gurantors_encoded)))),
                              type="response")
head(pred_lasso_test)
caret::confusionMatrix(ifelse(pred_lasso_test>0.5,1,0),test_data$target,positive="0")
MLmetrics::LogLoss(pred_lasso_test,test_data$target)
# Accuracy : 0.765        
# 95% CI : (0.7, 0.8219)
# No Information Rate : 0.715        
# P-Value [Acc > NIR] : 0.066358     
# 
# Kappa : 0.3438       
# Mcnemar's Test P-Value : 0.001332     
# 
# Sensitivity : 0.9161       
# Specificity : 0.3860       
# Pos Pred Value : 0.7892       
# Neg Pred Value : 0.6471


lasso_model=cv.glmnet(data.matrix(subset(train_data,select =-c(target) )),
                      train_data$target,family="binomial",alpha=1,standardize=F)

pred_lasso_validation=predict(lasso_model,newx=
                                subset(data.matrix(subset(validation_data,
                                                          select=-c(target)))),
                              type="response")
head(pred_lasso_validation)
caret::confusionMatrix(ifelse(pred_lasso_validation>0.5,1,0),validation_data$target,positive="0")
MLmetrics::LogLoss(pred_lasso_validation,validation_data$target)
# Accuracy : 0.7625          
# 95% CI : (0.6889, 0.8261)
# No Information Rate : 0.7062          
# P-Value [Acc > NIR] : 0.067931        
# 
# Kappa : 0.3464          
# Mcnemar's Test P-Value : 0.002055        
# 
# Sensitivity : 0.9204          
# Specificity : 0.3830          
# Pos Pred Value : 0.7820          
# Neg Pred Value : 0.6667 

pred_lasso_test=predict(lasso_model,newx=
                          subset(data.matrix(subset(test_data,
                                                    select=-c(target)))),
                        type="response")
head(pred_lasso_test)
caret::confusionMatrix(ifelse(pred_lasso_test>0.5,1,0),test_data$target,positive="0")
# Accuracy : 0.76            
# 95% CI : (0.6947, 0.8174)
# No Information Rate : 0.715           
# P-Value [Acc > NIR] : 0.089927        
# 
# Kappa : 0.3338          
# Mcnemar's Test P-Value : 0.002437        
#                                           
#             Sensitivity : 0.9091          
#             Specificity : 0.3860          
#          Pos Pred Value : 0.7879          
#          Neg Pred Value : 0.6286 

new_train=rbind(train_data,validation_data)
lasso_model=cv.glmnet(data.matrix(subset(new_train,select =-c(target) )),
                      new_train$target,family="binomial",alpha=1)
pred_lasso_test=predict(lasso_model,newx=
                          subset(data.matrix(subset(test_data,
                                                    select=-c(target)))),
                        type="response")
head(pred_lasso_test)
caret::confusionMatrix(ifelse(pred_lasso_test>0.5,1,0),test_data$target,positive="0")
# Accuracy : 0.755           
# 95% CI : (0.6894, 0.8129)
# No Information Rate : 0.715           
# P-Value [Acc > NIR] : 0.1190692       
# 
# Kappa : 0.299           
# Mcnemar's Test P-Value : 0.0002038       
# 
# Sensitivity : 0.9231          
# Specificity : 0.3333          
# Pos Pred Value : 0.7765          
# Neg Pred Value : 0.6333

#ridge model
ridge_model=cv.glmnet(data.matrix(subset(new_train,select =-c(target) )),
                      new_train$target,family="binomial",alpha=0)
pred_ridge_test=predict(ridge_model,newx=
                          subset(data.matrix(subset(test_data,
                                                    select=-c(target)))),
                        type="response")
head(pred_ridge_test)
caret::confusionMatrix(ifelse(pred_ridge_test>0.5,1,0),test_data$target,positive="0")
MLmetrics::LogLoss(pred_ridge_test,test_data$target)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 133  37
# 1  10  20
# 
# Accuracy : 0.765                 
# 95% CI : (0.7000381, 0.8219293)
# No Information Rate : 0.715                 
# P-Value [Acc > NIR] : 0.0663580628          
# 
# Kappa : 0.3276109             
# Mcnemar's Test P-Value : 0.0001491444          
# 
# Sensitivity : 0.9300699             
# Specificity : 0.3508772             
# Pos Pred Value : 0.7823529             
# Neg Pred Value : 0.6666667             
# Prevalence : 0.7150000             
# Detection Rate : 0.6650000             
# Detection Prevalence : 0.8500000             
# Balanced Accuracy : 0.6404736             
# 
# 'Positive' Class : 0    

#elastic net model
enet_model=cv.glmnet(data.matrix(subset(new_train,select =-c(target) )),
                      new_train$target,family="binomial")
pred_enet_test=predict(enet_model,newx=
                          data.matrix(subset(test_data,
                                                    select=-c(target))),
                        type="response")
head(pred_enet_test)
caret::confusionMatrix(ifelse(pred_enet_test>0.5,1,0),test_data$target,positive="0")
MLmetrics::LogLoss(pred_enet_test,test_data$target)

####################################################################
#neuralnet model
####################################################################
colnames=names(new_train)
print(colnames)
form=as.formula(paste("target ~ ",paste(colnames[!colnames %in% "target"],collapse = "+")))
print(form)
nnet_model1=neuralnet(formula = form,data = new_train,linear.output = F,err.fct = "ce")
print(nnet_model1)
plot(nnet_model1)
gwplot(nnet_model1,selected.covariate = "status_of_existing_checking_account_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "duration_in_month",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "credit_history_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "purpose_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "credit_amount",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "savings_account_or_bonds_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "present_employment_since_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "installment_rate",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "personal_status_and_sex_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "other_debtors_or_gurantors_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "present_residence_since",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "property_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "age",min=-5,max=5) # flat
gwplot(nnet_model1,selected.covariate = "other_installment_plans_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "housing_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "no_of_existing_credits_at_this_bank",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "job_encoded",min=-5,max=5) # flat
gwplot(nnet_model1,selected.covariate = "being_liable",min=-5,max=5) # flat
gwplot(nnet_model1,selected.covariate = "phone_encoded",min=-5,max=5)
gwplot(nnet_model1,selected.covariate = "foreign_worker_encoded",min=-5,max=5)


pred_nnet_model1=compute(nnet_model1,subset(test_data,select=-c(target)))
caret::confusionMatrix(ifelse(pred_nnet_model1$net.result>0.5,1,0),test_data$target,positive="0")
MLmetrics::LogLoss(pred_nnet_model1$net.result,test_data$target)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 113  22
# 1  30  35
# 
# Accuracy : 0.74                  
# 95% CI : (0.6734222, 0.7993151)
# No Information Rate : 0.715                 
# P-Value [Acc > NIR] : 0.2422505             
# 
# Kappa : 0.3878752             
# Mcnemar's Test P-Value : 0.3316851             
# 
# Sensitivity : 0.7902098             
# Specificity : 0.6140351             
# Pos Pred Value : 0.8370370             
# Neg Pred Value : 0.5384615             
# Prevalence : 0.7150000             
# Detection Rate : 0.5650000             
# Detection Prevalence : 0.6750000             
# Balanced Accuracy : 0.7021224             
# 
# 'Positive' Class : 0  

#neural net model - dropping age,being_liable,job_encoded
form=as.formula(paste("target ~ ",
                      paste(colnames[!colnames %in% c("age","job_encoded","being_liable","target")],collapse = "+")))
print(form)
nnet_model2=neuralnet(formula = form,data = new_train,linear.output = F,err.fct = "ce")
print(nnet_model2)
plot(nnet_model2)
gwplot(nnet_model2,selected.covariate = "status_of_existing_checking_account_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "duration_in_month",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "credit_history_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "purpose_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "credit_amount",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "savings_account_or_bonds_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "present_employment_since_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "installment_rate",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "personal_status_and_sex_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "other_debtors_or_gurantors_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "present_residence_since",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "property_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "other_installment_plans_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "housing_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "no_of_existing_credits_at_this_bank",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "phone_encoded",min=-5,max=5)
gwplot(nnet_model2,selected.covariate = "foreign_worker_encoded",min=-5,max=5)


pred_nnet_model2=compute(nnet_model2,subset(test_data,select=-c(age,job_encoded,being_liable,target)))
caret::confusionMatrix(ifelse(pred_nnet_model2$net.result>0.5,1,0),test_data$target,positive="0")
MLmetrics::LogLoss(pred_nnet_model2$net.result,test_data$target)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 113  23
# 1  30  34
# 
# Accuracy : 0.735                 
# 95% CI : (0.6681299, 0.7947609)
# No Information Rate : 0.715                 
# P-Value [Acc > NIR] : 0.2945354             
# 
# Kappa : 0.3729295             
# Mcnemar's Test P-Value : 0.4098467             
# 
# Sensitivity : 0.7902098             
# Specificity : 0.5964912             
# Pos Pred Value : 0.8308824             
# Neg Pred Value : 0.5312500             
# Prevalence : 0.7150000             
# Detection Rate : 0.5650000             
# Detection Prevalence : 0.6800000             
# Balanced Accuracy : 0.6933505             
# 
# 'Positive' Class : 0  

#neural net model - dropping age,being_liable,job_encoded - trying with more neurons in hidden layer
form=as.formula(paste("target ~ ",
                      paste(colnames[!colnames %in% c("age","job_encoded","being_liable","target")],collapse = "+")))
print(form)
nnet_model3=neuralnet(formula = form,data = new_train,hidden = 5,err.fct = "ce",linear.output = F)
print(nnet_model3)
plot(nnet_model3)
gwplot(nnet_model3,selected.covariate = "status_of_existing_checking_account_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "duration_in_month",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "credit_history_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "purpose_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "credit_amount",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "savings_account_or_bonds_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "present_employment_since_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "installment_rate",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "personal_status_and_sex_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "other_debtors_or_gurantors_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "present_residence_since",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "property_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "other_installment_plans_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "housing_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "no_of_existing_credits_at_this_bank",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "phone_encoded",min=-5,max=5)
gwplot(nnet_model3,selected.covariate = "foreign_worker_encoded",min=-5,max=5)


pred_nnet_model3=compute(nnet_model3,subset(test_data,select=-c(age,job_encoded,being_liable,target)))
caret::confusionMatrix(ifelse(pred_nnet_model3$net.result>0.5,1,0),test_data$target,positive="0")
MLmetrics::LogLoss(pred_nnet_model3$net.result,test_data$target)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 120  33
# 1  23  24
# 
# Accuracy : 0.72                  
# 95% CI : (0.6523113, 0.7810388)
# No Information Rate : 0.715                 
# P-Value [Acc > NIR] : 0.4732550             
# 
# Kappa : 0.2747053             
# Mcnemar's Test P-Value : 0.2291019             
# 
# Sensitivity : 0.8391608             
# Specificity : 0.4210526             
# Pos Pred Value : 0.7843137             
# Neg Pred Value : 0.5106383             
# Prevalence : 0.7150000             
# Detection Rate : 0.6000000             
# Detection Prevalence : 0.7650000             
# Balanced Accuracy : 0.6301067             
# 
# 'Positive' Class : 0 

library(rpart)
dt1=rpart(as.factor(target)~.,data=new_train)
plot(dt1)
print(dt1)
summary(dt1)
pred_dt1=predict(dt1,newdata = subset(test_data,select=-c(target)))
caret::confusionMatrix(ifelse(pred_dt1>0.5,1,0),test_data$target,positive="0")
MLmetrics::LogLoss(pred_dt1,test_data$target)
varImp(dt1)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 115  23
# 1  28  34
# 
# Accuracy : 0.745                 
# 95% CI : (0.6787246, 0.8038592)
# No Information Rate : 0.715                 
# P-Value [Acc > NIR] : 0.1952772             
# 
# Kappa : 0.3903897             
# Mcnemar's Test P-Value : 0.5754030             
# 
# Sensitivity : 0.8041958             
# Specificity : 0.5964912             
# Pos Pred Value : 0.8333333             
# Neg Pred Value : 0.5483871             
# Prevalence : 0.7150000             
# Detection Rate : 0.5750000             
# Detection Prevalence : 0.6900000             
# Balanced Accuracy : 0.7003435             
# 
# 'Positive' Class : 0  

#randomforest model
library(randomForest)
rf1=randomForest(as.factor(target)~.,data=new_train,ntree=1000,probability=T)
varImp(rf1)
pred_rf1=predict(rf1,newdata = subset(test_data,select=-c(target)),probability=T)
caret::confusionMatrix(pred_rf1,test_data$target,positive="0")
#MLmetrics::LogLoss(pred_rf1,test_data$target)

#SVM model
svm_1=svm(as.factor(target)~.,data=new_train,scale = F,probability=T)
pred_svm1=predict(svm_1,newdata = subset(test_data,select=-c(target)),probability=T)
head(pred_svm1)
caret::confusionMatrix(pred_svm1,test_data$target,positive="0")
Metrics::auc(test_data$target,pred_svm1)
MLmetrics::LogLoss(attr(pred_svm1,"probabilities")[,1],test_data$target)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 133  36
# 1  10  21
# 
# Accuracy : 0.77                  
# 95% CI : (0.7053936, 0.8264191)
# No Information Rate : 0.715                 
# P-Value [Acc > NIR] : 0.0478078658          
# 
# Kappa : 0.3459406             
# Mcnemar's Test P-Value : 0.0002277626          
# 
# Sensitivity : 0.9300699             
# Specificity : 0.3684211             
# Pos Pred Value : 0.7869822             
# Neg Pred Value : 0.6774194             
# Prevalence : 0.7150000             
# Detection Rate : 0.6650000             
# Detection Prevalence : 0.8450000             
# Balanced Accuracy : 0.6492455             
# 
# 'Positive' Class : 0 


svm_2=svm(as.factor(target)~.,data=new_train,scale = F, kernel="linear",probability=TRUE)
pred_svm2=predict(svm_2,newdata = subset(test_data,select=-c(target)),probability=TRUE)
head(pred_svm2)
head(attr(pred_svm2,"probabilities"))
caret::confusionMatrix(pred_svm2,test_data$target,positive="0")
MLmetrics::LogLoss(attr(pred_svm2,"probabilities")[,1],test_data$target)
#[1] 1.259624836
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 123  29
# 1  20  28
# 
# Accuracy : 0.755                 
# 95% CI : (0.6893601, 0.8129159)
# No Information Rate : 0.715                 
# P-Value [Acc > NIR] : 0.1190692             
# 
# Kappa : 0.368882              
# Mcnemar's Test P-Value : 0.2530979             
# 
# Sensitivity : 0.8601399             
# Specificity : 0.4912281             
# Pos Pred Value : 0.8092105             
# Neg Pred Value : 0.5833333             
# Prevalence : 0.7150000             
# Detection Rate : 0.6150000             
# Detection Prevalence : 0.7600000             
# Balanced Accuracy : 0.6756840             
# 
# 'Positive' Class : 0

svm_3=svm(as.factor(target)~.,data=train_data,scale = F, kernel="linear",probability=TRUE)
pred_svm3=predict(svm_3,newdata = subset(test_data,select=-c(target)),probability=TRUE)
head(pred_svm3)
head(attr(pred_svm3,"probabilities"))
caret::confusionMatrix(pred_svm3,test_data$target,positive="0")
MLmetrics::LogLoss(attr(pred_svm3,"probabilities")[,1],test_data$target)
#[1] 1.303175442