
rm(list=ls())
library(openxlsx)
library(caret)
library(car)

df=read.xlsx("german_credit_data.xlsx",sheet = "Input_Data_Encoded")

df$target=ifelse(df$good_or_bad==1,0,1)
table(df$target)
head(df)
cols_to_drop=c("good_or_bad","status_of_existing_checking_account","credit_history","purpose","savings_account_or_bonds","present_employment_since","personal_status_and_sex","other_debtors_or_gurantors","property","other_installment_plans","housing","job","phone","foreign_worker")

df=subset(df, select = !(names(df) %in% cols_to_drop))
str(df)
names(df)



for(col in names(df)) {
  if(col!="target") { df[,col]=scale(df[,col]) }
}


table(df$target)
head(df)
set.seed(11)
part_index=createDataPartition(df$target,p=0.75,list=FALSE)
part_index

train_data=df[part_index,]
test_data=df[-part_index,]

# run logistic regression 
model_1=glm(target ~.,family = binomial(link = 'logit'),data=train_data)
summary(model_1)
vif(model_1)


# drop column : being_liable
model_2=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable)))
summary(model_2)
vif(model_2)

# drop columns : being_liable,job_encoded
model_3=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable,job_encoded)))
summary(model_3)
vif(model_3)

# drop columns : being_liable,job_encoded,present_residence_since
model_4=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable,job_encoded,present_residence_since)))
summary(model_4)
vif(model_4)

# drop columns : being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank
model_5=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank)))
summary(model_5)
vif(model_5)

# drop columns : being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded
model_6=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded)))
summary(model_6)
vif(model_6)


# drop columns : being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age
model_7=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age)))
summary(model_7)
vif(model_7)

# drop columns : being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age,personal_status_and_sex_encoded
model_8=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age,personal_status_and_sex_encoded)))
summary(model_8)
vif(model_8)

# drop columns : being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age,personal_status_and_sex_encoded,phone_encoded
model_9=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age,personal_status_and_sex_encoded,phone_encoded)))
summary(model_9)
vif(model_9)

# drop columns : being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age,personal_status_and_sex_encoded,phone_encoded,credit_amount
model_10=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age,personal_status_and_sex_encoded,phone_encoded,credit_amount)))
summary(model_10)
vif(model_10)



# drop columns : being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age,personal_status_and_sex_encoded,phone_encoded,credit_amount,other_installment_plans_encoded
model_11=glm(target ~.,family = binomial(link = 'logit'),data=subset(train_data,select=-c(being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age,personal_status_and_sex_encoded,phone_encoded,credit_amount,other_installment_plans_encoded)))
summary(model_11)
vif(model_11)

#Finding Predicted Values on test data
#Plotting ROC Curve
library(ROCR)
test_predicted=predict(model_11,newdata=subset(test_data,select=-c(being_liable,job_encoded,present_residence_since,no_of_existing_credits_at_this_bank,property_encoded,age,personal_status_and_sex_encoded,phone_encoded,credit_amount,other_installment_plans_encoded)),type="response")
test_pred=prediction(test_predicted,test_data$target)
test_perf=performance(test_pred,"tpr","fpr")
plot(test_perf)
test_auc=performance(test_pred,"auc")
test_auc=unlist(slot(test_auc, "y.values"))
test_auc

#Confusion Matrix

table(test_data$target, test_predicted > 0.5)
# accuracy = ~78%
