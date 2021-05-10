##################
#load libraries
library(caret)
library(pROC)
#load data
data <- read.table("newthyroid.txt", header=TRUE, sep=",", stringsAsFactors=TRUE)
#view properties of data
str(data)
summary(data)
#scale data
data_s=scale(data[,-1])

##part 1##

# apply LDA and kNN, split data into training and test, repeat 20 times
A = 20 #Set A=20 for loop
set.seed(8) #set seed for reproducibility
#define index, repeated 20 times
trainIndex = createDataPartition(data$class, p = 0.7, list = FALSE, times = A)
table(data[,1])  # check data - imbalanced, therefore using SMOTE in kNN and LDA

#specify k values in a grid and create vectors to store AUC's
grid = expand.grid(k = c(3, 5, 7, 9, 11, 13, 15, 17, 19, 21))
auc_knn=vector("numeric",A)
auc_lda=vector("numeric",A)

#initiate for loop to repeat the training and test data split 
for(ii in 1:A){
  train.feature=data_s[trainIndex[,ii],-1] # training features 
  train.label=as.factor(data$class[trainIndex[,ii]]) # training labels 
  test.feature=data_s[-trainIndex[,ii],-1] # test features 
  test.label=as.factor(data$class[-trainIndex[,ii]]) # test labels
  
  ## set up train control, 5 fold CV, repeated 5 times
  fitControl <- trainControl(method = "repeatedcv",number = 5,
    repeats = 5,summaryFunction = twoClassSummary, classProbs = TRUE, 
    sampling="smote") #set upsampling
  
  ## training process - kNN
  set.seed(8)
  knnFit=train(train.feature,train.label, method = "knn",trControl = fitControl,
               metric = "ROC", preProcess = c("center","scale"), tuneGrid = grid)
  knnFit
  knnPred = predict(knnFit, test.feature)
  
  ## training process - LDA
  set.seed(8)
  ldaFit=train(train.feature,train.label, method = "lda",
                trControl = trainControl(sampling="smote")) #set upsampling
  ldaFit
  ldaPred = predict(ldaFit,test.feature) 
  
  #Recording 20 AUC values for kNN using the optimal k value
  knn.probs <- predict(knnFit,test.feature,type="prob") 
  knn.ROCs <- roc(predictor=knn.probs$h, response=test.label, levels = c("h", "n"), direction = ">")
  auc_knn[ii]=knn.ROCs$auc
  
  #Recording 20 AUC values for LDA 
  lda.probs <- predict(ldaFit,test.feature,type="prob") 
  lda.ROCs <- roc(predictor=lda.probs$h, response=test.label, levels = c("h", "n"), direction = ">")
  auc_lda[ii]=lda.ROCs$auc
}

#print the vector of 20 AUC values of knn lda
print(auc_knn)  
print(auc_lda)

##part 2##

#extract ROC 's for the first random split 
set.seed(8) 
train.feature1 = data_s[trainIndex[,1], -1]
train.label1 = data$class[trainIndex[,1]]

test.feature1 = data_s[-trainIndex[,1], -1]
test.label1 = data$class[-trainIndex[,1]]

#obtaining the area of the ROC curve for kNN
knnprobs1 <- predict(knnFit, test.feature1, type = "prob") 
knnROC1 <- roc(predictor = knnprobs1$h, response = test.label1,levels = c("h", "n"), direction = ">")
auc_knn1 = knnROC1$auc
print(auc_knn1)

#obtaining the area of the ROC curve for LDA
ldaprobs1 <- predict(ldaFit, test.feature1, type = "prob")
ldaROC1 <- roc(predictor = ldaprobs1$h,response = test.label1,levels = c("h", "n"), direction = ">")
auc_lda1 = ldaROC1$auc
print(auc_lda1)

#plot ROC curve of kNN and LDA on one plot
plot(knnROC1,main="ROC curve for first random split",col="deeppink",lwd = 4, lty=3 ) #plot ROC curve of kNN
lines(ldaROC1,col="cyan2") #plot ROC curve of LDA
legend("bottomright",legend=c("kNN","LDA"),col=c("deeppink","cyan2"),lty=c(1,1),cex=1,text.font=2) #legend properties

##part 3##

# Draw boxplot on one plot
boxplot(auc_knn, auc_lda, col=c("deeppink","cyan2"), main="AUC: kNN vs. LDA", 
        names = c('kNN','LDA'), lwd=0.7, las=2, ylim = c(0.990, 1.000))
