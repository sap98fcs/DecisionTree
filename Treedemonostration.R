#PLSC 597: Methods Tutorial -- Classification with Logistic Regression
#Gary Fong
#February 23, 2021

###############################################################################
### RESEARCH QUESTIONS ###
# Using posters circulated on social media platform to track online opnion of social movement
# Using texts in a poster to predict its purpose


### SPECIFIC GOALS ###
# The tutorial will cover the following topics:
# 1) Taking a look at the data set 
# 2) An simple example of binary classification
# 3) Tuning the tree model
# 4) Cross validation of the tuned model 
# 5) Improving the performance of decision tree by the random forest approach
# 6) Making out-of-sample predictions

### THE DATA ###
# Data are from the my personal access(as part of the research team) to the ANTIELAB Research Data Archive at HKU (https://antielabdata.jmsc.hku.hk)
# We use webscrapping to download images from telegram channels widely subscribed in Telegram during the 2019 HK Protests
# The text in the poster are extracted by OCR and the manually corrected and coded by students helper
# The small sample (N=129) is translated into English for easier processing for non-Chinese speaker

### VARIABLES I USE TODAY ###
# Type - Type of the poster - Solidarity, Mobilize, Police, HKGovt, China
# The rest are a term-document matrix of columns of dummy variables indicating whether a word appear 
# This tutorial will be collapsed into binary classification, for full classification, visit https://raw.githubusercontent.com/sap98fcs/DecisionTree/main/Treedemonostration2.R

##### 1) Taking a look at the data set 

#install and load the necessary packages
install.packages("mlr")
install.packages("tidyverse")
library(mlr)
library(tidyverse)

#Import data from github
data <- read.csv("https://raw.githubusercontent.com/sap98fcs/DecisionTree/main/DecisionTree.csv")

#make the data into a tibble for easier viewing and manipulation
data <- as_tibble(data)

#see the data is complete
sum(map_dfr(data, ~sum(is.na(.))))

#Summary on types of poster
data %>% 
  group_by(Type) %>%
  summarise(no_rows = length(Type))

#Summary on number of predictors
count(data[2:63],as.factor(rowSums(data[2:63])))

# 2) An simple example of binary classification

#modify the data set for binary classification 
data$Type[data$Type != "Mobilize" ] <- "nonMobilize"
data$Type <- as.character(data$Type)

#make sure we have the same random number
set.seed(1234)

# sample row for traning set and form the training and test set
train_rows <- sample(seq_len(nrow(data)), nrow(data)*3/4)
train_data <- data[train_rows, ]
test_data <- data[-train_rows, ]
rm(train_rows)

#Summary on types of poster of the two sets
train_data %>% 
  group_by(Type) %>%
  summarise(no_rows = length(Type))
test_data %>% 
  group_by(Type) %>%
  summarise(no_rows = length(Type))

#Summary on number of predictors of the two sets
count(train_data[2:63],as.factor(rowSums(train_data[2:63])))
count(test_data[2:63],as.factor(rowSums(test_data[2:63])))

#set the task, learner, and train the model
tree <- makeLearner("classif.rpart",predict.type = "prob")
PosterTask <- makeClassifTask(data = train_data, target = "Type")
naivemodel <- train(tree, PosterTask)

#cross validation of the naive test
CV <- makeResampleDesc("CV", iters = 5)
naivemodel_CV <- resample(tree, PosterTask,
                          resampling = CV,
                          measures = acc)

naivemodel_CV$aggr

#view the tree
treeModelData <- getLearnerModel(naivemodel)
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(treeModelData, roundint = FALSE,
           box.palette = "BuBn",
           type =5)

rm(treeModelData)
rm(naivemodel)

# 3) Tuning the tree model with cross validation 
#The hyperparameters of a decision tree, see figure 7.7 in Rhys(2020)

getParamSet(tree)
#minisplit - mini no. of cases in a node required to further split
#minbucket - mini no. of cases required in a branch after a split
#cp - does each step make the model enough more useful
#maxdepth - no. of layers of the tree

# set these variables, smaller size split/bucket and more layers
treeParamSpace <- makeParamSet(
  makeIntegerParam("minsplit", lower = 1, upper = 5),
  makeIntegerParam("minbucket", lower = 1, upper = 5),
  makeNumericParam("cp", lower = 0.01, upper = 0.1),
  makeIntegerParam("maxdepth", lower = 10, upper = 30))

#squeeze our computational power by using more core
install.packages("parallel")
install.packages("parallelMap")
library(parallel)
library(parallelMap)

#Set the random search parameter
randSearch <- makeTuneControlRandom(maxit = 100)

#Performing hyperparameters tuning

parallelStartSocket(cpus = detectCores())

tunedTreePars <- tuneParams(tree, task = PosterTask,
                            resampling = CV,
                            par.set = treeParamSpace,
                            control = randSearch)

parallelStop()

#see the tuned parameters
tunedTreePars$x

#compare the performance of the tuning process with the naive model
tunedTreePars$y
naivemodel_CV$aggr

#get the tuned model 
tunedTree <- setHyperPars(tree, par.vals = tunedTreePars$x)
tunedTreeModel <- train(tunedTree, PosterTask)

#plot the new tree
treeModelData <- getLearnerModel(tunedTreeModel)
rpart.plot(treeModelData, roundint = FALSE,
           box.palette = "BuBn",
           type =5)

rm(treeModelData)
rm(tunedTree)
rm(naivemodel_CV)

# 4) Cross validation of the tuned model 

#wrap the learner with the tuning process)
treeWrapper <- makeTuneWrapper(tree, resampling = CV,
                               par.set = treeParamSpace,
                               control = randSearch)

#Cross-validating the model-building process

parallelStartSocket(cpus = detectCores())

cvWithTuning <- resample(treeWrapper, PosterTask, resampling = CV)

parallelStop()

#Compare the nested CV's accuracy with the Tuning process
cvWithTuning$aggr
tunedTreePars$y

rm(tunedTreePars)
rm(treeWrapper)
rm(treeParamSpace)
rm(tree)

# 5) Improve the performance of decision tree by the random forest approach

#The basic idea to create multiple model by bootstrapping, and then each model cast a vote on the prediction 
#Bootstrapping could reduce variance (avoiding overfitting) as outliners have less chance to be selected
#In each node, the number of features to be considered is randomly selected to increase the independency between model

#install the RF packet
install.packages("randomForest")
library(randomForest)

#define the new learner
forest <- makeLearner("classif.randomForest", predict.type = "prob")

getParamSet(forest)
#ntree - The number of trees in the forest
#mtry - The number of features to randomly sample at each node
#nodesize - The minimum number of cases after a split (the same as minbucket in the tree)
#maxnodes - The maximum number of leaves allowed

#set hyperparameters of random forest learner
forestParamSpace <- makeParamSet(                        
  makeIntegerParam("ntree", lower = 100, upper = 200),
  makeIntegerParam("mtry", lower = 15, upper = 30),
  makeIntegerParam("nodesize", lower = 1, upper = 5),
  makeIntegerParam("maxnodes", lower = 5, upper = 20))

parallelStartSocket(cpus = detectCores())

tunedForestPars <- tuneParams(forest, task = PosterTask,     
                              resampling = CV,    
                              par.set = forestParamSpace,   
                              control = randSearch)         

parallelStop()

#see the tuned parameter, and compare the accuracy with the tree model 
tunedForestPars                                           
cvWithTuning$aggr

#Train the new model
tunedForestModel <- train(setHyperPars(forest, par.vals = tunedForestPars$x), PosterTask)

#Cross-validation of the modeling process

forestWrapper <- makeTuneWrapper("classif.randomForest",
                                 resampling = CV,
                                 par.set = forestParamSpace,
                                 control = randSearch)

parallelStartSocket(cpus = detectCores())

cvWithTuning <- resample(forestWrapper, PosterTask, resampling = CV)

parallelStop()

#Compare the nested CV's accuracy with the Tuning process
tunedForestPars$y
cvWithTuning$aggr

rm(forestParamSpace)
rm(randSearch)
rm(tunedForestPars)
rm(train_data)
rm(forest)
rm(CV)
rm(cvWithTuning)
rm(forestWrapper)
rm(PosterTask)

# 6) making out-of-sample predictions 

#accuracy of the tree model
prediction_tree <- as.tibble(predict(tunedTreeModel, newdata = test_data))

prediction_tree

prediction_tree <- prediction_tree %>%
  mutate(correct = case_when(
    truth == response ~ 1,
    truth != response ~ 0
  ))

acc_tree <- sum(prediction_tree$correct)/nrow(prediction_tree)
acc_tree

#accuracy of the random-forest model

prediction_forest <- as.tibble(predict(tunedForestModel, newdata = test_data))

prediction_forest

prediction_forest <- prediction_forest %>%
  mutate(correct = case_when(
    truth == response ~ 1,
    truth != response ~ 0
  ))

acc_forest <- sum(prediction_forest$correct)/nrow(prediction_forest)
acc_forest

# Recode the data into 1(Mobilize) <-positive and 0(nonMobilize) <-negative

prediction_tree$truth <- as.character(prediction_tree$truth)
prediction_tree$response <- as.character(prediction_tree$response)
prediction_forest$truth <- as.character(prediction_forest$truth)
prediction_forest$response <- as.character(prediction_forest$response)

prediction_tree$truth[prediction_tree$truth == "nonMobilize"] <- 0
prediction_tree$truth[prediction_tree$truth == "Mobilize"] <- 1
prediction_tree$response[prediction_tree$response == "nonMobilize"] <- 0
prediction_tree$response[prediction_tree$response == "Mobilize"] <- 1

prediction_forest$truth[prediction_forest$truth == "nonMobilize"] <- 0
prediction_forest$truth[prediction_forest$truth == "Mobilize"] <- 1
prediction_forest$response[prediction_forest$response == "nonMobilize"] <- 0
prediction_forest$response[prediction_forest$response == "Mobilize"] <- 1


#False Positive Rate
#first, create a variable that identifies everything that was true 0, but classified as 1
prediction_tree <- prediction_tree %>%
  mutate(fp = case_when(
    (truth == 0 & response == 1) ~ 1,
    (truth == 1 | response == 0) ~ 0
  ))

#same for random forest 
prediction_forest <- prediction_forest %>%
  mutate(fp = case_when(
    (truth == 0 & response == 1) ~ 1,
    (truth == 1 | response == 0) ~ 0
  ))

#False Positive Rate = FP/All Negative
fpr_tree <- sum(prediction_tree$fp)/(nrow(prediction_tree)-sum(as.numeric(prediction_tree$truth)))
fpr_forest <- sum(prediction_forest$fp)/(nrow(prediction_forest)-sum(as.numeric(prediction_forest$truth)))

fpr_tree
fpr_forest 


#True Positive Rate (Recall) and Precision 

#first, create a variable that identifies everything that was true 1, and also classified as 1
prediction_tree <- prediction_tree %>%
  mutate(tp = case_when(
    (truth == 1 & response == 1) ~ 1,
    (truth == 0 | response == 0) ~ 0
  ))

prediction_forest <- prediction_forest %>%
  mutate(tp = case_when(
    (truth == 1 & response == 1) ~ 1,
    (truth == 0 | response == 0) ~ 0
  ))

#True Positive Rate/Recall = TP/All Positive
tpr_tree <- sum(prediction_tree$tp)/sum(as.numeric(prediction_tree$truth))
tpr_forest <- sum(prediction_forest$tp)/sum(as.numeric(prediction_forest$truth))

tpr_tree
tpr_forest

#Precision = TP/TP+FP
pre_tree <- sum(prediction_tree$tp)/(sum(prediction_tree$tp)+sum(prediction_tree$fp))
pre_forest <- sum(prediction_forest$tp)/(sum(prediction_forest$tp)+sum(prediction_forest$fp))

pre_tree
pre_forest

#install package for AUC PR Curve
install.packages("MLmetrics")
library(MLmetrics)

# Area under ROC curve
AUC(y_pred = prediction_tree$prob.Mobilize,y_true=prediction_tree$truth)
AUC(y_pred = prediction_forest$prob.Mobilize,y_true=prediction_forest$truth)

# Area under precision-recall curve
PRAUC(y_pred = prediction_tree$prob.Mobilize,y_true=prediction_tree$truth)
PRAUC(y_pred = prediction_forest$prob.Mobilize,y_true=prediction_forest$truth)
