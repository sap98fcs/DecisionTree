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
# 3) Training a tree model for full classification and testing model performance with cross validation
# 4) Tuning the tree model
# 5) Cross validation of the tuned model 
# 6) Improving the performance of decision tree by the random forest approach
# 7) Making out-of-sample predictions and compare the normal and random forest model

### THE DATA ###
# Data are from the my personal access(as part of the research team) to the ANTIELAB Research Data Archive at HKU (https://antielabdata.jmsc.hku.hk)
# We use webscrapping to download images from telegram channels widely subscribed in Telegram during the 2019 HK Protests
# The text in the poster are extracted by OCR and the manually corrected and coded by students helper
# The small sample (N=129) is translated into English for easier processing for non-Chinese speaker

### VARIABLES I USE TODAY ###
# Type - Type of the poster - Solidarity, Mobilize, Police, HKGovt, China
# The rest are a term-document matrix of columns of dummy variables indicating whether a word appear 


##### 1) Taking a look at the data set 

#install  and load the necessary packages
install.packages("mlr")
install.packages("tidyverse")
install.packages("randomForest")
library(mlr)
library(tidyverse)
library(randomForest)

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

#See the predictors
colnames(data[,2:ncol(data)])

#Summary on number of predictors
count(data[2:63],as.factor(rowSums(data[2:63])))

# 2) An simple example of binary classification

#modify the dataset
data_bin <- data
data_bin$Type[data_bin$Type != "Mobilize" ] <- "nonMobilize" 

#set the task, learner, and train the model
simplePosterTask <- makeClassifTask(data = data_bin, target = "Type")
tree <- makeLearner("classif.rpart",predict.type = "prob")
simplemodel <- train(tree, simplePosterTask)

#cross validation of the naive test
CV <- makeResampleDesc("CV", iters = 5)
simplemodel_CV <- resample(tree, simplePosterTask,
                          resampling = CV,
                          measures = acc)
#view the tree
treeModelData <- getLearnerModel(simplemodel)
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(treeModelData, roundint = FALSE,
           box.palette = "BuBn",
           type =5)

rm(data_bin)
rm(simplemodel_CV)
rm(simplePosterTask)
rm(simplemodel)
rm(treeModelData)

# 3) Training a tree model for full classification and testing model performance with cross validation
#separate test set and training set

#make sure we have the same random number
set.seed(1234)

# sample row for traning set and form the training and test set
train_rows <- sample(seq_len(nrow(data)), nrow(data)*3/4)
train_data <- data[train_rows, ]
test_data <- data[-train_rows, ]
rm(train_rows)
rm(data)

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


#setting up test and do a naive test
PosterTask <- makeClassifTask(data = train_data, target = "Type")
naivemodel <- train(tree, PosterTask)

#cross validation of the naive test
CV <- makeResampleDesc("CV", iters = 5)
naivemodel_CV <- resample(tree, PosterTask,
                     resampling = CV,
                     measures = acc)
#It performs worse compare to the previous model for binary classification 

#view the tree
treeModelData <- getLearnerModel(naivemodel)
rpart.plot(treeModelData, roundint = FALSE,
           box.palette = "BuBn",
           type =5)

#It also wasted a lot of variables!
rm(naivemodel)
rm(treeModelData)

# 4) Tuning the tree model with cross validation 
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
randSearch <- makeTuneControlRandom(maxit = 200)

#Performing hyperparameters tuning

parallelStartSocket(cpus = detectCores())

tunedTreePars <- tuneParams(tree, task = PosterTask,
                            resampling = CV,
                            par.set = treeParamSpace,
                            control = randSearch)

parallelStop()

#compare the performance of the tuning process with the naive model
tunedTreePars$y
naivemodel_CV$aggr
rm(naivemodel_CV)

#see the tuned parameters
tunedTreePars$x

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

#Download the pdf and see what have happened!

# 5) Cross validation of the tuned model 

#wrap the learner with the tuning process)
treeWrapper <- makeTuneWrapper(tree, resampling = CV,
                               par.set = treeParamSpace,
                               control = randSearch)

#Cross-validating the model-building process

parallelStartSocket(cpus = detectCores())

cvWithTuning <- resample(treeWrapper, PosterTask, resampling = CV)

parallelStop()

#The nested CV's accuracy is similar to that of the Tuning process, meaning that the model is a good fit!
cvWithTuning
tunedTreePars$y

rm(tunedTreePars)
rm(treeWrapper)
rm(treeParamSpace)
rm(tree)

# 6) Improve the performance of decision tree by the random forest approach

#The basic idea to create multiple model by bootstrapping, and then each model cast a vote on the prediction 
#Bootstrapping could reduce variance (avoiding overfitting) as outliners have less chance to be selected
#In each node, the number of features to be considered is randomly selected to increase the independency between model

#define the new learner
forest <- makeLearner("classif.randomForest", predict.type = "prob")

#set hyperparameters of random forest learner
#ntree— The number of trees in the forest
#mtry— The number of features to randomly sample at each node
#nodesize— The minimum number of cases after a split (the same as minbucket in the tree)
#maxnodes— The maximum number of leaves allowed

forestParamSpace <- makeParamSet(                        
   makeIntegerParam("ntree", lower = 100, upper = 300),
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

#The nested CV's accuracy a little bit lower than that of the tuning process, meaning that the model is a little bit over-fitted!
cvWithTuning
tunedTreePars$y

rm(forestParamSpace)
rm(randSearch)
rm(tunedForestPars)
rm(train_data)
rm(forest)
rm(CV)
rm(cvWithTuning)
rm(forestParamSpace)
rm(forestWrapper)
rm(PosterTask)

# 7) Making out-of-sample predictions 

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

#The accuracy is the same because posters for Mobilize and Solidarity share many similar feature.

#get feature importance in a model
t <- getFeatureImportance(tunedForestModel)
t <- t$res[order(t$res[, "importance"], decreasing = TRUE),]

z <- getFeatureImportance(tunedTreeModel)
z <- z$res[order(z$res[, "importance"], decreasing = TRUE),]

install.packages("dplyr")
library(dplyr)
featureimportance <- inner_join(t,z, by="variable")
