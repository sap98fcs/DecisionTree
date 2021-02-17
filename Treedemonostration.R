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
# 2) training a tree model and testing model performance with cross validation
# 3) Tune the tree model
# 4) Cross validation of the tuned model 
# 5) making out-of-sample predictions

### THE DATA ###
# Data are from the my personal access(as part of the resaerch team) to the ANTIELAB Research Data Archive at HKU (https://antielabdata.jmsc.hku.hk)
# We use webscrapping to download images from telegram channels widely subscribed in Telegram durng the 2019 HK Protests
# The text in the poster are extracted by OCR and the manually corrected and coded by students helper
# The small sample (N=129) is translated into english for easier processing for non-chinese speaker

### VARIABLES I USE TODAY ###
# Type - Type of the poster - Solidarity, Mobilize, Police, HKGovt, China
# The rest are a term-document matrix of columns of dummy variables indicating whether a word appear 


##### 1) Taking a look at the data set 

#install  and load the necessary packages
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

# 2) training a tree model and testing model performance with cross validation

#separate test set and training set

#make sure we have the same random number
set.seed(1000)

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


#setting up test and do a naive test
PosterTask <- makeClassifTask(data = train_data, target = "Type")
tree <- makeLearner("classif.rpart")
naivemodel <- train(tree, PosterTask)

#cross validation of the naive test
CV <- makeResampleDesc("CV", iters = 5, stratify = TRUE)
naivemodel_CV <- resample(tree, PosterTask,
                     resampling = CV,
                     measures = acc)
#view the tree
treeModelData <- getLearnerModel(naivemodel)
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(treeModelData, roundint = FALSE,
           box.palette = "BuBn",
           type =5)

#It wasted a lot of variables!

# 3) Tune the tree model with cross validation 
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

#see the parameters
tunedTreePars$x

#get the tuned model 
tunedTree <- setHyperPars(tree, par.vals = tunedTreePars$x)
tunedTreeModel <- train(tunedTree, PosterTask)

#plot the new tree
treeModelData <- getLearnerModel(tunedTreeModel)
rpart.plot(treeModelData, roundint = FALSE,
           box.palette = "BuBn",
           type =5)

#Download the pdf and see what have happened!

# 4) Cross validation of the tuned model 

#wrap the learner with the tuning process)
treeWrapper <- makeTuneWrapper(tree, resampling = CV,
                               par.set = treeParamSpace,
                               control = randSearch)

#Cross-validating the model-building process

outer <- makeResampleDesc("CV", iters = 4)

parallelStartSocket(cpus = detectCores())

cvWithTuning <- resample(treeWrapper, PosterTask, resampling = outer)

parallelStop()

cvWithTuning
#compare with the tuning accuracy rate it is largely similiar, meaning that the model is a good fit!

# 5) making out-of-sample predictions 

prediction <- as.tibble(predict(tunedTreeModel, newdata = test_data))
prediction <- prediction %>%
  mutate(correct = case_when(
    truth == response ~ 1,
    truth != response ~ 0
  ))
acc <- sum(prediction$correct)/nrow(prediction)
acc

