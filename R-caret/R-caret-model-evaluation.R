 library(psych) 
 library(corrplot)
 library(caret)
 library(doParallel)
 library(plyr)
 library(dplyr)
 library(ggplot2)
 library(pROC)

#caret runs CV in parallel
detectCores()
cl <- makeCluster(4)
registerDoParallel(cl)
getDoParWorkers()

#load data
raw_train <- read.csv(file='//...../train.csv')

#column stats
summary(raw_train)
meta_data <- describe(raw_train, skew = TRUE, type = 1 , IQR = TRUE, fast = FALSE)

#check of values/cardinality
raw_train %>% select(contains('cat')) %>% apply(2,unique)
raw_train %>% select(contains('cat')) %>% apply(2,n_distinct)
raw_train %>% select(contains('bin')) %>% apply(2,unique)
raw_train %>% select(contains('bin')) %>% apply(2,n_distinct)


raw_train %>% select(contains('cat')) %>% apply(2,function(x) sum(x==-1))  #check for missing value, potential impute

  
#graphs to visualise  data
hist(raw_train$ps_ind_05_cat)
barchart(data.frame(ifelse(raw_train[,2] ==1,"Claimers", "Non-Claimers")), horizontal=F, col="blue" , main="Distribution of claimers and non-claimers
")


plot_histogram <- function(table, col_to_plot){
  
  g <- ggplot(table, aes(col_to_plot))
  g + geom_histogram(#aes(fill=table$target), 
                     binwidth = .1, 
                     col="blue", 
                     size=.1) +  # change binwidth
    labs(title="Histogram")   + facet_grid(.~table$target) 
}
plot_histogram(raw_train, raw_train$ps_ind_05_cat)



#Histograms to visualise the data
sapply (names(raw_train)[which( grepl('_bin$', names(raw_train)))], function(x)  hist(raw_train[,x]))
sapply (names(raw_train)[which( grepl('_cat$', names(raw_train)))], function(x)  hist(raw_train[,x]))
sapply (names(raw_train)[-which( grepl('_cat$', names(raw_train)) | grepl('_bin$', names(raw_train)) )], function(x)  hist(raw_train[,x], main = x))


#Correlation test/graph
corr.test(raw_train[2:59])
corrplot(cor(raw_train[2:59]),  order="hclust")



#helper variables
features_non_cat_bin <- names(raw_train)[-which( grepl('_cat$', names(raw_train)) | grepl('_bin$', names(raw_train)) )]
features_cat <- names(raw_train)[which( grepl('_cat$', names(raw_train)))]
features_bin <- names(raw_train)[which( grepl('_bin$', names(raw_train)))]
features_calc <- names(raw_train)[which( grepl('_calc_', names(raw_train)))]


#Stratified subset of data due to limited processing power
set.seed(100)
rows_partition <- createDataPartition(raw_train$target, p = .022)[[1]]
raw_train_subset <- raw_train[rows_partition,]

#make categorical values factors for dummy variables/one hot encoding
for(cnt in setdiff(features_cat, 'ps_car_11_cat')){
    raw_train_subset[, cnt] <- factor(raw_train_subset[, cnt] )
}


dummyVars_1 <- dummyVars(~.,              #create dummy variables                                     
                         data = raw_train_subset ,
                         levelsOnly = FALSE)

dummyVars_2 <-as.data.frame(predict( dummyVars_1,raw_train_subset))
tmp_columns_to_remove <- nearZeroVar(dummyVars_2, freqCut = 99.5/0.5)  #remove low variance data
dummyVars_3 <- dummyVars_2[, -tmp_columns_to_remove]



set.seed(100)   #split into train/test
rows_partition_2 <- createDataPartition(raw_train_subset$target, p = .8)[[1]]


train_set <- dummyVars_3[rows_partition_2, 3:ncol(dummyVars_3) ]

test_set <- dummyVars_3[-rows_partition_2, 3:ncol(dummyVars_3) ]

#Vectors with target variable
output_value <- ifelse(dummyVars_3[rows_partition_2,2] ==1,"Yes", "No")
output_value_test <- ifelse(dummyVars_3[-rows_partition_2,2] ==1,"Yes", "No")


#Cross-validation control, the same folds used for all models, number of folds should be tuned - kept low due processing time
set.seed(100)    
indx <- createFolds(output_value, returnTrain = TRUE,k=3 )
ctrl <- trainControl(method = "cv", index = indx, savePredictions = "final", summaryFunction = twoClassSummary,
                     classProbs = TRUE,  returnResamp = "final", verboseIter = TRUE)



############### FULL TUNING GRIDS - USED INITIALLY 

# mtryGrid <- data.frame(mtry = 3:15)
# 
# plsGrid <- expand.grid(ncomp = 3:25)
# 
# 
# glmnGrid <- expand.grid(alpha = seq(0, 1, length = 10),
#                         lambda = seq(0, 1, length = 10))
# 
# xgbGrid <- expand.grid(nrounds = c( 100,  300, 500),
#                        max_depth = c( 2, 4, 6),
#                        eta = c(.1,  .02, .03, .07),
#                        gamma = c(0 , 0.5, 1),
#                        colsample_bytree = c( .5, .7),
#                        min_child_weight = c(1,3,5),
#                        subsample = c(.8, .3, .5))



############### NARROWED GRIDS - USED AFTER REASONABLE PARAMETERS WERE FOUND

mtryGrid <- data.frame(mtry = 7:8)

plsGrid <- expand.grid(ncomp = 3:25)


glmnGrid <- expand.grid(alpha = seq(0, 1, length = 10),
                        lambda = seq(0, 1, length = 10))


xgbGrid <- expand.grid(nrounds = c( 100),
                       max_depth = c( 2, 4, 6),
                       eta = c( .01 ,.03, .07),
                       gamma = c(0 ),
                       colsample_bytree = c( .7, .8 , .9),
                       min_child_weight = c(5),
                       subsample = c( .3))



############ Models configuration

MODEL_CONFIG_RF <- list(method = "rf",
                        tuneGrid = mtryGrid,
                        ntree = 500)


MODEL_CONFIG_PLS <- list(method = "pls",
                         tuneGrid = plsGrid)


MODEL_CONFIG_glmnet <- list(method = "glmnet",
                            tuneGrid = glmnGrid,
                            preProc = c("center", "scale"))


MODEL_CONFIG_xgbTree <- list(method = "xgbTree",
                             tuneGrid = xgbGrid,
                             preProc = c("center", "scale"))



MODEL_CONFIG_LIST_OF_MODELS <- list(
  RF = MODEL_CONFIG_RF,
  PLS = MODEL_CONFIG_PLS,
  glmnet = MODEL_CONFIG_glmnet,
  xgbTree = MODEL_CONFIG_xgbTree
)

list_store_models <-list() #List to store models



#Loop to tune all models
for(cnt in 1:length(MODEL_CONFIG_LIST_OF_MODELS)){
  
  set.seed(100)
  modelTune <- do.call (train , c(list(x = train_set, y = output_value),
                                  (MODEL_CONFIG_LIST_OF_MODELS[[cnt]]),
                                  list(trControl = ctrl, metric = "ROC")))
  
  list_store_models[[ (names(MODEL_CONFIG_LIST_OF_MODELS)[cnt])]] <- modelTune

}



#Plots of predictors' importance
plot(varImp(list_store_models$RF, scale = FALSE), top = 25)

plot(varImp(list_store_models$xgbTree, scale = FALSE), top = 25)

plot(varImp(list_store_models$glmnet, scale = FALSE), top = 25)

plot(varImp(list_store_models$PLS, scale = FALSE), top = 25)




#calculate ROC/AUC on train set
list_train_set_predictions_ROC <- lapply(list_store_models, function(x) roc(x$pred$obs, x$pred$Yes,
                                                                            levels = c("No" , "Yes")  ))


#predict on the test set
list_test_set_predictions <- lapply(list_store_models, function(x) predict(x, test_set,  type="prob"  ))

#calculate ROC/AUC on test set
list_test_set_predictions_ROC <- lapply(list_test_set_predictions, function(x) roc(output_value_test, x[,1],
                                                                                   levels = c("No" , "Yes")  ))


#Plot ROC curves
plot(list_train_set_predictions_ROC$glmnet, legacy.axes = TRUE)
plot(list_train_set_predictions_ROC$xgbTree, legacy.axes = TRUE)
plot(list_train_set_predictions_ROC$RF, legacy.axes = TRUE)
plot(list_train_set_predictions_ROC$PLS, legacy.axes = TRUE)



#List to manage feature engineering/removal
features_list <- list()

features_list [["Feature_00_all"]]  <-  list(( ( names(raw_train) ) ) , "Feature_00_all")

features_list [["Feature_01"]]  <-  list(c( setdiff( names(raw_train) ,features_calc) ) , "Feature_01")

features_list [["Feature_02"]]  <-  list(c( setdiff( names(raw_train) , features_bin )) , "Feature_02")





#Function to adjust train/test set to different sets of features
func_features_selection <- function(tmp_list){
  

  tmp_data_set <- raw_train_subset[, tmp_list]
  dummyVars_1 <- dummyVars(~.,              #create dummy variables                                     
                           data = tmp_data_set ,
                           levelsOnly = FALSE)
  
  dummyVars_2 <-as.data.frame(predict( dummyVars_1,tmp_data_set))
  tmp_columns_to_remove <- nearZeroVar(dummyVars_2, freqCut = 99.5/0.5)  #remove low variance data
  dummyVars_3 <- dummyVars_2[, -tmp_columns_to_remove]
  
  
  
  set.seed(100)   #split into train/test
  rows_partition_2 <- createDataPartition(tmp_data_set$target, p = .8)[[1]]
  
  
  train_set <<- dummyVars_3[rows_partition_2, 3:ncol(dummyVars_3) ]
  
  test_set <<- dummyVars_3[-rows_partition_2, 3:ncol(dummyVars_3) ]
  
  output_value <<- ifelse(dummyVars_3[rows_partition_2,2] ==1,"Yes", "No")

  output_value_test <<- ifelse(dummyVars_3[-rows_partition_2,2] ==1,"Yes", "No")
  
}


#Lists to store results
list_store_models_features_sel <-list() ; list_train_set_predictions_ROC_features <-list(); list_test_set_predictions_ROC_features <- list()


#Tune model over different sets of features
for(cnt in 1:length(features_list)){

      func_features_selection (unlist(features_list[[cnt]][1]))

      set.seed(100)
      modelTune2 <- do.call (train , c(list(x = train_set, y = output_value),
                                       (MODEL_CONFIG_LIST_OF_MODELS[['glmnet']])  ,  #needs to be changed to PLS for pls results
                                       list(trControl = ctrl, metric = "ROC")))
      
      list_store_models_features_sel[[ (names(features_list)[cnt])]] <- modelTune2   
      list_train_set_predictions_ROC_features[[(names(features_list)[cnt]) ]] <- roc(modelTune2$pred$obs, modelTune2$pred$Yes,
                                                                                     levels = c("No" , "Yes")  )
      list_test_set_predictions_ROC_features[[  names(features_list)[cnt] ]]  <-   roc( output_value_test, ((predict(modelTune2, test_set,  type="prob"  ))[,1]),
                                                                                        levels = c("No" , "Yes")  )
}





#reset the train/test sets
func_features_selection (unlist(features_list[["Feature_00_all"]][1]))

#downsample train set
set.seed(100) 
downSampDf <- downSample(x = train_set, y = as.factor(output_value), list = FALSE )
output_value <- ifelse(downSampDf[,"Class"] =="Yes","Yes", "No")
train_set <- downSampDf[,-which (names(downSampDf)=="Class")] 
set.seed(100)    
indx <- createFolds(output_value, returnTrain = TRUE,k=3 )
ctrl <- trainControl(method = "cv", index = indx, savePredictions = "final", summaryFunction = twoClassSummary,
                     classProbs = TRUE,  returnResamp = "final", verboseIter = TRUE)


#tune models
set.seed(100)
modelTune3 <- do.call (train , c(list(x = train_set, y = output_value),
                                 (MODEL_CONFIG_LIST_OF_MODELS[['glmnet']])  ,
                                 list(trControl = ctrl, metric = "ROC")))

roc(modelTune3$pred$obs, modelTune3$pred$Yes,
    levels = c("No" , "Yes")  )
roc( output_value_test, ((predict(modelTune3, test_set,  type="prob"  ))[,1]),
     levels = c("No" , "Yes")  )

set.seed(100)
modelTune4 <- do.call (train , c(list(x = train_set, y = output_value),
                                 (MODEL_CONFIG_LIST_OF_MODELS[['PLS']])  ,
                                 list(trControl = ctrl, metric = "ROC")))

roc(modelTune4$pred$obs, modelTune4$pred$Yes,
    levels = c("No" , "Yes")  )
roc( output_value_test, ((predict(modelTune4, test_set,  type="prob"  ))[,1]),
     levels = c("No" , "Yes")  )



#reset the train/test sets
func_features_selection (unlist(features_list[["Feature_00_all"]][1]))

#upsample train set
set.seed(100) 
upSampDf <- upSample(x = train_set, y = as.factor(output_value), list = FALSE )
output_value <- ifelse(upSampDf[,"Class"] =="Yes","Yes", "No")
train_set <- upSampDf[,-which (names(upSampDf)=="Class")] 
set.seed(100)    
indx <- createFolds(output_value, returnTrain = TRUE,k=3 )
ctrl <- trainControl(method = "cv", index = indx, savePredictions = "final", summaryFunction = twoClassSummary,
                     classProbs = TRUE,  returnResamp = "final", verboseIter = TRUE)



#tune models
set.seed(100)
modelTune5 <- do.call (train , c(list(x = train_set, y = output_value),
                                 (MODEL_CONFIG_LIST_OF_MODELS[['glmnet']])  ,
                                 list(trControl = ctrl, metric = "ROC")))

roc(modelTune5$pred$obs, modelTune5$pred$Yes,
    levels = c("No" , "Yes")  )
roc( output_value_test, ((predict(modelTune5, test_set,  type="prob"  ))[,1]),
     levels = c("No" , "Yes")  )

set.seed(100)
modelTune6 <- do.call (train , c(list(x = train_set, y = output_value),
                                 (MODEL_CONFIG_LIST_OF_MODELS[['PLS']])  ,
                                 list(trControl = ctrl, metric = "ROC")))

roc(modelTune6$pred$obs, modelTune6$pred$Yes,
    levels = c("No" , "Yes")  )
roc( output_value_test, ((predict(modelTune6, test_set,  type="prob"  ))[,1]),
     levels = c("No" , "Yes")  )






