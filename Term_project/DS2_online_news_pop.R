##################################
##        Data science 2        ##
##                              ##
##     Kaggle competition       ##
##   ONLINE NEWS POPULARITIY    ##
##                              ##
##################################


# Clear memory
rm(list=ls())

# Load packages
library(tidyverse)
library(caret)
library(data.table)
library(GGally)
library(pROC)
library(kableExtra)
library(modelsummary)

my_path <- "c://Users/MViki/Documents/CEU/Winter_semester/DS_2/Term_project/Data/"


df <- read.csv(paste0(my_path, "train.csv"))

data_test <- read.csv(paste0(my_path, "test.csv"))




# EDA ---------------------------------------------------------------------

ggplot(df) +
  geom_histogram(aes(is_popular))

df %>% 
  group_by(is_popular) %>% 
  summarise("Number of companies" = n(), "Percentage" = paste0(round(n()/27752*100),'%')) %>% 
  kbl() %>% 
  kable_classic(full_width = F, html_font = "Cambria")

df %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~key, scales = "free") +
  geom_histogram(color = "cyan4")

ggcorr(df)

df[1,]



# Data cleaning -----------------------------------------------------------

df$is_popular <- factor(df$is_popular, levels = 0:1, labels = c("NotPopular", "Popular"))

  
# Create a training and validation data
train_indices <- as.integer(createDataPartition(df$is_popular, p = 0.8, list = FALSE))
data_train <- df[train_indices, ]
data_valid <- df[-train_indices, ]

dim(data_train)
dim(data_valid)

Hmisc::describe(df$is_popular)
Hmisc::describe(data_train$is_popular)
Hmisc::describe(data_valid$is_popular)

my_seed <- 2021



#####################################
##                                 ##
##      0. Predict No popular      ##
##                                 ##
#####################################

stupid_prediction <- tibble(article_id = data_test$article_id, score = 0)
write_csv(stupid_prediction, paste0(my_path,"Submissions/All_no_pop.csv"))





#####################################
##                                 ##
##       I. Linear models          ##
##                                 ##
#####################################

# 5 fold cross-validation ----------------------------------------------------------------------
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

#####################
##  Simple linear  ##
#####################

set.seed(my_seed)
simple_lm <- train(is_popular~. -article_id, 
                   method = "glm",
                   data = data_train,
                   family = binomial,
                   trControl = train_control)
simple_lm$resample

saveRDS(simple_lm, paste0(my_path, "Models/LM.rds"))

lmRoc <- roc(
  predictor=predict(simple_lm, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
lmRoc

plot(lmRoc)


####################
##   Elastic net  ##
####################

enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = 10^seq(-1, -7, length = 20)
)

set.seed(my_seed)

enet <- train(
  is_popular~. -article_id,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  family = "binomial",
  trControl = train_control,
  tuneGrid = enet_tune_grid
)

enet$resample

saveRDS(enet, paste0(my_path, "Models/Enet.rds"))

ggplot(enet)


## ROC
enetRoc <- roc(
  predictor=predict(enet, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
enetRoc

plot(enetRoc)

## Confusion matrix
confusionMatrix(data_valid$is_popular, predict(enet, data_valid, decision.values=T))


#############################
##   Elastic net with PCA  ##
#############################

trctrl_PCA <- trainControl(method = "cv", 
                             classProbs=TRUE, 
                             summaryFunction=twoClassSummary,
                             preProcOptions = list(thresh=0.9)
)
enet_w_PCA <- train(is_popular~. -article_id, 
                                  data = data_train, 
                                  method = "glmnet",
                                  preProcess=c("center", "scale", "pca"),
                                  trControl=trctrl_PCA,
                                  tuneLength=10,
                                  metric='ROC',
                                  tuneGrid = enet_tune_grid
)

saveRDS(enet_w_PCA, paste0(my_path, "Models/Enet_w_PCA.rds"))


# COMPARISON --------------------------------------------------------------

resample_profile <- resamples(
  list("linear" = simple_lm,
       "elastic net" = enet,
       "enet with PCA" = enet_w_PCA
  )
) 

summary(resample_profile)




# SAVE SUBMISSION ---------------------------------------------------------

enet_prediction <- tibble(article_id = data_test$article_id, score = predict(enet, data_test, type='prob', decision.values=T)$Popular)
write_csv(enet_prediction, paste0(my_path,"Submissions/Linear_model.csv"))


enet_prediction %>% 
  group_by(score) %>% 
  summarise(cnt = n())







#####################################
##                                 ##
##       II. Random forests        ##
##                                 ##
#####################################


# set tune grid
rf_grid <- expand.grid(
  .mtry = c(5, 7, 9),
  .splitrule = "gini",
  .min.node.size = c(5, 10, 15)
)

rf_grid2 <- expand.grid(
  .mtry = c(2, 3, 4, 5, 6, 7, 8, 9),
  .splitrule = "gini",
  .min.node.size = c(3, 5, 7, 10)
)


# build rf model 1
set.seed(my_seed)
rf_model <- train(
  is_popular~. -article_id,
  method = "ranger",
  metric = "ROC",
  data = data_train,
  tuneGrid = rf_grid,
  trControl = train_control,
  importance = "impurity"
)

rf_model
saveRDS(rf_model, paste0(my_path, "Models/RF_model_1.rds"))

# build rf model 2
set.seed(my_seed)
rf_model_2 <- train(
  is_popular~. -article_id,
  method = "ranger",
  metric = "ROC",
  data = data_train,
  tuneGrid = rf_grid2,
  trControl = train_control,
  importance = "impurity")

saveRDS(rf_model_2, paste0(my_path, "Models/RF_model_2.rds"))


## Autotune
set.seed(my_seed)
rf_model_auto <- train(
  is_popular~. -article_id,
  data = data_train,
  method = "ranger",
  metric = "ROC",
  trControl = train_control,
  importance = "impurity"
)

saveRDS(rf_model_auto, paste0(my_path, "Models/RF_model_auto.rds"))



# COMPARISONS -------------------------------------------------------------

my_models <- list( "rf_model", "rf_model_2", "rf_model_auto")

ROC <- c()

for (i in my_models) {
  ROC[i]  <- max(get(i)$results$ROC)
}

ROC %>% 
  kbl(col.names = "ROC") %>% 
  kable_classic(full_width = F, html_font = "Cambria")



# BEST MODEL - RF 2 -------------------------------------------------------

## ROC
rfRoc <- roc(
  predictor=predict(rf_model_2, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
rfRoc

plot(rfRoc)

## Confusion matrix
confusionMatrix(data_valid$is_popular, predict(rf_model_2, data_valid, decision.values=T))


# SAVE SUBMISSION ---------------------------------------------------------

rf_prediction <- tibble(article_id = data_test$article_id, score = predict(rf_model_2, data_test, type='prob', decision.values=T)$Popular)
write_csv(rf_prediction, paste0(my_path,"Submissions/RF_model.csv"))


rf_prediction %>% 
  group_by(score) %>% 
  summarise(cnt = n())




#####################################
##                                 ##
##   III. Gradient boosting        ##
##                                 ##
#####################################

## GBM model 1
gbm_grid <-  expand.grid(interaction.depth = c(1, 5, 10), 
                         n.trees = (4:10)*50,
                         shrinkage = 0.1,
                         n.minobsinnode = 20 
)


set.seed(my_seed)
gbm_model <- train(is_popular~. -article_id,
                   data = data_train,
                   method = "gbm",
                   metric = "ROC",
                   trControl = train_control,
                   verbose = FALSE,
                   tuneGrid = gbm_grid)

saveRDS(gbm_model, paste0(my_path, "Models/GBM_model_1.rds"))


## GBM model 2
gbm_grid2 <-  expand.grid(interaction.depth = c(1, 3, 5, 7, 9, 11), 
                          n.trees = (1:10)*50, 
                          shrinkage = c(0.02, 0.05, 0.1, 0.15, 0.2), 
                          n.minobsinnode = c(5,10,20,30))

set.seed(my_seed)
system.time({
  gbm_model_2 <- train(is_popular~. -article_id,
                       data = data_train,
                       method = "gbm",
                       metric = "ROC",
                       trControl = train_control,
                       verbose = FALSE,
                       tuneGrid = gbm_grid2)
})
saveRDS(gbm_model_2, paste0(my_path, "Models/GBM_model_2.rds"))

gbm_prediction <- tibble(article_id = data_test$article_id, score = predict(gbm_model_2, data_test, type='prob', decision.values=T)$Popular)
write_csv(gbm_prediction, paste0(my_path,"Submissions/GBM_model.csv"))



## XGBoost
xg_grid <-  expand.grid(
  nrounds=c(350),
  max_depth = c(2, 3, 4, 5),
  eta = c(0.03,0.05, 0.06),
  gamma = c(0.01),
  colsample_bytree = c(0.5),
  subsample = c(0.75),
  min_child_weight = c(0))

set.seed(my_seed)

xgb_model_2 <- train(
  is_popular~.,
  method = "xgbTree",
  metric = "ROC",
  data = data_train,
  tuneGrid = xg_grid,
  trControl = train_control
)

saveRDS(xgb_model_2, paste0(my_path, "Models/XGB_model_2.rds"))


# COMPARISONS -------------------------------------------------------------

my_models <- list( "gbm_model", "gbm_model_2", "xgb_model")

ROC <- c()

for (i in my_models) {
  ROC[i]  <- max(get(i)$results$ROC)
}

ROC %>% 
  kbl(col.names = "ROC") %>% 
  kable_classic(full_width = F, html_font = "Cambria")



# BEST MODEL - GBM -------------------------------------------------------

## ROC
gbmRoc <- roc(
  predictor=predict(xgb_model, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
gbmRoc

plot(gbmRoc)

## Confusion matrix
confusionMatrix(data_valid$is_popular, predict(xgb_model, data_valid, decision.values=T))


# SAVE SUBMISSION ---------------------------------------------------------

gbm_prediction <- tibble(article_id = data_test$article_id, score = predict(xgb_model_2, data_test, type='prob', decision.values=T)$Popular)
write_csv(gbm_prediction, paste0(my_path,"Submissions/XGBoosting_model_2.csv"))





#####################################
##                                 ##
##       IV. Neural network        ##
##                                 ##
#####################################

tune_grid <- expand.grid(
  size = c(3, 5, 7, 10, 15),
  decay = c(0.1, 0.5, 1, 1.5, 2, 2.5, 5)
)

set.seed(my_seed)
nnet_model <- train(
  is_popular~. -article_id,
  method = "nnet",
  data = data_train,
  trControl = train_control,
  tuneGrid = tune_grid,
  preProcess = c("center", "scale", "pca"),
  metric = "ROC",
  trace = FALSE
)
nnet_model
nnet_model$finalModel
confusionMatrix(nnet_model)

saveRDS(nnet_model, paste0(my_path, "Models/NNet_model.rds"))


## Net with different starting point
tune_grid <- expand.grid(size = 5, decay = 1, bag = FALSE)

set.seed(my_seed)
avnnet_model <- train(
  is_popular~. -article_id,
  method = "avNNet",
  data = data_train,
  repeats = 5,
  trControl = train_control,
  tuneGrid = tune_grid,
  preProcess = c("center", "scale", "pca"),
  metric = "ROC",
  trace = FALSE
)

saveRDS(avnnet_model, paste0(my_path, "Models/AVNNet_model.rds"))



# COMPARISONS -------------------------------------------------------------

my_models <- list( "nnet_model", "avnnet_model")

ROC <- c()

for (i in my_models) {
  ROC[i]  <- max(get(i)$results$ROC)
}

ROC %>% 
  kbl(col.names = "ROC") %>% 
  kable_classic(full_width = F, html_font = "Cambria")


# BEST MODEL - Neural net  -------------------------------------------------------

## ROC
netRoc <- roc(
  predictor=predict(avnnet_model, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
netRoc

plot(netRoc)

## Confusion matrix
confusionMatrix(data_valid$is_popular, predict(avnnet_model, data_valid, decision.values=T))


# SAVE SUBMISSION ---------------------------------------------------------

net_prediction <- tibble(article_id = data_test$article_id, score = predict(avnnet_model, data_test, type='prob', decision.values=T)$Popular)
write_csv(net_prediction, paste0(my_path,"Submissions/Neural_net_model.csv"))




# USING H2O ---------------------------------------------------------------

library(h2o)
h2o.init()
h2o.no_progress()


data_train_h2o <- as.h2o(data_train)
data_valid_h2o <- as.h2o(data_valid)
data_test_h2o <- as.h2o(data_test)

y <- "is_popular"
X <- setdiff(names(data_train), c(y, "article_id"))


dl_model <- h2o.deeplearning(
  x = X, y = y,
  training_frame = data_train_h2o,
  validation_frame = data_valid_h2o,
  model_id = "DL_default",
  score_each_iteration =  TRUE,  # for the purpose of illustration
  seed = my_seed
)
dl_model


deep_large_model <- h2o.deeplearning(
  x = X, y = y,
  model_id = "DL_deep_large",
  training_frame = data_train_h2o,
  validation_frame = data_valid_h2o,
  hidden = c(128, 128, 128),
  seed = my_seed
)

dl_model_high_batch <- h2o.deeplearning(
  x = X, y = y,
  model_id = "DL_model_high_batch",
  training_frame = data_train_h2o,
  validation_frame = data_valid_h2o,
  hidden = c(512),
  mini_batch_size = 10,
  seed = my_seed
)

dropout_model <- h2o.deeplearning(
  x = X, y = y,
  model_id = "DL_dropout",
  training_frame = data_train_h2o,
  validation_frame = data_valid_h2o,
  hidden = c(512),
  activation = "RectifierWithDropout",
  hidden_dropout_ratios = 0.6,
  seed = my_seed
)

regularized_model <- h2o.deeplearning(
  x = X, y = y,
  model_id = "DL_regularized",
  training_frame = data_train_h2o,
  validation_frame = data_valid_h2o,
  hidden = 512,
  l1 = 0.01,
  l2 = 0.01,
  seed = my_seed
)


compareAUC <- function(list_of_models) {
  map_df(
    list_of_models,
    ~tibble(model = .@model_id, auc = h2o.auc(h2o.performance(., newdata = data_valid_h2o)))
  ) %>%
    arrange(-auc)
}


compareAUC(list(dl_model, deep_large_model, dl_model_high_batch,  dropout_model, regularized_model))

h2o_net_prediction <- h2o.predict(dl_model, data_test_h2o, type = "prob")
h2o_net_prediction <- as.data.frame(h2o_net_prediction)

# SAVE SUBMISSION ---------------------------------------------------------

h2o_net_pred <- tibble(article_id = data_test$article_id, score = h2o_net_prediction$Popular)
write_csv(h2o_net_pred, paste0(my_path,"Submissions/H2O_net_model.csv"))




#####################################
##                                 ##
##        IV. Extra models         ##
##                                 ##
#####################################


#############################
##   Nearest neighbours    ##
#############################

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

set.seed(my_seed)
knnModel <- train(is_popular~. -article_id, data = data_train, method = "knn",
                  trControl=trctrl,
                  preProcess=c("center", "scale"),
                  tuneLength=20,
                  tuneGrid = data.frame(k=c(2:8))
)
knnModel

## ROC
knnRoc <- roc(
  predictor=predict(knnModel, data_valid, type='prob', decision.values=T)$Popular, 
  response=data_valid$is_popular)
knnRoc

plot(knnRoc)

## Confusion matrix
confusionMatrix(data_valid$is_popular, predict(knnModel, data_valid, decision.values=T))


# SAVE SUBMISSION ---------------------------------------------------------

knn_prediction <- tibble(article_id = data_test$article_id, score = predict(knnModel, data_test, type='prob', decision.values=T)$Popular)
write_csv(knn_prediction, paste0(my_path,"Submissions/KNN_model.csv"))



#####################################
##                                 ##
##        V. Stacked model         ##
##                                 ##
#####################################


glm_model_s <- h2o.glm(
  X, y,
  training_frame = data_train_h2o,
  model_id = "lasso",
  family = "binomial",
  alpha = 1,
  lambda_search = TRUE,
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE  # this is necessary to perform later stacking
)

rf_model_s <- h2o.randomForest(
  X, y,
  training_frame = data_train_h2o,
  model_id = "rf",
  ntrees = 200,
  max_depth = 10,
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

gbm_model_s <- h2o.gbm(
  X, y,
  training_frame = data_train_h2o,
  model_id = "gbm",
  ntrees = 200,
  max_depth = 5,
  learn_rate = 0.1,
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

deeplearning_model_s <- h2o.deeplearning(
  X, y,
  training_frame = data_train_h2o,
  model_id = "deeplearning",
  hidden = c(32, 8),
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)



# PERFORMANCE COMPARISONS -------------------------------------------------
getPerformanceMetrics <- function(model, newdata = NULL, xval = FALSE) {
  h2o.performance(model, newdata = newdata, xval = xval)@metrics$thresholds_and_metric_scores %>%
    as_tibble() %>%
    mutate(model = model@model_id)
}

plotROC <- function(performance_df) {
  ggplot(performance_df, aes(fpr, tpr, color = model)) +
    geom_path() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    coord_fixed() +
    labs(x = "False Positive Rate", y = "True Positive Rate")
}

my_models <- list(glm_model_s, rf_model_s, gbm_model_s, deeplearning_model_s)
all_performance <- map_df(c(my_models), getPerformanceMetrics, xval = TRUE)
plotROC(all_performance)

compareAUC(my_models)



# STACK BASE LEARNERS -----------------------------------------------------

ensemble_model <- h2o.stackedEnsemble(
  X, y,
  training_frame = data_train_h2o,
  base_models = my_models,
  keep_levelone_frame = TRUE
)
ensemble_model
ensemble_model@model$metalearner_model@model$coefficients_table

saveRDS(ensemble_model, paste0(my_path, "Models/Ensemble_model.rds"))

## ROC
ensembleRoc <- roc(
  predictor=as.data.frame(predict(ensemble_model, data_valid_h2o, type='prob', decision.values=T))$Popular, 
  response=data_valid$is_popular)
ensembleRoc

plot(ensembleRoc)



# SAVE SUBMISSION ---------------------------------------------------------

ensemble_prediction <- tibble(article_id = data_test$article_id, score = as.data.frame(predict(ensemble_model, data_test_h2o, type='prob', decision.values=T))$Popular)
write_csv(ensemble_prediction, paste0(my_path,"Submissions/Ensemble_model.csv"))



#####################################
##                                 ##
##          VI. Analysis           ##
##                                 ##
#####################################



plot(varImp(rf_model_2))

plot(varImp(gbm_model_2))

plot(varImp(xgb_model))









