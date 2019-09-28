# Categorical embedding demo script
# Why-R conference
# Warszawa, 28.09.2019
# Tamas Burgard

# We make embeddings for 3 categorical vars:
# uniqueCarrier : 22  levels
# Origin        : 291 levels
# Dest          : 292 levels

# All of them will be transformed to 10-10 dimensions

library(data.table)
library(recipes)
library(embed)
library(xgboost)
library(purrr)
library(ROCR)


# Dataset from the GBM performance benchmark (Szilard Pafka)
# visit: https://github.com/szilard/GBM-perf
# get these files:
# https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv # faster
# https://s3.amazonaws.com/benchm-ml--main/train-1m.csv   
# https://s3.amazonaws.com/benchm-ml--main/test.csv


pth <- "WHERE YOU DOWNLOADED THE DATA PATH"

data <- fread(paste0(pth, 'train-1m.csv')) # choose 0.1m for first try 
test <- fread(paste0(pth, 'test.csv'))

my_recipe <- data %>%
  recipe(dep_delayed_15min ~ .) %>%    # dep_delayed_15min is our outcome
  step_center(Distance, DepTime) %>%   # deep learning is better with normalized input
  step_scale( Distance, DepTime) %>%
  step_string2factor(-Distance, -DepTime, -dep_delayed_15min) %>% 
  step_integer(-Distance, -DepTime, -dep_delayed_15min, 
               -Origin,   -Dest,    -UniqueCarrier) %>% # the last 3 will be encoded
  step_dummy(dep_delayed_15min) %>%
  
  step_embed(Origin, Dest, UniqueCarrier,               # the 3 vars to encode
             outcome    = vars(dep_delayed_15min_Y),    # outcome for THIS step
             predictors = vars(Distance, DepTime),      # 2 numeric var to help
             num_terms  = 10L,                          # embedding dimension
             hidden_units = 10L,                        # I picked a random number
             options    = embed_control(verbose = 1, validation_split = .2, 
                                        batch_size = 64, epochs = 8)
             ) %>%
  
  prep(training = data)

d_transformed <- bake(my_recipe, new_data = data)
d_tr_test     <- bake(my_recipe, new_data = test)

D_data <- xgb.DMatrix(data  = d_transformed %>% select(-dep_delayed_15min_Y) %>% as.matrix(),
                      label = d_transformed$dep_delayed_15min_Y)

D_test <- xgb.DMatrix(data  = d_tr_test %>% select(-dep_delayed_15min_Y) %>% as.matrix(),
                      label = d_tr_test$dep_delayed_15min_Y)

watch_list  <- list(data = D_data) 

# these params were selected by Szilard Pafka for the comparison
params_list <- list(eta = 0.1,
                    objective = "binary:logistic",
                    eval_metric = "auc", 
                    max_depth = 10L, 
                    nthread = 4L) # change this according to your computer

bst <- xgb.train(params = params_list, data = D_data, nrounds = 100L,
                 print_every_n = 10L, watchlist = watch_list)

phat      <- predict(bst, newdata = D_test)
rocr_pred <- prediction(phat, test$dep_delayed_15min)
performance(rocr_pred, "auc")

