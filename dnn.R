# libraries
library(keras)
library(ggplot2)
library(reshape2)
library(viridis)
library(dplyr)
library(tfruns)
library(gridExtra)
library(jsonlite)


# load helper functions
source('helpers.R')

# load data
load('data_activity_recognition.RData')

# check the dimensions
dim(x_test)
dim(x_train)

# reshape the matrix into vectors - each signal segment is represented by a 
# vector of 125*45 measurements
x_train <- array_reshape(x_train, c(nrow(x_train), 125 * 45)) 
x_test <- array_reshape(x_test, c(nrow(x_test), 125 * 45))

#############################################################################
### Visualisation of the classes ############################################
#############################################################################

# reduce the dimension of the data with PCA
x_test_pca <- prcomp(x_test)

# compute cumulative proportion of variance 
prop <- cumsum(x_test_pca$sdev^2) / sum(x_test_pca$sdev^2) 

# proportion of variance explained by the first two and ten components
prop[2]

# plot the first two principal components (around 31 % of the variance)
ggplot(as.data.frame(x_test_pca$x[, c(1:2)]), aes(x = x_test_pca$x[, 1], 
                                                  y = x_test_pca$x[, 2])) +
  geom_point(aes(colour = y_test)) +
  theme_pv() +
  scale_colour_viridis_d(option = 'C', name = 'Type of activity:') +
  labs(title = 'Sport activities reduced to \ntwo dimensions', 
       y = "PC2", x = "PC1") +
  theme(legend.position = 'right', 
        legend.text = element_text(size = 12)) 

#####

######## Data preparation ####################################################

# see the range of the value
range(x_train)

# convert classes from character to numeric value
y_test <- as.factor(y_test)
y_test <- as.numeric(y_test)
y_test <- y_test - 1

y_train <- as.factor(y_train)
y_train <- as.numeric(y_train)
y_train <- y_train - 1

# one-hot encoding
y_train <- to_categorical(y_train) 
y_test <- to_categorical(y_test)

######## Validation-test split ###############################################

# split the test data in two halves: one for validation
# and the other for actual testing
set.seed(12)

val <- sample(1:nrow(x_test), floor(nrow(x_test) * 0.5)) 
test <- setdiff(1:nrow(x_test), val) 

x_val <- x_test[val, ]
y_val <- y_test[val, ]

x_test <- x_test[test, ]
y_test <- y_test[test, ]

V <- ncol(x_train)
N <- nrow(x_train)

# check if the classes are distributed equaly in the test and val data
mean(colSums(y_val))
mean(colSums(y_test))

sd(colSums(y_val))
sd(colSums(y_test))

###############################################################################
################ Selecting data preprocessing methods #########################
###############################################################################

# normalisation
normalise <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

x_train_n <- apply(x_train, 2, normalise)
x_val_n <- apply(x_val, 2, normalise)

# standardisation
x_train_s <- scale(x_train)
x_val_s <- scale(x_val)



# dnn to select the data preprocessing method
model <- keras_model_sequential() %>%
  layer_dense(units = 512, input_shape = V, activation = "relu", name = "layer_1") %>%
  layer_dense(units = 128, activation = "relu", name = "layer_2") %>%
  layer_dense(units = ncol(y_train), activation = "softmax", name = "layer_out") %>%
  
  compile(
    loss = "categorical_crossentropy", 
    metrics = "accuracy",
    optimizer = optimizer_adam()
  )

# put the three data in a list
dprep_test <- list(norm = list(x_train_n, x_val_n),
                   identity = list(x_train, x_val), 
                   stand = list(x_train_s, x_val_s))

# initialise list to store the learning curve data
fits <- list()


# run three models
for(i in 1:3){
  fit <- model %>% fit(
    x = dprep_test[[i]][1], y = y_train,
    validation_data = list(dprep_test[[i]][2], y_val),
    epochs = 100,
    verbose = 1)
  fits[[i]] <- fit
}


###### Plotting #############

coln <- c('Identity_train', 'Identity_val',
          'Normalize_train', 'normalise_val',
          'Standardize_train', 'Standardize_val')

# see the helpers.R file for details for data_pt
plot_learn_c <- data_pt(fits, coln = coln)

# colour palette
col <- rep(viridis(3, option = 'C', end = 0.8), each = 2) 

# plot train and validation learning curves
ggplot(plot_learn_c, aes(x = id, y = value, group = variable, 
                         colour = variable, linetype = variable)) +
  stat_smooth(data = plot_learn_c, method = 'loess', geom = 'line',
              se = FALSE, size = 1) + #, linetype = "dashed") +
  #stat_smooth(data = plot_learn_c_v, method = 'loess', geom = 'line', 
              #se = FALSE, size = 1) +
  scale_y_log10(limits = c(0.8, 1), breaks = seq(0.80, 1, by = 0.04)) +
  theme_pv() +
  scale_colour_manual(values = col) +
  scale_linetype_manual(values = c(2, 1, 2, 1, 2, 1)) +
  geom_point(alpha = 0.3) +
  labs(title = 'Learning curves for three different \ndata preprocessing methods', 
       y = "Accuracy", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.title = element_blank(),
        legend.text = element_text(size = 12))

# we go with standardising

######################################################################
################ Selecting number of layers ##########################
######################################################################

# Model with 2 - 4 - 6 layers, each with decreasing number of units

# 3 dnn with different number of layers
model_2 <- keras_model_sequential() %>%
  layer_dense(units = 512, input_shape = V, activation = "relu", name = "layer_1") %>%
  layer_dense(units = 256, activation = "relu", name = "layer_2") %>%
  layer_dense(units = ncol(y_train), activation = "softmax", name = "layer_out") %>%
  
  compile(
    loss = "categorical_crossentropy", 
    metrics = "accuracy",
    optimizer = optimizer_adam()
  )

model_4 <- keras_model_sequential() %>%
  layer_dense(units = 512, input_shape = V, activation = "relu", name = "layer_1") %>%
  layer_dense(units = 256, activation = "relu", name = "layer_2") %>%
  layer_dense(units = 128, activation = "relu", name = "layer_3") %>%
  layer_dense(units = 64, activation = "relu", name = "layer_4") %>%
  layer_dense(units = ncol(y_train), activation = "softmax", name = "layer_out") %>%
  
  compile(
    loss = "categorical_crossentropy", 
    metrics = "accuracy",
    optimizer = optimizer_adam()
  )

model_6 <- keras_model_sequential() %>%
  layer_dense(units = 512, input_shape = V, activation = "relu", name = "layer_1") %>%
  layer_dense(units = 256, activation = "relu", name = "layer_2") %>%
  layer_dense(units = 192, activation = "relu", name = "layer_3") %>%
  layer_dense(units = 128, activation = "relu", name = "layer_4") %>%
  layer_dense(units = 64, activation = "relu", name = "layer_5") %>%
  layer_dense(units = 32, activation = "relu", name = "layer_6") %>%
  layer_dense(units = ncol(y_train), activation = "softmax", name = "layer_out") %>%
  
  compile(
    loss = "categorical_crossentropy", 
    metrics = "accuracy",
    optimizer = optimizer_adam()
  )


# initialise list to store the learning curve data
fits_a <- list()

# run three models

# 2 layrers
fit <- model_2 %>% fit(
  x = x_train_s, y = y_train,
  validation_data = list(x_val_s, y_val),
  epochs = 100,
  verbose = 1)
fits_a[[1]] <- fit
  
# 4 layrers
fit <- model_4 %>% fit(
  x = x_train_s, y = y_train,
  validation_data = list(x_val_s, y_val),
  epochs = 100,
  verbose = 1)
fits_a[[2]] <- fit

# 4 layers
fit <- model_6 %>% fit(
  x = x_train_s, y = y_train,
  validation_data = list(x_val_s, y_val),
  epochs = 100,
  verbose = 1)
fits_a[[3]] <- fit


###### Plotting #############

coln_a <- c('2_layers_train', '2_layers_val',
          '4_layers_train', '4_layers_val',
          '6_layers_train', '6_layers_val')

# see the helpers.R file for details for data_pt
plot_learn_c_a <- data_pt(fits_a, coln = coln_a) 

# colour palette
col <- rep(viridis(3, option = 'C', end = 0.8), each = 2) 

# plot train and validation learning curves
ggplot(plot_learn_c_a, aes(x = id, y = value, group = variable, 
                         colour = variable, linetype = variable)) +
  stat_smooth(data = plot_learn_c_a, method = 'loess', geom = 'line', 
              se = FALSE, size = 1) + 
  scale_y_log10(limits = c(0.93, 1), breaks = seq(0.92, 1, by = 0.02)) +
  theme_pv() +
  scale_colour_manual(values = col) +
  scale_linetype_manual(values = c(2, 1, 2, 1, 2, 1)) +
  geom_point(alpha = 0.3) +
  labs(title = 'Learning curves for models with three \ndifferent number of layers', 
       y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.title = element_blank(),
        legend.text = element_text(size = 12))

# we go with 2 layers

######################################################################
################ Hyperparameter tuning - rough tuning ################
######################################################################

# run #########
dropout_set <- c(0, 0.2, 0.4)
units_1_set <- c(512, 256)
units_2_set <- c(256, 128, 64)
lambda_set <- c(0, exp( seq(-9, -4, length = 3) )) 
bs_set <- floor(c(0.003, 0.01, 0.02) * N)
lr_set <- c(0.001, 0.005, 0.01)
patience_set <- c(10, 20)

runs <- tuning_run("model_conf.R",
                   runs_dir = "runs_3",
                   flags = list(
                     dropout = dropout_set,
                     units_1 = units_1_set,
                     units_2 = units_2_set,
                     lambda = lambda_set,
                     bs = bs_set,
                     lr = lr_set,
                     patience = patience_set),
                   sample = 0.03)


# get the worst models and their parameters
worst <- ls_runs(runs_dir = "runs_3", order = metric_val_accuracy)

# select only the relevant paramenters
worst <- s[, c(2, 4, 8:14, 18)] 
worst[order(worst$eval_accuracy)[1:10], ] 


########### extract results ############ 

run_3 <- read_metrics("runs_3")

# extract validation accuracy and plot learning curve
acc_3 <- as.data.frame(sapply(run_3, "[[", "val_accuracy"))

# extract the parameters values for each run
param_3 <- as.data.frame(sapply(run_3, "[[", "flags"))
param_3 <- apply(param_3, 2, unlist)

# add id variable for easy melting
acc_3$id <- c(1:100)

# convert to long format for easy plotting
acc_3 <- melt(acc_3, id.var = 'id')

# extact column names for each hyperparameter
dropout <- as.factor(param_3['dropout', ])
lambda <- as.factor(param_3['lambda', ])
batch <- as.factor(param_3['bs', ])
lr <- as.factor(param_3['lr', ])
patience <- as.factor(param_3['patience', ])
units_1 <- as.factor(param_3['units_1', ])
units_2 <- as.factor(param_3['units_2', ])

# convert to long format
dropout <- rep(dropout, each = 100)
lambda <- rep(lambda, each = 100)
batch <- rep(batch, each = 100)
lr <- rep(lr, each = 100)
patience <- rep(patience, each = 100)
units_1 <- rep(units_1, each = 100)
units_2 <- rep(units_2, each = 100)

# add the hyperparameter values to the metrics data frame
acc_3$dropout <- dropout
acc_3$lambda <- lambda
acc_3$batch <- batch
acc_3$lr <- lr
acc_3$patience <- patience
acc_3$units_1 <- units_1
acc_3$units_2 <- units_2

# colour palette
col <- viridis(2, option = 'C', end = 0.8) 



############  plotting ###################

# plot the learning curves colour coded by hyberparameter values considered


# dropout
dr <- ggplot(acc_3, aes(x = id, y = value, group = variable, colour = dropout)) +
  geom_point(data = acc_3, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Dropout rate:', option = 'C', end = 0.9) +
  scale_y_continuous(limits = c(0.52, 1), breaks = seq(0.52, 1, by = 0.08)) +
  theme_pv() +
  labs(title = 'Dropout rate', y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))


# batch size
bs <- ggplot(acc_3, aes(x = id, y = value, group = variable, colour = batch)) +
  geom_point(data = acc_3, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Batch size:', option = 'C', end = 0.9) +
  scale_y_continuous(limits = c(0.52, 1), breaks = seq(0.52, 1, by = 0.08)) +
  theme_pv() +
  labs(title = 'Batch size', y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))


# learning rate
ler <- ggplot(acc_3, aes(x = id, y = value, group = variable, colour = lr)) +
  geom_point(data = acc_3, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Learning rate:', option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.52, 1), breaks = seq(0.52, 1, by = 0.08)) +
  theme_pv() +
  labs(title = 'Learning rate', y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))


# lambda
lam <- ggplot(acc_3, aes(x = id, y = value, group = variable, colour = lambda)) +
  geom_point(data = acc_3, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Lambda:', option = 'C', end = 0.9) +
  scale_y_continuous(limits = c(0.52, 1), breaks = seq(0.52, 1, by = 0.08)) +
  theme_pv() +
  labs(title = 'Lambda', y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))

# patience
pat <- ggplot(acc_3, aes(x = id, y = value, group = variable, colour = patience)) +
  geom_point(data = acc_3, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Patience:', option = 'C', end = 0.9) +
  scale_y_continuous(limits = c(0.52, 1), breaks = seq(0.52, 1, by = 0.08)) +
  theme_pv() +
  labs(title = 'Patience', y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))

# units first layer
u_1 <- ggplot(acc_3, aes(x = id, y = value, group = variable, colour = units_1)) +
  geom_point(data = acc_3, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Number of units:\n1st layer:', option = 'C', end = 0.9) +
  scale_y_continuous(limits = c(0.52, 1), breaks = seq(0.52, 1, by = 0.08)) +
  theme_pv() +
  labs(title = 'Number of units:\n1st layer', y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))

# units second layer
u_2 <- ggplot(acc_3, aes(x = id, y = value, group = variable, colour = units_2)) +
  geom_point(data = acc_3, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Number of units:\n2nd layer', option = 'C', end = 0.9) +
  scale_y_continuous(limits = c(0.52, 1), breaks = seq(0.52, 1, by = 0.08)) +
  theme_pv() +
  labs(title = 'Number of units:\n2nd layer', y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))
##########

# Plotting multiple plots
grid.arrange(dr, pat, ler, lam, u_1, u_2)
grid.arrange(u_1, u_2, widths = c(1, 1),
             layout_matrix = rbind(c(1, 2), c(3, NA)))


######################################################################
################ Hyperparameter tuning - fine tuning #################
######################################################################

dropout_set <- c(0, 0.2, 0.4)
lambda_set <- c(0, 1e-04) 
bs_set <- floor(c(0.003, 0.01, 0.02) * N)
lr_set <- c(0.001, 0.0005)


runs_2 <- tuning_run("model_conf_2.R",
                   runs_dir = "runs_4",
                   flags = list(
                     dropout = dropout_set,
                     lambda = lambda_set,
                     bs = bs_set,
                     lr = lr_set))

# get the worst models and their parameters
best <- ls_runs(runs_dir = "runs_4", order = eval_accuracy)

# select only the relevant paramenters
best <- best[, c(2, 7:11, 15)] 


# plot all learning curves 

# extract validation accuracy and loss and save as data frame
run_4 <- read_metrics("runs_4")
acc_4 <- as.data.frame(sapply(run_4, "[[", "val_accuracy"))
loss_4 <- as.data.frame(sapply(run_4, "[[", "val_loss"))

# extract evaluation metrics
eval <- as.data.frame(sapply(run_4, "[[", "evaluation"))
eval <- apply(eval, 2, unlist)
eval <- eval['accuracy', ]

eval_ord <- order(eval, decreasing = TRUE)

acc_4_best <- acc_4[, eval_ord[1:10]]
acc_4 <- acc_4[, - eval_ord[1:10]]

loss_4_best <- loss_4[, eval_ord[1:10]]
loss_4 <- loss_4[, - eval_ord[1:10]]

# add id variable for easy melting
acc_4$id <- c(1:100)
acc_4_best$id <- c(1:100)

loss_4$id <- c(1:100)
loss_4_best$id <- c(1:100)

# convert to long format for easy plotting
acc_4 <- melt(acc_4, id.var = 'id')
loss_4 <- melt(loss_4, id.var = 'id')
acc_4_best <- melt(acc_4_best, id.var = 'id')
loss_4_best <- melt(loss_4_best, id.var = 'id')


########### plotting ############ 

# colour palette
col <- viridis(2, option = 'C', end = 0.8) 

# accuracy
accu <- ggplot(acc_4, aes(x = id, y = value, group = variable)) +
  geom_point(data = acc_4, alpha = 0.2, colour = col[2]) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, size = 0.3, 
              colour = col[2]) +
  geom_point(data = acc_4_best, alpha = 0.2, colour = col[1]) +
  stat_smooth(data = acc_4_best, method = 'loess', geom = 'line', se = FALSE, 
              size = 0.7, colour = col[1]) +
  scale_y_log10(limits = c(0.92, 1), breaks = seq(0.92, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Validation accuracy for 38 models', 
       y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))


# accuracy
loss <- ggplot(loss_4, aes(x = id, y = value, group = variable)) +
  geom_point(data = loss_4, alpha = 0.2, colour = col[2]) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, 
              size = 0.3, colour = col[2]) +
  geom_point(data = loss_4_best, alpha = 0.2, colour = col[1]) +
  stat_smooth(data = loss_4_best, method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7, colour = col[1]) +
  scale_y_log10(limits = c(0.04, 1)) +
  theme_pv() +
  labs(title = 'Validation loss for 38 models', y = "Loss (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))

# Plotting multiple plots
grid.arrange(accu, loss, nrow = 1)




# plotting the validation and training accuracy for the final model

# extract the training and validation data
final_val <- as.data.frame(sapply(run_4, "[[", "val_accuracy"))
final_train <- as.data.frame(sapply(run_4, "[[", "accuracy"))

# select the best model
final_val <- as.data.frame(final_val[, 29])
final_train <- as.data.frame(final_train[, 29])

final <- cbind(final_val, final_train)

# add id variable for easy melting
final$id <- c(1:100)

# add a column name
names(final)[1] <- 'validation'
names(final)[2] <- 'training'

final <- melt(final, id.var = 'id')

# final plotting
ggplot(final, aes(x = id, y = value, group = variable, colour = variable)) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, size = 1) +
  geom_point(alpha = 0.5) +
  scale_colour_viridis_d(option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.92, 1), breaks = seq(0.92, 1, by = 0.02)) +
  theme_pv() +
  xlim(0, 75) +
  labs(title = 'Training and validation learning curves \nfor the final model', 
       y = "Accuracy (log)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.title = element_blank(),
        legend.text = element_text(size = 12))


##########