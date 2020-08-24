# libraries
library(keras)
library(ggplot2)
library(reshape2)
library(viridis)
library(dplyr)
library(tfruns)
library(gridExtra)
library(jsonlite)
library(googledrive)

# load helper functions
source('helpers.R')

# download data to working directory
public_file <- drive_get(as_id("1xkLIiZwh4N2H1n8n_jmGMoARXmbU7Pi3"))

drive_download(public_file)
load('data_activity_recognition.RData')

# check the dimensions
dim(x_test)
dim(x_train)

# reshape the matrix into vectors - each signal segment is represented by a 
# vector of 125*45 measurements
x_train <- array_reshape(x_train, c(nrow(x_train), 125 * 45)) 
x_test <- array_reshape(x_test, c(nrow(x_test), 125 * 45))

######################################################################################
### Visualisation of the classes #####################################################
######################################################################################

# reduce the dimension of the data with PCA
x_test_pca <- prcomp(x_test)

# compute cumulative proportion of variance 
prop <- cumsum(x_test_pca$sdev^2) / sum(x_test_pca$sdev^2) 

# proportion of variance explained by the first two and ten components
prop[2]

# plot the first two principal components (around 31 % of the variance)
ggplot(as.data.frame(x_test_pca$x[, c(1:2)]), aes(x = x_test_pca$x[, 1], y = x_test_pca$x[, 2])) +
  geom_point(aes(colour = y_test)) +
  theme_pv() +
  scale_colour_viridis_d(option = 'C', name = 'Type of activity:') +
  labs(title = 'Sport activities reduced to 2D (~30 % of variance)', y = "PC2", x = "PC1") +
  theme(legend.position = 'right', 
        legend.text = element_text(size = 12)) 

#####

######## Data preparation ######################################################

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

######## Validation-test split ######################################################

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

#################################################################################################################
################ Selecting data preprocessing methods ###########################################################
#################################################################################################################

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
  scale_y_log10(limits = c(0.8, 1)) +
  theme_pv() +
  scale_colour_manual(values = col) +
  scale_linetype_manual(values = c(2, 1, 2, 1, 2, 1)) +
  geom_point(alpha = 0.3) +
  labs(title = 'Learning curves for three different data preprocessing methods', y = "Accuracy", x = "Epochs") +
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
fits <- list()

# run three models

# 2 layrers
fit <- model_2 %>% fit(
  x = x_train_s, y = y_train,
  validation_data = list(x_val_s, y_val),
  epochs = 100,
  verbose = 1)
fits[[1]] <- fit
  
# 4 layrers
fit <- model_4 %>% fit(
  x = x_train_s, y = y_train,
  validation_data = list(x_val_s, y_val),
  epochs = 100,
  verbose = 1)
fits[[2]] <- fit

# 4 layers
fit <- model_6 %>% fit(
  x = x_train_s, y = y_train,
  validation_data = list(x_val_s, y_val),
  epochs = 100,
  verbose = 1)
fits[[3]] <- fit


###### Plotting #############

coln <- c('2_layers_train', '2_layers_val',
          '4_layers_train', '4_layers_val',
          '6_layers_train', '6_layers_val')

# see the helpers.R file for details for data_pt
plot_learn_c <- data_pt(fits, coln = coln) 

# colour palette
col <- rep(viridis(3, option = 'C', end = 0.8), each = 2) 

# plot train and validation learning curves
ggplot(plot_learn_c, aes(x = id, y = value, group = variable, 
                         colour = variable, linetype = variable)) +
  stat_smooth(data = plot_learn_c, method = 'loess', geom = 'line', se = FALSE, size = 1) + 
  scale_y_log10(limits = c(0.93, 1), breaks = seq(0.92, 1, by = 0.02)) +
  theme_pv() +
  scale_colour_manual(values = col) +
  scale_linetype_manual(values = c(2, 1, 2, 1, 2, 1)) +
  geom_point(alpha = 0.3) +
  labs(title = 'Learning curves for models with three different number of layers', y = "Accuracy (log scale)", x = "Epochs") +
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
patience_set <- c(5, 10, 20)

runs <- tuning_run("model_conf.R",
                   runs_dir = "runs",
                   flags = list(
                     dropout = dropout_set,
                     units_1 = units_1_set,
                     units_2 = units_2_set,
                     lambda = lambda_set,
                     bs = bs_set,
                     lr = lr_set,
                     patience = patience_set),
                   sample = 0.02)

########### extract results ############ NOT Included in the report ###

out <- read_metrics("runs")

# extract validation accuracy and plot learning curve
met <- sapply(met, "[[", "val_accuracy")

# extract the parameters values for each run
param <- as.data.frame(sapply(out, "[[", "flags"))

# ggplot needs data frame
met <- as.data.frame(met)

# get the columns index of the best runs - not included in hthe report !!!
met_sort <- apply(met[, -39], 2, max, na.rm = TRUE)
best <- order(met_sort)  

# select the best five runs and store them separately from the rest
met_b <- met[, tail(best, 5)]

# select the rest
met <- met[, best[1:(length(best) - 5)]]

# add id variable for easy melting
met$id <- c(1:100)
met_b$id <- c(1:100)

# convert to long format for easy plotting
met <- melt(met, id.var = 'id')
met_b <- melt(met_b, id.var = 'id')

# colour palette
col <- viridis(2, option = 'C', end = 0.8) 

########### plotting ############ Not included in the final report !!!!
ggplot(met, aes(x = id, y = value, group = variable)) +
  stat_smooth(data = met_b, method = 'loess', geom = 'line', se = FALSE, size = 1,
              colour = col[1]) +
  geom_point(data = met_b, alpha = 0.3, colour = col[1]) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, size = 0.2,
              colour = col[2]) +
  geom_point(data = met, alpha = 0.3, colour = col[2]) +
  scale_y_log10(limits = c(0.82, 1), breaks = seq(0.82, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Learning curves for models with different hyperparameters', y = "Accuracy (log scale)", x = "Epochs")
  
##########


# get the worst models and their parameters
worst <- ls_runs(runs_dir = "runs", order = metric_val_accuracy)

# select only the relevant paramenters
worst <- s[, c(2, 4, 8:14, 18)] 
worst[order(worst$eval_accuracy)[1:10], ] 


####################################################
############ Additional plotting ###################

# plot the learning curves colour coded by hyberparameter values considered

# See helpers.R for read_metrics function
out <- read_metrics("runs")

# extract hyperparameter configuration
param <- sapply(out, "[[", "flags")

# extract validation accuracy and save as data frame
out <- as.data.frame(sapply(out, "[[", "val_accuracy"))
out$id <- c(1:100)
out <- melt(out, id.var = 'id')

# flatten the data frame
param <- apply(param, 2, unlist)

# extact column names for each hyperparameter
dropout <- as.factor(param['dropout', ])
lambda <- as.factor(param['lambda', ])
batch <- as.factor(param['bs', ])
lr <- as.factor(param['lr', ])
patience <- as.factor(param['patience', ])
units_1 <- as.factor(param['units_1', ])
units_2 <- as.factor(param['units_2', ])

# convert to long format
dropout <- rep(dropout, each = 100)
lambda <- rep(lambda, each = 100)
batch <- rep(batch, each = 100)
lr <- rep(lr, each = 100)
patience <- rep(patience, each = 100)
units_1 <- rep(units_1, each = 100)
units_2 <- rep(units_2, each = 100)

# add the hyperparameter values to the metrics data frame
out$dropout <- dropout
out$lambda <- lambda
out$batch <- batch
out$lr <- lr
out$patience <- patience
out$units_1 <- units_1
out$units_2 <- units_2

############ plotting ############

# dropout
dr <- ggplot(out, aes(x = id, y = value, group = variable, colour = dropout)) +
  geom_point(data = out, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Dropout rate:', option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.82, 1), breaks = seq(0.82, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Dropout rate', y = "Accuracy (log scale)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))


# batch size
bs <- ggplot(out, aes(x = id, y = value, group = variable, colour = batch)) +
  geom_point(data = out, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Batch size:', option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.82, 1), breaks = seq(0.82, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Batch size', y = "Accuracy (log scale)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))


# learning rate
ler <- ggplot(out, aes(x = id, y = value, group = variable, colour = lr)) +
  geom_point(data = out, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Learning rate:', option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.82, 1), breaks = seq(0.82, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Learning rate', y = "Accuracy (log scale)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))


# lambda
lam <- ggplot(out, aes(x = id, y = value, group = variable, colour = lambda)) +
  geom_point(data = out, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Lambda:', option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.82, 1), breaks = seq(0.82, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Lambda', y = "Accuracy (log scale)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))

# patience
pat <- ggplot(out, aes(x = id, y = value, group = variable, colour = patience)) +
  geom_point(data = out, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Patience:', option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.82, 1), breaks = seq(0.82, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Patience', y = "Accuracy (log scale)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))

# units first layer
u_1 <- ggplot(out, aes(x = id, y = value, group = variable, colour = units_1)) +
  geom_point(data = out, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Number of units: 1st layer:', option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.82, 1), breaks = seq(0.82, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Number of units: 1st layer:', y = "Accuracy (log scale)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))

# units second layer
u_2 <- ggplot(out, aes(x = id, y = value, group = variable, colour = units_2)) +
  geom_point(data = out, alpha = 0.2) +
  stat_smooth(method = 'loess', geom = 'line', 
              se = FALSE, size = 0.7) +
  scale_colour_viridis_d(name = 'Number of units: 2nd layer', option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.82, 1), breaks = seq(0.82, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Number of units: 2nd layer:', y = "Accuracy (log scale)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.text = element_text(size = 12))
##########

# Plotting multiple plots
grid.arrange(dr, bs, ler, lam)
grid.arrange(pat, u_1, u_2, widths = c(1, 1),
             layout_matrix = rbind(c(1, 2), c(3, NA)))


######################################################################
################ Hyperparameter tuning - fine tuning #################
######################################################################

dropout_set <- c(0, 0.2)
units_2_set <- c(128, 64)
lambda_set <- c(0, 1e-04) 
bs_set <- floor(c(0.003, 0.01, 0.02) * N)
lr_set <- c(0.005, 0.01)


runs_2 <- tuning_run("model_conf_2.R",
                   runs_dir = "runs_2",
                   flags = list(
                     dropout = dropout_set,
                     units_2 = units_2_set,
                     lambda = lambda_set,
                     bs = bs_set,
                     lr = lr_set))


# get the models and their parameters
fine <- ls_runs(runs_dir = "runs_2", order = eval_accuracy)

# select only the relevant paramenters
fine <- fine[12, c(2, 5, 8:12, 16)] 


# plot all learning curves 

# extract validation accuracy and save as data frame
out_fine <- read_metrics("runs_2")
out_fine <- as.data.frame(sapply(out_fine, "[[", "val_accuracy"))
out_fine_l <- as.data.frame(sapply(out_fine_l, "[[", "val_loss"))

# add id variable for easy melting
out_fine$id <- c(1:100)
out_fine_l$id <- c(1:100)

# convert to long format for easy plotting
out_fine <- melt(out_fine, id.var = 'id')
out_fine_l <- melt(out_fine_l, id.var = 'id')

# colour palette
col <- viridis(2, option = 'C', end = 0.8) 


########### plotting ############ 
# For some reasons the folders in the 'runs_2' directory are not in the same order as the one 
# generated by lr_runs function, the position of the validation data in the 'out_fine' set corresponding to the 12 best models were extracted manually/visually rather than programmatically 

# plotingng acccuracy
ggplot(out_fine, aes(x = id, y = value, group = variable)) +
  
  # Plot the 12 models in separate geoms
  stat_smooth(data = out_fine[out_fine$variable  == 'V26', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V26', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V33', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V33', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V36', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V36', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V41', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V41', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V23', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V23', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V22', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V22', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V40', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V40', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V43', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V43', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V18', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V18', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V45', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V45', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V5', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine[out_fine$variable  == 'V5', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine[out_fine$variable  == 'V19', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  
  geom_point(data = out_fine[out_fine$variable  == 'V19', ], alpha = 0.3, colour = col[1]) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, size = 0.2,
              colour = col[2]) +
  geom_point(data = out_fine, alpha = 0.1, colour = col[2]) +
  scale_y_log10(limits = c(0.93, 1), breaks = seq(0.82, 1, by = 0.04)) +
  theme_pv() +
  labs(title = 'Learning curves for 48 models with 12 best models highlighted', 
       y = "Accuracy (log scale)", x = "Epochs")

# # plotingng  loss
ggplot(out_fine_l, aes(x = id, y = value, group = variable)) +
  
  # Plot the 12 models in separate geoms
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V26', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V26', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V33', ], method = 'loess',
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V33', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V36', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V36', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V41', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V41', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V23', ], method = 'loess',
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V23', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V22', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V22', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V40', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V40', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V43', ], method = 'loess',
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V43', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V18', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V18', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V45', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V45', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V5', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V5', ], alpha = 0.3, colour = col[1]) +
  
  stat_smooth(data = out_fine_l[out_fine_l$variable  == 'V19', ], method = 'loess', 
              geom = 'line', se = FALSE, size = 1, colour = col[1]) +
  
  geom_point(data = out_fine_l[out_fine_l$variable  == 'V19', ], alpha = 0.3, colour = col[1]) +
  stat_smooth(method = 'loess', geom = 'line', se = FALSE, size = 0.2,
              colour = col[2]) +
  geom_point(data = out_fine_l, alpha = 0.1, colour = col[2]) +
  scale_y_log10(limits = c(0.015, 1)) +
  theme_pv() +
  labs(title = 'Learning curves for 48 models with 12 best models highlighted', 
       y = "Loss (log scale)", x = "Epochs")


# plotting the validation and training accuracy for the final model

# extract the training and validation data
final <- read_metrics("runs_2")
final_val <- as.data.frame(sapply(final, "[[", "val_accuracy"))
final_train <- as.data.frame(sapply(final, "[[", "accuracy"))

# select the best model
final_val <- as.data.frame(final_val[, 43])
final_train <- as.data.frame(final_train[, 43])

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
  geom_point(alpha = 0.3) +
  scale_colour_viridis_d(option = 'C', end = 0.9) +
  scale_y_log10(limits = c(0.95, 1), breaks = seq(0.94, 1, by = 0.02)) +
  theme_pv() +
  xlim(0, 75) +
  labs(title = 'Training and validation learning curves for the final model', y = "Accuracy (log scale)", x = "Epochs") +
  theme(legend.position = "bottom", 
        legend.title = element_blank(),
        legend.text = element_text(size = 12))


##########