# libraries
library(keras)
library(ggplot2)

# load data
load('data_activity_recognition.RData')

# check the dimensions
dim(x_test)
dim(x_train)

# reshape the matrix into vectors - each signal segment is represented by a 
# vector of 125*45 measurements
x_train <- array_reshape(x_train, c(nrow(x_train), 125 * 45)) 
x_test <- array_reshape(x_test, c(nrow(x_test), 125 * 45))

### Visualisation of the classes ##########

# reduce the dimension of the data with PCA
x_test_pca <- prcomp(x_test)

# compute cumulative proportion of variance 
prop <- cumsum(x_test_pca$sdev^2) / sum(x_test_pca$sdev^2) 

# proportion of variance explained by the first two and ten components
prop[2]

# plot the first two principal components (around 31 % of the variance)
ggplot(as.data.frame(x_test_pca$x[, c(1:2)]), aes(x = x_test_pca$x[, 1], y = x_test_pca$x[, 2])) +
  geom_point(aes(colour = y_test))

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




