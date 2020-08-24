#======== Model and settings configuration
#

# model instantiation -----------------------------------------------
# set defaul flags
FLAGS <- flags(
  flag_numeric("dropout", 0),
  flag_numeric("units_1", 512),
  flag_numeric("units_2", 256),
  flag_numeric('lambda', 0),
  flag_numeric('bs', floor(0.003 * N)),
  flag_numeric('lr', 0.001),
  flag_numeric('patience', 10)
)

# model configuration
model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$units_1, input_shape = V, 
              activation = "relu", name = "layer_1",
              kernel_regularizer = regularizer_l2(l = FLAGS$lambda)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = FLAGS$units_2, activation = "relu", name = "layer_2",
              kernel_regularizer = regularizer_l2(l = FLAGS$lambda)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = ncol(y_train), activation = "softmax", name = "layer_out") %>%
  
  compile(loss = "categorical_crossentropy", metrics = "accuracy",
          optimizer = optimizer_adam(lr = FLAGS$lr))

fit <- model %>% fit(
  x = x_train_s, y = y_train,
  validation_data = list(x_val_s, y_val),
  epochs = 100,
  batch_size = FLAGS$bs,
  verbose = 1, 
  callbacks = callback_early_stopping(monitor = "val_accuracy", patience = FLAGS$patience)
)

# store accuracy on test set for each run
score <- model %>% evaluate(
  scale(x_test), y_test,
  verbose = 0
)