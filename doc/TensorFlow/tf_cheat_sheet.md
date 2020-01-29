# TENSORFLOW CHEAT SHEET

## LAYERS

To access the layers, you need to import tensorflow and use tf.keras.layers. as prefix.

- Dense(num_nodes, activation='name_of_activation_function', (input_shape=(size_of_inputs))).
  - Dense is a fully connected layer.
  - The num_nodes correspond to the number of nodes you want in the layer.
  - activation defines the activation function you want to use in that layer.
  - input_shape is only required for the first layer defined in the model, after that it can be ignored.
- Flatten(input_shape=(size))
  - Flatten will change an nD array into a 1D array (this counts as a layer eventhough there is no learning)
- Conv2D(num_filters, (filter_dimensions), activation='name_of_activation_function', (input_shape=(size_of_inputs)))
  - Conv2D does is a 2-dimensional convolution layer (used often with images).
  - num_filters is the number of filters you want to use in your layer (similar-ish to the number of nodes in Dense)
  - (filter_dimensions) is the size of the filters used to do convolution (since we do 2D convolution, you need to dimension sizes, however usually the filters are square).
  - activation defines the activation function you want to use in that layer.
  - input_shape is only required for the first layer defined in the model, after that it can be ignored.
- MaxPooling2D((pooling_dimensions))
  - MaxPooling2D is used to reduce the size of the image. It does this by only keeping the max value in the area considered.
  - (pooling_dimensions) is the size of the area to consider while finding the max value.

## MODEL

To create a new model in tensorflow you need to start by initializing it:

```python
model = tf.keras.layers.Sequential()
```

By initializing a model as Sequential, we make sure that when we add layers to the model they follow each other sequentially.
The other way to initialize a model is to use Model but that is more complicated and I recommend going to the tesorflow guides
to learn how to do this.

Now that we have initialize our model, it is now turn to add our layer and make our model.

```python
model.add(tf.keras.layers.layer_type(*params, input_shape=(size_of_inputs))) # for this first layer we need the input_shape
model.add(tf.keras.layers.layer_type(*params))
# ...
model.add(tf.keras.layers.layer_type(*params, activation='output_activation_function')) # the last layer always correspond to
                                                                                        # the output layer, therefore make sure
                                                                                        # that the number of nodes correspond
                                                                                        # to the number of outputs.
```

The layer_type is one of the layers mentioned in the layer section, or layers that can be found on the tensorflow website. The
depth of the model (number of layers), is totally up to you, you just need to make sure that the first layer contains the
input_shape and that the last layer correspond to the output layer and therefore has as number of nodes the number or outputs.

Finally, the last step to creating a TensorFlow model is to add what loss we consider, which optimizer technique we use and
the metrics we would like to consider. This is called compiling the model:

```python
model.compile(optimizer='optimizer_name', loss='type_of_loss', metrics['list of metrics to consider'])
```

In general we use MSE as loss when doing regression and crossentropy when doing classification. With this, your model is done
and is ready to be trained and used. If you want to see a summary of your model, you just need to type:

```python
model.summary()
```

## ACTIVATION FUNCTIONS

- sigmoid
- tanh
- linear
- exponential
- relu
- elu
- selu
- softmax

## OPTIMIZERS

- sgd
- adam
- adamax
- adagrad
- adadelta
- nadam
- rmsprop
- ftrl

## LOSSES

- Regression
  - mean_squared_error
  - mean_absolute_error
  - mean_absolute_percentage_error
  - mean_squared_logarithmic_error
- Classification
  - binary_crossentropy (2-class)
  - categorical_crossentropy
  - sparse_categorical_crossentropy
  - hinge (2-class)
  - squared_hinge (2-class)
  - categorical_hinge
- Both
  - huber_loss
  - cosine_similarity
  - kullback_leibler_divergence
  - logcosh
  - poisson

## METRICS

- all the losses can also be metrics.
- accuracy
- tf.keras.metrics.AUC()
- binary_accuracy
- categorical_accuracy
- tf.keras.metrics.FalseNegatives()
- tf.keras.metrics.FalsePositives()
- mean
- tf.keras.metrics.MeanIoU()
- tf.keras.metrics.MeanRelativeError()
- mean_tensor
- tf.keras.metrics.Precision()
- tf.keras.metrics.PrecisionAtRecall()
- tf.keras.metrics.Recall()
- root_mean_squared_error
- tf.keras.metrics.SensitivityAtSpecificity()
- tf.keras.metrics.SpecificityAtSensitivity()
- sparse_categorical_accuracy
- sum
- tf.keras.metrics.TrueNegatives
- tf.keras.metrics.TruePositives
- top_k_categorical_accuracy
- sparse_top_k_categorical_accuracy

## Training the model

To train a model it is realtively simple, all that is needed to do is:

```python
model.fit(training_inputs, training_outputs, epochs=num_training_cycles, (batch_size=size_of_subset, verbose=verbose_flag))
```

In the above code, the training_inputs correspond to your inputs the model receive, these are the ones you want to use for
the training. training_outputs correspond to the labels/true value of the inputs given to the model, again the training
values. num_training_cycles represent the number of times you go through your whole dataset during the training phase.
Setting a batch_size is optional, it will set it by default. However, if you want to set it yourself, then you can specify
it in the fit code. The verbose attribute decides how much text is printed by the internal TensorFlow code. 0 is silent
(no print), 1 prints a progress bar plus the ETA, loss and metrics.

If you want to save the history of evolution of the model then you can just do:

```python
history = model.fit(training_inputs, training_outputs, epochs=num_training_cycles, (batch_size=size_of_subset))
```

All the history of the evolution will be stored in the history variable.

## Evaluating the model

Last but not least is the code to evaluate your model with the test set.

```python
model.evaluate(test_inputs, test_outputs, (batch_size=size_of_subset, verbose=verbose_flag))
```

The test_inputs and test_outputs are your test_set and its labels/true value. Again you can set batch_size to evaluate
the test_set in parts. Finally, verbose is basically the same as in the training, however there is the additional flag 2
for more info. If you want to save the loss and metrics outputed by the evaluation, you just need to do:

```python
loss, metrics = model.evaluate(test_inputs, test_outputs, (batch_size=size_of_subset, verbose=verbose_flag))
```

Now the loss is stored in the loss variable and the metrics in the metrics variable. If you are looking at multiple metrics,
then your metrics variable will be a list.

This is just a short and basic cheat sheet on TensorFlow and more specifically on the Keras module of TensorFlow, there is
still a lot more that can be done with TensorFlow, and I recommend looking through the documentation of TensorFlow to deepen
your knowledge of it.
