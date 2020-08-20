# keras_load_pretrained_model_without_loading_the_model
Often it is required to initialize the layers of a model with the weights of a pre-trained model, but also freeing the memory occupied by the pre-trained model becomes necessary in a low GPU memory nevironment

1. Build a model **new_model** the output of which can be used as input to the rest of the model you are building
2. It is assumed here that the structure of this **new_model** is same as the structure of the layers of the pre-trained model, which we want to be initialized.
