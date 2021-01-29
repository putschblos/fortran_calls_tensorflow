# fortran_calls_tensorflow

With the growing interest in artificial neural networks one might try to use a trained neural network together with Fortran code.

It is important to notice that the ANN is already **trained**.
If you have already trained your model through Python, R or any other Tensorflow-API this cloud work for you.

Furthermore it should be noted that there's also the option to call the Fortran part of your program out of python via [f2py](https://numpy.org/doc/stable/f2py/). 
In some cases that might be better.


Here's how I did it.

## Tensorflow
For this to work you need to build Tensorflow from source. Unfortunately this guide does **not** work with just the pip package.
[Here's](https://www.tensorflow.org/install/source) were to start, if you don't know how to do that.  

## Step 1 -- Save your model
With your trained Tensorflow model in the variable `model` you can now save the Tensorflow graph and weights.
For this you need the following imports.  

```python
from keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import variables
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import freeze_graph 
```
Convert the still trainable model with the following code in to a function.
With that all the variables are turned into constants, thus the model is not trainable anymore.

```python
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))


# Freaz function
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

```
After that you can write the graph weights and structure to disk via:

```python
# Write Frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=".",
                  name="graph.pb",
                  as_text=False)

```

## Step 1
After you you compiled Tensorflow, you have to build
