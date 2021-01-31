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
Futhermore we have to compile the modul `tf2xla_pb2`.

## Step 1 - Save your model
With your trained Tensorflow model in the variable `model` you can now save the Tensorflow graph and weights.
For this you need the following imports. 

```python
from keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import variables
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import freeze_graph 
import tf2xla_pb2
``` 
By using the `tf2xla_pb2` we can save the structure of the in and outputs of the model/graph.
The structure is than saved in the file "graph.config.pbtxt".  

```python
config = tf2xla_pb2.Config()

batch_size = 1

for x in model.inputs:
    x.set_shape([batch_size] + list(x.shape)[1:])
    feed = config.feed.add()
    feed.id.node_name = x.op.name
    feed.shape.MergeFrom(x.shape.as_proto())

for x in model.outputs:
    fetch = config.fetch.add()
    fetch.id.node_name = x.op.name

with open('graph.config.pbtxt', 'w') as f:
    f.write(str(config))
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
After that you can write the graph weights to disk via:

```python
# Write Frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=".",
                  name="graph.pb",
                  as_text=False)

```

## Step 2 - Compiling the model as a Cpp-function

Now that we have saved the structure and weights, we can compile the graph as function with the following Cpp-code.  
graph.cc:
```Cpp
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "graph.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

extern "C" void run(void *inputptr, void *outputptr, int input_size, int output_size) {
/* Basic function to compile and run the Modell defined in graph.pb and graph.config.pbtext
 */

  Eigen::ThreadPool tp(std::thread::hardware_concurrency());
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  Graph graph; 
  graph.set_thread_pool(&device);


  double *input;
  input=(double *)inputptr; double *output;
  output=(double *)outputptr; 
  std::copy(input, input + input_size, graph.arg0_data());

  auto ok = graph.Run();
  std::copy(graph.result0_data(), graph.result0_data() + output_size, output);
}
```
All the files (`graph.cc`, `graph.pb` and `graph.config.pbtxt`) must be in the root folder of the Tensorflow installation.
To compile the Code we modify the `BUILD` File in the Tensorflow root folder. 
The new `BUILD` File is given through:

```
load('@org_tensorflow//tensorflow/compiler/aot:tfcompile.bzl', 'tf_library')
tf_library(
    name = 'graph',
    config = 'graph.config.pbtxt',
    cpp_class = 'Graph',
    graph = 'graph.pb',
)
cc_binary(
    name = "mymodel.so",
    srcs = ["graph.cc"],
    deps = [":graph", "//third_party/eigen3"],
    linkopts = ["-lpthread"],
    linkshared = 1,
    copts = ["-O3 -fPIC"],
) 
```

For the first compilation we have to generate `graph.h` by executing:

```
bazel build --show_progress_rate_limit=600 @org_tensorflow//:graph
```
After that or for all following compilations we compile the graph via:
```
bazel build --show_progress_rate_limit=600 @org_tensorflow//:mymodel.so
```
Essentially we just have to exchange `graph.pb` and `graph.config.pbtxt` to compile different models after the first compilation.

## Step 3 - Calling the model in our Fortran code
The following interface defines the compiled model in the Fortran code.
```fortran 
INTERFACE
        SUBROUTINE model(xptr,yptr,sizex,sizey) BIND(C, name='run')
            IMPORT ::c_ptr
            IMPORT ::C_INT
            TYPE(C_ptr), VALUE :: xptr
            TYPE(C_ptr), VALUE :: yptr
            INTEGER(C_INT), VALUE ::  sizey
            INTEGER(C_INT), VALUE ::  sizex
	END SUBROUTINE
END INTERFACE
``` 
We could call a model with 15 intputs and 1 output by the following.
```fortran 
FUNCTION localWrapper(modelInput) RESULT(modelOutput)
        !------------------------------------------------------------
        ! INPUT/OUTPUT Variables
        REAL(KIND=15), DIMENSION(1:15),INTENT(IN) ::  modelInput
        ReaL(KIND=15)                             ::  modelOutput
        !------------------------------------------------------------
        ! LOCAL VARIABLES
        REAL(C_double), TARGET                    ::  x(15)
        REAL(C_double), TARGET                    ::  y(1)
        INTEGER                                   ::  x_size,y_size
        TYPE(C_ptr)                               ::  xptr,yptr
        !------------------------------------------------------------
        x_size = 15
        y_size = 1


        x = modelInput

        yptr = c_loc(y(1))
        xptr = c_loc(x(1))
        CALL model(xptr,yptr,x_size,y_size)

	modelOutput=y(1)
END FUNCTION localWrapper
```
It is important that we populate the input variable `x` exactly in the same order as in the training of the model.
The same statement holds true for the output.

When compiling the Fortran code we have to link the compiled shared library `mymodel.so`.
When using `gfortran` this can be achieved by adding the flags:
```console
-L. -Wl,-rpath,. mymodel.so 
``` 
