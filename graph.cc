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
  input=(double *)inputptr;
  double *output;
  output=(double *)outputptr; 
  std::copy(input, input + input_size, graph.arg0_data());

  auto ok = graph.Run();
  std::copy(graph.result0_data(), graph.result0_data() + output_size, output);
}
