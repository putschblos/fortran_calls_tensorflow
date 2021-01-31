load('@org_tensorflow//tensorflow/compiler/aot:tfcompile.bzl', 'tf_library')

tf_library(
    name = 'graph',
    config = 'graph.config.pbtxt',
    cpp_class = 'Graph',
    graph = 'graph.pb',
)
cc_binary(
    name = "libmodel.so",
    srcs = ["graph.cc"],
    deps = [":graph", "//third_party/eigen3"],
    linkopts = ["-lpthread"],
    linkshared = 1,
    copts = ["-O3 -fPIC"],
)


