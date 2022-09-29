load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object",
    "tf_cc_test",
    "tf_cc_binary",
)

load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

cc_library(
  name = "legate_hlo_runner_impl",
  hdrs = [ "legate_hlo_runner.h" ],
  srcs = [
    "legate_hlo_runner.cc"
  ],
  copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
  deps = [
    "//tensorflow/compiler/xla/service:hlo_runner",
    "//tensorflow/compiler/xla/service/cpu:cpu_executable",
    "//tensorflow/compiler/xla/service:cpu_plugin",
    "//tensorflow/compiler/xla/service:buffer_assignment",
    "//tensorflow/compiler/xla/tools:hlo_module_loader",
  ] + if_cuda_is_configured([
    "//tensorflow/compiler/xla/service/gpu:gpu_executable",
    "//tensorflow/compiler/xla/service:gpu_plugin",
  ])
)

tf_cc_shared_object(
    name = "liblegate_hlo_runner.so",
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":legate_hlo_runner_impl",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**",
    ]),
)

tf_cc_test(
    name = "legate_hlo_runner_test",
    srcs = ["legate_hlo_runner_test.cc"],
    data = [
        ":testdata",
    ],
    deps = [
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        ":legate_hlo_runner_impl",
        "//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
       "//tensorflow/core/platform:path",
       "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:test",
        "//tensorflow/core:framework_internal",
    ],
)

py_library(
    name = "legate_hlo_pyutils",
    srcs_version = "PY3",
    srcs = ["legate_hlo_pyutils.py"],
    deps = [
        "//tensorflow/compiler/xla:xla_proto_py"
    ]
)

py_test(
    name = "legate_hlo_pyutils_test",
    srcs_version = "PY3",
    srcs = [
        "legate_hlo_pyutils_test.py"
    ],
    deps = [
        ":legate_hlo_pyutils",
#        "//tensorflow:tensorflow_py",
    ]
)
