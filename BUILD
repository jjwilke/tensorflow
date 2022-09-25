exports_files([
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "LICENSE",
])

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = {
      "//tensorflow/compiler/xla/tools:run_hlo_module" : "",
      "//tensorflow/compiler/xla/python:xla_client" : "",
      "//third_party/python_runtime:headers" : "",
    },
)
