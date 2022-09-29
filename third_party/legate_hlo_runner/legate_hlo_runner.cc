
#include "third_party/legate_hlo_runner/legate_hlo_runner.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

#ifdef GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#endif

#include "tensorflow/core/platform/errors.h"

namespace xla {

StatusOr<std::unique_ptr<Executable>>
LoadAndCompileProtoPath(absl::string_view proto_path, HloRunner& runner,
                        bool run_hlo_passes){
  static constexpr absl::string_view kInputFormat = "pb";
  TF_ASSIGN_OR_RETURN(auto module,
      LoadModuleFromFile(std::string(proto_path), hlo_module_loader_details::Config(),
                         std::string(kInputFormat), /*config_modifier_hook=*/{}));

  return runner.CreateExecutable(std::move(module), run_hlo_passes);
}

absl::string_view GetName(const xla::HloRunner& runner){
  return runner.Name();
}

StatusOr<absl::Span<const BufferAllocation>> VisitAllocations(const Executable& exe){
  auto* cpu_exe = dynamic_cast<const cpu::CpuExecutable*>(&exe);
  if (cpu_exe){
    return absl::Span<const BufferAllocation>(cpu_exe->buffer_assignment().Allocations());
  }

#ifdef GOOGLE_CUDA
  auto* gpu_exe = dynamic_cast<const gpu::GpuExecutable*>(&exe);
  if (gpu_exe){
    return gpu_exe->GetAllocations();
  }
#endif

  return tensorflow::errors::InvalidArgument(
    "VisitAllocations: executable is not a valid xla::Executable");
}

void PrintBuffers(const Executable& exe){
  auto* cpu_exe = dynamic_cast<const cpu::CpuExecutable*>(&exe);
  if (!cpu_exe){
    std::cerr << "Not a CPU executable" << std::endl;
    return;
  }

  std::cout << "Have executable for " << exe.module().ToString() << std::endl;

  ShapeTree<void*> tree(exe.module().result_shape());
  tree.ForEachElement([](const ShapeIndex& idx, void *const ptr){
    std::cout << idx << " has ptr " << ptr << std::endl;
  });

  const auto& assignment = cpu_exe->buffer_assignment();
  for (BufferAllocation::Index i = 0; i < assignment.Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment.GetAllocation(i);
    std::cout << "Allocation " << i << " is " << allocation.ToString() << std::endl;
  }
}

}



