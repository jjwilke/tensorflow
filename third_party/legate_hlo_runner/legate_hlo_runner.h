#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"

namespace xla {

StatusOr<std::unique_ptr<Executable>>
LoadAndCompileProtoPath(absl::string_view proto_path, HloRunner& runner,
                        bool run_hlo_passes = false);

void PrintBuffers(const xla::Executable& exe);

StatusOr<absl::Span<const BufferAllocation>> GetBufferAllocations(const xla::Executable& exe);

template <class T, class Visitor>
Status VisitInputs(const xla::Executable& exe, Visitor&& visitor){
    TF_ASSIGN_OR_RETURN(absl::Span<const BufferAllocation> allocations, GetBufferAllocations(exe));
    for (const BufferAllocation& alloc : allocations){
        if (alloc.is_entry_computation_parameter()){
            visitor(alloc);
        }
    }
}

template <class T, class Visitor>
Status VisitOutputs(const xla::Executable& exe, Visitor&& visitor){
    TF_ASSIGN_OR_RETURN(absl::Span<const BufferAllocation> allocations,
        GetBufferAllocations(exe));
    for (const BufferAllocation& alloc : allocations){
        if (alloc.maybe_live_out()){
            visitor(alloc);
        }
    }
}

}