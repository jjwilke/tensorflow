#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/io/path.h"
#include "third_party/legate_hlo_runner/legate_hlo_runner.h"
#include "tensorflow/compiler/xla/service/platform_util.h"

constexpr char kSmallHloPbPath[] =
    "third_party/legate_hlo_runner/testdata/small_hlo.pb";

using namespace tensorflow;

namespace xla {

TEST(LegateHloRunner, TestLoadModuleAndBuildExecutable){
  EXPECT_TRUE(true);

  const string proto_path = kSmallHloPbPath;
  //   io::JoinPath(tensorflow::testing::TensorFlowSrcRoot(), kSmallHloPbPath);

  TF_ASSERT_OK_AND_ASSIGN(auto platform, PlatformUtil::GetPlatform("cpu"));

  HloRunner runner(platform);
  TF_ASSERT_OK_AND_ASSIGN(auto exe, LoadAndCompileProtoPath(proto_path, runner));

  PrintBuffers(*exe);
}

}