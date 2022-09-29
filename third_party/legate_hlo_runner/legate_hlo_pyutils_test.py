
import unittest
import third_party.legate_hlo_runner.legate_hlo_pyutils as lhp
from tensorflow.compiler.xla.service import hlo_pb2

class HloProtoTest(unittest.TestCase):
    def test_load_hlo(self):
        #pb_path = "third_party/legate_hlo_runner/testdata/small_hlo.pb"
        pb_path = "third_party/legate_hlo_runner/testdata/layer_hlo.pb"
        pb = open(pb_path, "rb").read()
        hlo_proto = hlo_pb2.HloProto()
        hlo_proto.ParseFromString(pb)
        print(hlo_proto)

        layer_names = ["layer1", "layer2"]
        layers = lhp.decompose_hlo_into_layers(layer_names, hlo_proto.hlo_module)
        for layer in layers:
            dump_name = layer.name + ".pb"
            with open(dump_name, "wb") as f:
                print(layer.to_hlo_proto())
                f.write(layer.to_hlo_proto().SerializeToString())
            print(layer)
            #print(layer.to_hlo_proto())
            #print("")

    def test_print_hlo(self):
        print("hello")


if __name__ == '__main__':
    unittest.main()
