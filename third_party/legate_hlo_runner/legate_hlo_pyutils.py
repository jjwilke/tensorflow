from __future__ import annotations
from tkinter import W
from typing_extensions import Self

from tensorflow.compiler.xla.service import hlo_pb2
from tensorflow.compiler.xla import xla_data_pb2
import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Optional, Mapping, Set, Callable
import functools



_TYPE_MAPPING = {
    xla_data_pb2.PrimitiveType.S32 : np.int,
    xla_data_pb2.PrimitiveType.F32 : np.float
}

def get_shape_short_str(shape: xla_data_pb2.ShapeProto) -> str:
    type_name = xla_data_pb2.PrimitiveType.Name(shape.element_type)
    return "%s %s" % (type_name, shape.dimensions)

def get_instr_short_str(instr: hlo_pb2.HloInstructionProto) -> str:
    return "(%d) %s %s" % (instr.id, instr.name, get_shape_short_str(instr.shape))

def get_logical_buffer_mapping(hlo_proto: hlo_pb2.HloProto) -> Mapping[Tuple[int, ...], hlo_pb2.LogicalBufferProto]:
    logical_buffers = {}
    for log_buf in hlo_proto.buffer_assignment.logical_buffers:
        idx = (log_buf.defined_at.instruction_id,) + tuple(log_buf.defined_at.shape_index)
        logical_buffers[idx] = log_buf
    return logical_buffers

# Decompose computations into layers

class HloComputationTree:
    def __init__(self, computation: hlo_pb2.HloComputationProto):
        self.computation = computation
        self.instructions: List[HloInstructionTree] = []

    def __str__(self):
        str_arr = []
        self._append_str(str_arr)
        return "\n".join(str_arr)

    def _append_str(self, str_arr: List[str], indent: str = ""):
        str_arr.append("%s%s" % (indent, self.computation.name))
        for instr in self.instructions:
            instr._append_str(str_arr, indent + " ")

class HloInstructionTree:
    def __init__(self, root_instruction: hlo_pb2.HloInstructionProto, hlo_module: hlo_pb2.HloModuleProto):
        self.root_instruction = root_instruction
        self.computations: List[HloComputationTree] = []
        # This is N^2, but whatever for now
        for id in root_instruction.called_computation_ids:
            for comp in hlo_module.computations:
                if comp.id == id:
                    tree = HloComputationTree(comp)
                    self.computations.append(tree)
                    for instr in comp.instructions:
                        tree.instructions.append(HloInstructionTree(instr, hlo_module))

    def append_computations(self, hlo_module: hlo_pb2.HloModuleProto):
        for comp in self.computations:
            hlo_module.computations.append(comp)
            for instr in comp.instructions:
                instr.append_computations(hlo_module)

    def add_instruction_defs(self, def_mapping: Mapping[int, hlo_pb2.HloInstructionProto]):
        def_mapping[self.root_instruction.id] = self.root_instruction
        for comp in self.computations:
            for instr in comp.instructions:
                instr.add_instruction_defs(def_mapping)

    def undefined_uses(self, def_mapping: Mapping[int, hlo_pb2.HloInstructionProto]):
        undefined = set()
        for operand in self.root_instruction.operand_ids:
            if not operand in def_mapping:
                undefined.add(operand)

        for comp in self.computations:
            for instr in comp.instructions:
                undefined.update(instr.undefined_uses(def_mapping))
        return undefined


    def uses(self, instruction: hlo_pb2.HloInstructionProto) -> bool:
        for operand in self.root_instruction.operand_ids:
            if operand == instruction.id:
                return True

        for comp in self.computations:
            for instr in comp.instructions:
                if instr.uses(instruction):
                    return True

        return False

    def outputs(self, inputs: Set[int]) -> List[hlo_pb2.HloInstructionProto]:
        outputs = []
        for operand in self.root_instruction.operand_ids:
            if operand in inputs:
                outputs.append(self.root_instruction)
                break

        for comp in self.computations:
            for instr in comp.instructions:
                outputs.extend(instr.outputs(inputs))

        return outputs


    def __str__(self):
        str_arr = []
        self._append_str(str_arr)
        return "\n".join(str_arr)

    def _append_str(self, str_arr: List[str], indent: str = ""):
        str_arr.append("%s(%d): %s -> %s" % (self.root_instruction.name, self.root_instruction.id,
            get_shape_short_str(self.root_instruction.shape), self.root_instruction.operand_ids))
        for comp in self.computations:
            comp._append_str(str_arr, indent + " ")

class HloLayer:
    def __init__(self, name: str, hlo_module: hlo_pb2.HloModuleProto, backprop: bool = True):
        if backprop:
            self.name = name + ".backward"
        else:
            self.name = name + ".forward"
        self.instructions: List[HloInstructionTree] = []
        self.instruction_defs: Mapping[int, hlo_pb2.HloInstructionProto] = {}
        self.parameters: List[hlo_pb2.HloInstructionProto] = []
        self.undefined_inputs: List[hlo_pb2.HloInstructionProto] = []
        self.undefined_constants: List[hlo_pb2.HloInstructionProto] = []
        self.outputs: List[hlo_pb2.HloInstructionProto] = []
        self.roots: List[hlo_pb2.HloInstructionProto] = []

        # find the entry computation
        entry_comp = None
        for comp in hlo_module.computations:
            if comp.id == hlo_module.entry_computation_id:
                entry_comp = comp
                break
        self.entry_id = entry_comp.id
        self.root_id = entry_comp.root_id

        assert entry_comp is not None
        for instr in entry_comp.instructions:
            if name in instr.metadata.op_name:
                is_backwards_pass = "transpose" in instr.metadata.op_name
                if backprop == is_backwards_pass:
                    self.instructions.append(HloInstructionTree(instr, hlo_module))

        # The set of instruction ids that are produced by this layer
        for instr in self.instructions:
            instr.add_instruction_defs(self.instruction_defs)

        for instr in entry_comp.instructions:
            if instr.id == entry_comp.root_id: # The root that produces the result
                for operand in instr.operand_ids:
                    if operand in self.instruction_defs:
                        self.roots.append(self.instruction_defs[operand])

        # now see which parameters need to be included
        for instr in entry_comp.instructions:
            if instr.opcode in ["parameter", "constant"]:
                for potential_user in self.instructions:
                    if potential_user.uses(instr):
                        self.parameters.append(instr)
                        # consider this layer to be the definer of the ID
                        self.instruction_defs[instr.id] = instr
                        break


        undefined_uses = set()
        for instr in self.instructions:
            undefined_uses.update(instr.undefined_uses(self.instruction_defs))

        for comp in hlo_module.computations:
            for instr in comp.instructions:
                if instr.id in undefined_uses:
                    if instr.opcode == "constant":
                        self.undefined_constants.append(instr)
                    else:
                        self.undefined_inputs.append(instr)

    def to_hlo_proto(self) -> Optional[hlo_pb2.HloProto]:
        hlo_proto = hlo_pb2.HloProto()
        hlo_module = hlo_pb2.HloModuleProto()
        hlo_module.name = self.name

        entry_parameter_ids = set()

        entry_comp = hlo_pb2.HloComputationProto()
        entry_comp.id = self.entry_id
        entry_comp.name = "%s_main.%d" % (self.name, self.entry_id)
        new_parameter_number = 0

        new_program_shape = xla_data_pb2.ProgramShapeProto()
        for param in self.parameters + self.undefined_inputs:
            if param.opcode == "parameter":
                new_param = param
            else: # some other input coming from another module
                new_param = hlo_pb2.HloInstructionProto()
                new_param.CopyFrom(param)
                new_param.ClearField("operand_ids")
                new_param.opcode = "parameter"

            new_param.parameter_number = new_parameter_number
            new_parameter_number += 1
            entry_comp.instructions.append(new_param)
            entry_parameter_ids.add(new_param.id)
            new_program_shape.parameters.append(new_param.shape)
            new_program_shape.parameter_names.append(new_param.name)

        for constant in self.undefined_constants:
            entry_comp.instructions.append(constant)

        all_outputs = self.outputs + self.roots
        if len(all_outputs) == 1:
            entry_comp.root_id = all_outputs[0].id
            new_program_shape.result.CopyFrom(all_outputs[0].shape)
        else:
            root_instruction = hlo_pb2.HloInstructionProto()
            root_instruction.id = self.root_id
            root_instruction.name = "new_root.%d" % self.root_id
            root_instruction.opcode = "tuple"
            root_shape = xla_data_pb2.ShapeProto()
            for output in all_outputs:
                root_shape.tuple_shapes.append(output.shape)
                root_instruction.operand_ids.append(output.id)
            root_instruction.shape.CopyFrom(root_shape)
            new_program_shape.result.CopyFrom(root_shape)
            entry_comp.instructions.append(root_instruction)
            entry_comp.root_id = self.root_id

        for instr in self.instructions:
            print("top-level ", instr)
            if not instr.root_instruction.id in entry_parameter_ids:
                entry_comp.instructions.append(instr.root_instruction)
            instr.append_computations(hlo_module)

        entry_comp.program_shape.CopyFrom(new_program_shape)
        hlo_module.entry_computation_id = entry_comp.id
        hlo_module.computations.append(entry_comp)
        hlo_module.host_program_shape.CopyFrom(new_program_shape)

        # validate the module
        all_computation_ids = set()
        all_instruction_ids = set()
        for comp in hlo_module.computations:
            if comp.id in all_computation_ids:
                return None
            all_computation_ids.add(comp.id)
            for instr in comp.instructions:
                if instr.id in all_instruction_ids:
                    return None
                all_instruction_ids.add(instr.id)

        hlo_proto.hlo_module.CopyFrom(hlo_module)
        return hlo_proto


    def compute_outputs(self, inputs: Set[int]):
        self.outputs = []
        for id, instr in self.instruction_defs.items():
            if id in inputs:
                self.outputs.append(instr)


    def __str__(self):
        str_arr = [ "Layer %s" % self.name ]
        for instr in self.instructions:
            str_arr.append(str(instr))
        for param in self.parameters:
            str_arr.append("Parameter %s" % get_instr_short_str(param))
        for input in self.undefined_inputs:
            str_arr.append("Input %s" % get_instr_short_str(input))
        for output in self.outputs:
            str_arr.append("Output %s" % get_instr_short_str(output))
        for root in self.roots:
            str_arr.append("Root %s" % get_instr_short_str(root))
        return "\n".join(str_arr)


class HloStore:
    def __init__(self, id: int, shape: xla_data_pb2.ShapeProto):
        self.id = id
        self.shape = shape


def assign_global_store_ids(layers: List[HloLayer]) -> List[HloStore]:
    stores: List[HloStore] = []
    # A mapping from the original HloInstruction id to the logical HloStore
    store_mapping: Mapping[int, HloStore] = {}

    for layer in layers:
        # first figure out all the outputs from the layers
        for output in layer.outputs:
            store = HloStore(id=len(stores), shape=output.shape)
            layer.output_stores.append(store)
            store_mapping[output.id] = store
            stores.append(store)

    # give unique IDs to all the parameters
    params_defined = set()
    for layer in layers:
        for param in layer.parameters:
            if not param.id in store_mapping:
                params_defined.add(param.id)
                store = HloStore(id=len(stores), shape=param.shape)
                stores.append(store)
                store_mapping[param.id] = store
            else:
                store = store_mapping[param.id]
            layer.parameter_stores.append(store)

    for layer in layers:
        # now loop through inputs and connect them to their output
        for input in layer.undefined_inputs:
            store = store_mapping[input.id]
            layer.input_stores.append(store)

    return stores

class PendingHloLayer:
    def __init__(self, layer: HloLayer):
        self.layer = layer
        self.pending_count = 0

class HloLayerQueue:
    def __init__(self, layers: List[HloLayer]):
        self.depends_on: Mapping[int, List[PendingHloLayer]]
        self.ready_layers = []

        for layer in layers:
            pending = PendingHloLayer(layer)
            if layer.inputs:
                for store in layer.input_stores:
                    if not store.id in self.depends_on:
                        self.depends_on[store.id] = [ ]
                        pending.pending_count += 1
            else:
                self.ready_layers.append(layer)


    def pop_ready(self) -> Optional[HloLayer]:
        if not self.ready_layers:
            return None
        return self.ready_layers.pop()

    def finish(self, layer: HloLayer):
        for store in layer.output_stores:
            if store.id in self.depends_on:
                for pending in self.depends_on[store.id]:
                    pending.pending_count -= 1
                    if pending.pending_count == 0:
                        self.ready_layers.append(pending.layer)
                del self.depends_on[store.id]



def topological_iterate_layers_and_stores(layers: List[HloLayer], callback: Callable[[HloLayer], None]):
    stores =  assign_global_store_ids(layers)
    queue = HloLayerQueue(layers)
    while ready := queue.pop_ready() is not None:
        # TODO(magic in the callback to create a legate task)
        callback(ready)

def decompose_hlo_into_layers(names: List[str], hlo_module: hlo_pb2.HloModuleProto) -> List[HloLayer]:
    layers = []
    for name in names:
        layers.append(HloLayer(name, hlo_module, backprop=False))
        layers.append(HloLayer(name, hlo_module, backprop=True))

    # create a global index of all undefined inputs to different layers
    undefined_inputs = set()
    for layer in layers:
        for input in layer.undefined_inputs:
            undefined_inputs.add(input.id)

    for layer in layers:
        layer.compute_outputs(undefined_inputs)

    return layers

class LegateHloStore:
    pass

class LegateLogicalBuffer:
    def __init__(self, dtype: npt.DTypeLike, shape: Tuple[int], logical_buffer: Optional[hlo_pb2.LogicalBufferProto] = None):
        self.dtype = dtype
        self.shape = shape
        self.size = functools.reduce(lambda x, y: x*y, shape, 1)
        self.logical_buffer = logical_buffer

    def __str__(self):
        return "%s: %s\n%s" % (self.dtype, self.shape, self.logical_buffer)

    @classmethod
    def from_shape(cls, shape: hlo_pb2.ShapeProto, buffer_index: Tuple[int], logical_buffers: Mapping[Tuple(int), hlo_pb2.LogicalBufferProto]) -> List[LegateLogicalBuffer]:
        if len(shape.tuple_shapes) > 0:
            stores = []
            for idx, subshape in enumerate(shape.tuple_shapes):
                stores.extend(cls.from_shape(subshape, buffer_index + (idx,), logical_buffers))
            return stores
        return [ LegateLogicalBuffer(_TYPE_MAPPING[shape.element_type], tuple(shape.dimensions), logical_buffers[buffer_index]) ]

    @classmethod
    def from_hlo_instruction(cls, hlo_instruction: hlo_pb2.HloInstructionProto, logical_buffers: Mapping[Tuple(int), hlo_pb2.LogicalBufferProto]) -> List[LegateLogicalBuffer]:
        return cls.from_shape(hlo_instruction.shape, (hlo_instruction.id,), logical_buffers)

