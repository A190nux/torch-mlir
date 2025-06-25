# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import Optional

import torch
import torch.export
import torch.nn as nn

from torch_mlir import fx

from torch_mlir.ir import (
    Operation,
    StringAttr,
)


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# Tests that constants and parameters work generally with the mutation path.
# This doesn't do mutation but ensures that the basics remain functional.
# CHECK-LABEL: test_import_frozen_exported_program
# CHECK:     func.func @main(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK-DAG: %[[a:.+]] = torch.aten.randn
# CHECK-DAG: %[[b:.+]] = torch.vtensor.literal(dense_resource<torch_tensor_3_1_torch.float32> : tensor<3x1xf32>) : !torch.vtensor<[3,1],f32>
# CHECK-DAG: %[[p:.+]] = torch.vtensor.literal(dense<{{.*>+}} : tensor<1x1xf32>) : !torch.vtensor<[1,1],f32>
# CHECK-DAG: %[[tanh:.+]] = torch.aten.tanh %[[ARG0]]
# CHECK-DAG: %[[mul_a:.+]] = torch.aten.mul.Tensor %[[tanh]], %[[a]]
# CHECK-DAG: %[[mul_b:.+]] = torch.aten.mul.Tensor %[[mul_a]], %[[b]]
# CHECK-DAG: %[[mul_p:.+]] = torch.aten.mul.Tensor %[[mul_b]], %[[p]]
# CHECK:     return %[[mul_p]]
def test_import_frozen_exported_program():
    @torch._dynamo.assume_constant_result
    def get_a():
        return torch.randn(1, 4)

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = torch.randn(3, 1)
            self.p = nn.Parameter(torch.randn(1, 1))

        def forward(self, x):
            return torch.tanh(x) * get_a() * self.b * self.p

    m = fx.export_and_import(
        Basic(), torch.randn(3, 4), experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_user_input_mutate
# CHECK:     func.func @main(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.tensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK-DAG: %[[arg1_copy:.+]] = torch.copy.to_vtensor %arg1 : !torch.vtensor<[3,4],f32>
# CHECK-DAG: %[[arg1_mul:.+]] = torch.aten.mul.Tensor %[[arg1_copy]], %arg0
# The Torch 2.6 generates `torch.aten.copy` as an op in this example while the torch versions < 2.6 does not, hence this check is kept as a "COM".
# COM: %{{.*}} = torch.aten.copy %[[arg1_copy]], %[[arg1_mul]], %false : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.bool -> !torch.vtensor<[3,4],f32>
# CHECK-DAG: torch.overwrite.tensor.contents %{{.*}} overwrites %arg1
# CHECK-DAG: %[[arg0_mul:.+]] = torch.aten.mul.Tensor %arg0, %[[arg1_mul]]
# CHECK:     return %[[arg0_mul]]
def test_user_input_mutate():
    class Basic(nn.Module):
        def forward(self, x, y):
            y.mul_(x)
            return x * y

    m = fx.export_and_import(
        Basic(),
        torch.randn(3, 4),
        torch.randn(3, 4),
        experimental_support_mutation=True,
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_frozen_buffer
# CHECK: %[[buffer_literal:.+]] = torch.vtensor.literal
# CHECK: %[[mul:.+]] = torch.aten.mul.Tensor %arg0, %0
# CHECK: return %[[mul]]
def test_frozen_buffer():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.randn(3, 4))

        def forward(self, x):
            return x * self.buffer

    m = fx.export_and_import(
        Basic(), torch.randn(3, 4), experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_frozen_buffer_non_persistent
# CHECK: %[[buffer_literal:.+]] = torch.vtensor.literal
# CHECK: %[[mul:.+]] = torch.aten.mul.Tensor %arg0, %0
# CHECK: return %[[mul]]
def test_frozen_buffer_non_persistent():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.randn(3, 4), persistent=False)

        def forward(self, x):
            return x * self.buffer

    m = fx.export_and_import(
        Basic(), torch.randn(3, 4), experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


class ExternalBufferHooks(fx.FxImporterHooks):
    def prepare_module(self, module_op: Operation):
        module_op.context.allow_unregistered_dialects = True

    def resolve_input(self, gni, value, info):
        return Operation.create(
            "my_dialect.import_buffer",
            results=[info.ir_type],
            attributes={"name": StringAttr.get(info.input_spec.target)},
        ).result

    def store_produced_value(self, gni, py_value, produced_ir_value, info):
        Operation.create(
            "my_dialect.store_buffer",
            operands=[produced_ir_value],
            attributes={"name": StringAttr.get(info.input_spec.target)},
        )


@run
# CHECK-LABEL: test_mutable_buffer
# CHECK: %[[buffer:.+]] = "my_dialect.import_buffer"() {name = "buffer"} : () -> !torch.vtensor<[3,4],f32>
# CHECK: %[[mul:.+]] = torch.aten.mul.Tensor %[[buffer]], %arg0
# CHECK: "my_dialect.store_buffer"(%[[mul]]) {name = "buffer"} : (!torch.vtensor<[3,4],f32>) -> ()
# CHECK: return %arg0
def test_mutable_buffer():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.randn(3, 4))

        def forward(self, x):
            self.buffer.mul_(x)
            return x

    m = fx.export_and_import(
        Basic(),
        torch.randn(3, 4),
        experimental_support_mutation=True,
        hooks=ExternalBufferHooks(),
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_single_input_ConstantArgument
# CHECK: func.func @main
# CHECK: return
def test_single_input_ConstantArgument():
    """Test case 1: Single input ConstantArgument (y=2)"""
    class SingleConstantInputModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y=2):  # Single constant input
            return x * y

    m = fx.export_and_import(
        SingleConstantInputModule(), 
        torch.randn(3, 4),
        experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_single_output_ConstantArgument
# CHECK: func.func @main
# CHECK: return
def test_single_output_ConstantArgument():
    """Test case 2: Single output ConstantArgument (self.scale)"""
    class SingleConstantOutputModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 0.5  # Single constant output

        def forward(self, x):
            scaled = x * self.scale
            return scaled, self.scale  # Return tensor + constant

    m = fx.export_and_import(
        SingleConstantOutputModule(), 
        torch.randn(3, 4),
        experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_multiple_input_ConstantArgument
# CHECK: func.func @main
# CHECK: return
def test_multiple_input_ConstantArgument():
    """Test case 3: Multiple input ConstantArguments"""
    class MultipleConstantInputModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, scale=2.0, offset=1.0, multiplier=3):  # Multiple constant inputs
            return x * scale + offset * multiplier

    m = fx.export_and_import(
        MultipleConstantInputModule(), 
        torch.randn(3, 4),
        experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_multiple_output_ConstantArgument
# CHECK: func.func @main
# CHECK: return
def test_multiple_output_ConstantArgument():
    """Test case 4: Multiple output ConstantArguments"""
    class MultipleConstantOutputModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 0.5
            self.name = "model"
            self.version = 42

        def forward(self, x):
            result = x * self.scale
            # Return tensor + multiple constants
            return result, self.scale, self.name, self.version, True, None

    m = fx.export_and_import(
        MultipleConstantOutputModule(), 
        torch.randn(3, 4),
        experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_input_output_ConstantArgument
# CHECK: func.func @main
# CHECK: return
def test_input_output_ConstantArgument():
    """Test case 5: Multiple input AND output ConstantArguments combined"""
    class CombinedConstantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_scale = 0.5
            self.model_name = "combined_model"

        def forward(self, x, user_scale=2.0, add_bias=True, bias_value=1.0):
            if add_bias:
                result = (x * self.base_scale * user_scale) + bias_value
            else:
                result = x * self.base_scale * user_scale
            
            # Return mix of tensors and constants (both output and input)
            return (
                result,              # tensor
                self.base_scale,     # constantArgument output
                self.model_name,     # constantArgument output
                user_scale,          # constantArgument input
                add_bias,            # constantArgument input
                bias_value,          # constantArgument input
                None                 # constantArgument literal (output)
            )

    m = fx.export_and_import(
        CombinedConstantModule(), 
        torch.randn(3, 4),
        experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_ConstantArgument_edge_cases
# CHECK: func.func @main
# CHECK: return
def test_ConstantArgument_edge_cases():
    """Test edge cases: different constant types and None values"""
    class EdgeCaseConstantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.float_val = 3.14
            self.int_val = 42
            self.str_val = "test"
            self.bool_val = True
            self.none_val = None

        def forward(self, x, input_none=None, input_str="default"):
            result = x * self.float_val
            
            # Return all different ConstantArgument types
            return (
                result,           # tensor
                self.float_val,   # float output constantArgument
                self.int_val,     # int output constantArgument
                self.str_val,     # string output constantArgument
                self.bool_val,    # bool output constantArgument
                self.none_val,    # None output constantArgument
                input_none,       # None input constantArgument
                input_str,        # string input constantArgument
                0,                # literal int
                False             # literal bool
            )

    m = fx.export_and_import(
        EdgeCaseConstantModule(), 
        torch.randn(3, 4),
        experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_ConstantArgument_from_attention_layer
# CHECK: func.func @main
# CHECK: return
def test_ConstantArgument_from_attention_layer():
    """
    Test case using actual MultiheadAttention where a constantArgument appears automatically
    due to returning the attention layer without the weights (need_weights=False)
    """
    class AttentionLikeConstantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(
                embed_dim=64, 
                num_heads=1, 
                dropout=0.1,
                batch_first=True
            )

        def forward(self, query, key, value, need_weights=False):
            return self.attn(query, key, value, need_weights=need_weights)

    m = fx.export_and_import(
        AttentionLikeConstantModule(), 
        torch.randn(1, 10, 64),  # query
        torch.randn(1, 10, 64),  # key  
        torch.randn(1, 10, 64),  # value
        experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()
