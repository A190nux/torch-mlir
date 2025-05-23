# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torch.nn as nn
from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================
# Test cases based on your actual failing scenarios
# ==============================================================================

class AttentionWithConstantNeedsWeightsModule(torch.nn.Module):
    """Test case similar to pytorch_attention.py - need_weights=False creates ConstantArgument(value=None)"""
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=1, batch_first=True)

    @export
    @annotate_args([
        None,
        ([1, 10, 64], torch.float32, True),  # query
        ([1, 10, 64], torch.float32, True),  # key  
        ([1, 10, 64], torch.float32, True),  # value
    ])
    def forward(self, query, key, value):
        # need_weights=False should create a ConstantArgument with value=None in outputs
        attn_output, attn_weights = self.attn(query, key, value, need_weights=False)
        return attn_output, attn_weights  # attn_weights will be None (ConstantArgument)


@register_test_case(module_factory=lambda: AttentionWithConstantNeedsWeightsModule())
def AttentionWithConstantNeedsWeightsModule_basic(module, tu: TestUtils):
    query = tu.rand(1, 10, 64)
    key = tu.rand(1, 10, 64) 
    value = tu.rand(1, 10, 64)
    module.forward(query, key, value)


class ModelWithConstantScaleModule(torch.nn.Module):
    """Test case similar to Test2.py - self.constant_scale creates ConstantArgument(value=0.5)"""
    def __init__(self):
        super().__init__()
        self.constant_scale = 0.5  # This becomes a ConstantArgument in outputs

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        scaled_output = x * self.constant_scale
        # Return both the tensor and the constant scale value
        return scaled_output, self.constant_scale


@register_test_case(module_factory=lambda: ModelWithConstantScaleModule())
def ModelWithConstantScaleModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


class ModelWithKwargsConstantModule(torch.nn.Module):
    """Test case similar to Test2.py - constant_multiplier=2.0 creates ConstantArgument input"""
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, constant_multiplier=2.0):  # Default kwarg becomes ConstantArgument
        result = x * constant_multiplier
        return result, constant_multiplier  # Return both tensor and the constant


@register_test_case(module_factory=lambda: ModelWithKwargsConstantModule())
def ModelWithKwargsConstantModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


class ComplexConstantTupleModule(torch.nn.Module):
    """Test case with multiple constant types in a tuple output"""
    def __init__(self):
        super().__init__()
        self.constant_scale = 0.5
        self.constant_name = "model_output"

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        scaled_output = x * self.constant_scale
        # Return a complex tuple with mixed types
        return (
            scaled_output,           # tensor
            self.constant_scale,     # float constant  
            True,                    # bool constant
            self.constant_name,      # string constant
            None,                    # None constant
            42                       # int constant
        )


@register_test_case(module_factory=lambda: ComplexConstantTupleModule())
def ComplexConstantTupleModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================
# Edge cases for ConstantArguments
# ==============================================================================

class OnlyConstantOutputsModule(torch.nn.Module):
    """Test module that returns only constants (no tensors) - edge case"""
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None])
    def forward(self):
        # All outputs are constants
        return 42, "hello", True, None, 3.14


@register_test_case(module_factory=lambda: OnlyConstantOutputsModule())
def OnlyConstantOutputsModule_basic(module, tu: TestUtils):
    module.forward()


class ConstantInOperationModule(torch.nn.Module):
    """Test constants used in tensor operations"""
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        # Various operations with constants
        y = x + 1.0      # float constant
        z = y * 2        # int constant  
        w = z > 0.5      # bool result with float constant
        return y, z, w, 1.0, 2, 0.5  # Mix of tensors and constants


@register_test_case(module_factory=lambda: ConstantInOperationModule())
def ConstantInOperationModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


class NestedConstantModule(torch.nn.Module):
    """Test nested operations that might produce constant arguments"""
    def __init__(self):
        super().__init__()
        self.scale1 = 2.0
        self.scale2 = 3.0

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        # Operations that combine constants
        combined_scale = self.scale1 * self.scale2  # Should be 6.0
        result = x * combined_scale
        return result, self.scale1, self.scale2, combined_scale


@register_test_case(module_factory=lambda: NestedConstantModule())
def NestedConstantModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================
# Test cases that mirror your original failing examples more closely
# ==============================================================================

class AttentionLikeModule(torch.nn.Module):
    """Simplified version of your pytorch_attention.py test case"""
    def __init__(self):
        super().__init__()
        self.embed_dim = 64
        self.num_heads = 1

    @export
    @annotate_args([
        None,
        ([1, 50, 64], torch.float32, True),  # query
        ([1, 10, 16], torch.float32, True),  # key
        ([1, 10, 64], torch.float32, True),  # value
    ])
    def forward(self, query, key, value):
        # Simulate attention operation that returns None for weights
        attn_output = torch.matmul(query, value.transpose(-2, -1))
        attn_weights = None  # This creates ConstantArgument(value=None)
        return attn_output, attn_weights


@register_test_case(module_factory=lambda: AttentionLikeModule())
def AttentionLikeModule_basic(module, tu: TestUtils):
    query = tu.rand(1, 50, 64)
    key = tu.rand(1, 10, 16) 
    value = tu.rand(1, 10, 64)
    module.forward(query, key, value)


class CustomModelWithConstantsLikeTest2(torch.nn.Module):
    """Closely mirrors your Test2.py structure"""
    def __init__(self):
        super().__init__()
        self.constant_scale = 0.5

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, constant_multiplier=2.0):
        scaled_output = x * self.constant_scale * constant_multiplier
        
        # Return tuple similar to your Test2.py
        return (
            scaled_output,              # tensor output
            self.constant_scale,        # float constant
            True,                       # bool constant  
            "attention_output",         # string constant
            None,                       # None constant
            42,                         # int constant
            constant_multiplier         # input constant reflected in output
        )


@register_test_case(module_factory=lambda: CustomModelWithConstantsLikeTest2())
def CustomModelWithConstantsLikeTest2_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))