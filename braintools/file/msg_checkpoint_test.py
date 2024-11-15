# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-


import importlib.util
import unittest
from tempfile import TemporaryDirectory

import brainstate as bst
import brainunit as u
import pytest

import braintools as bts

spec = importlib.util.find_spec("msgpack")

if spec is None:
    pytest.skip("msgpack not installed", allow_module_level=True)


class TestMsgCheckpoint(unittest.TestCase):
    def test_checkpoint_quantity(self):
        data = {
            "name": bst.random.rand(3) * u.ms,
        }

        with TemporaryDirectory() as tmpdirname:
            filename = tmpdirname + "/test_msg_checkpoint.msg"
            bts.file.msgpack_save(filename, data)
            data['name'] += 1 * u.ms

            data2 = bts.file.msgpack_load(filename, target=data)
            self.assertTrue('name' in data2)
            self.assertTrue(isinstance(data2['name'], u.Quantity))
            self.assertTrue(not u.math.allclose(data['name'], data2['name']))

    def test_checkpoint_state(self):
        data = {
            "a": bst.State(bst.random.rand(1)),
            "b": bst.ShortTermState(bst.random.rand(2)),
            "c": bst.ParamState(bst.random.rand(3)),
        }

        with TemporaryDirectory() as tmpdirname:
            filename = tmpdirname + "/test_msg_checkpoint.msg"
            bts.file.msgpack_save(filename, data)

            data2 = bts.file.msgpack_load(filename, target=data)
            self.assertTrue('a' in data2)
            self.assertTrue('b' in data2)
            self.assertTrue('c' in data2)
            self.assertTrue(isinstance(data2['a'], bst.State))
            self.assertTrue(isinstance(data2['b'], bst.ShortTermState))
            self.assertTrue(isinstance(data2['c'], bst.ParamState))
            self.assertTrue(u.math.allclose(data['a'].value, data2['a'].value))
            self.assertTrue(u.math.allclose(data['b'].value, data2['b'].value))
            self.assertTrue(u.math.allclose(data['c'].value, data2['c'].value))
