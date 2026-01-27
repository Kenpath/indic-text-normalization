# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from indic_text_normalization.gu.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space


class PowerFst(GraphFst):
    """
    Finite state transducer for verbalizing power, e.g.
        power { base: "દસ" sign: "ઋણાત્મક" exponent: "સાત" } -> દસ પાવર ઋણાત્મક સાત
        power { base: "બે" exponent: "ત્રણ" } -> બે પાવર ત્રણ
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="power", kind="verbalize", deterministic=deterministic)

        delete_space = pynutil.delete(" ")
        base = pynutil.delete('base: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        
        optional_sign = pynini.closure(
            delete_space
            + pynutil.delete('sign: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + insert_space,
            0,
            1,
        )
        
        exponent = (
            delete_space
            + pynutil.delete('exponent: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = base + pynutil.insert(" પાવર ") + optional_sign + exponent

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
