# Copyright (c) 2025
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

from indic_text_normalization.brx.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space


class ScientificFst(GraphFst):
    """
    Verbalize scientific-notation tokens, e.g.
      scientific { mantissa: "स्से दशमलव से" exponent: "बा" } ->
        स्से दशमलव से गुणा स्से पावर बा

      scientific { mantissa: "स्से दशमलव से" sign: "ऋणात्मक" exponent: "बा" } ->
        स्से दशमलव से गुणा स्से पावर ऋणात्मक बा
    Note: Numbers are in Bodo native script (से, नै, थाम, etc.)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="scientific", kind="verbalize", deterministic=deterministic)

        delete_space = pynutil.delete(" ")

        mantissa = (
            pynutil.delete('mantissa: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

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

        graph = mantissa + pynutil.insert(" गुणा स्से पावर ") + optional_sign + exponent
        self.fst = self.delete_tokens(graph).optimize()
