# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from indic_text_normalization.te.graph_utils import MINUS, NEMO_NOT_QUOTE, GraphFst, insert_space
from indic_text_normalization.te.taggers.decimal import quantities


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "పన్నెండు"  fractional_part: "ఐదు సున్నా సున్నా ఆరు" quantity: "కోటి" }
            -> రుణాత్మక పన్నెండు దశాంశం ఐదు సున్నా సున్నా ఆరు కోటి
        decimal { integer_part: "పన్నెండు" quantity: "కోటి" } -> పన్నెండు కోటి
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        delete_space = pynutil.delete(" ")
        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", MINUS) + delete_space, 0, 1)

        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        fractional_default = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )
        fractional = pynutil.insert(" దశాంశం ") + fractional_default

        quantity = delete_space + insert_space + pynutil.delete("quantity: \"") + quantities + pynutil.delete("\"")
        optional_quantity = pynini.closure(quantity, 0, 1)

        graph = optional_sign + (integer + quantity | integer + delete_space + fractional + optional_quantity)

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

