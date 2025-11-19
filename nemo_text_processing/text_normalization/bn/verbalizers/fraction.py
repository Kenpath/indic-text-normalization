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

from nemo_text_processing.text_normalization.bn.graph_utils import (
    NEMO_NOT_QUOTE, 
    GraphFst, 
    delete_space, 
    insert_space,
    BN_DEDH,
    BN_DHAI,
    BN_PAUNE,
    BN_SADHE,
    BN_SAVVA,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction, e.g.
        fraction { integer: "তেইশ" numerator: "চার" denominator: "ছয়"} -> তেইশ চার ভাগ ছয়
        fraction { morphosyntactic_features: "দেড়" } -> দেড়

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"true\"")
            + delete_space
            + pynutil.insert("ঋণাত্মক ")
            + delete_space,
            0,
            1,
        )

        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        numerator = (
            pynutil.delete("numerator:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        denominator = (
            pynutil.delete("denominator:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Handle special Bengali fraction words
        morphosyntactic_features = (
            pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Regular fraction: integer (optional) + numerator + "ভাগ" + denominator
        regular_fraction = (
            pynini.closure(integer + insert_space, 0, 1) 
            + numerator 
            + insert_space 
            + pynutil.insert("ভাগ") 
            + insert_space 
            + denominator
        )

        # Special fraction words (dedh, dhai, sadhe, savva, paune)
        special_fraction = morphosyntactic_features

        graph = regular_fraction | special_fraction
        graph = optional_sign + graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

