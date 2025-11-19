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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_NOT_QUOTE, GraphFst
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


class RomanFst(GraphFst):
    """
    Finite state transducer for verbalizing roman numerals
        e.g. tokens { roman { integer: "एक" } } -> एक

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="roman", kind="verbalize", deterministic=deterministic)
        
        # Load ordinal mappings for Hindi
        try:
            ordinal_digit = pynini.string_file(get_abs_path("data/ordinal/digit.tsv")).invert()
            ordinal_teen = pynini.string_file(get_abs_path("data/ordinal/teen.tsv")).invert()
            # Create suffix conversion for ordinals
            from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_SIGMA
            suffix = pynini.cdrewrite(
                ordinal_digit | ordinal_teen,
                "",
                "[EOS]",
                NEMO_SIGMA,
            ).optimize()
        except:
            # If ordinal files don't exist, use identity
            suffix = pynini.closure(pynini.cross("", ""))

        cardinal = pynini.closure(NEMO_NOT_QUOTE)
        ordinal = pynini.compose(cardinal, suffix)

        graph = (
            pynutil.delete("key_cardinal: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynini.accep(" ")
            + pynutil.delete("integer: \"")
            + cardinal
            + pynutil.delete("\"")
        ).optimize()

        graph |= (
            pynutil.delete("default_cardinal: \"default\" integer: \"") + cardinal + pynutil.delete("\"")
        ).optimize()

        graph |= (
            pynutil.delete("default_ordinal: \"default\" integer: \"") + ordinal + pynutil.delete("\"")
        ).optimize()

        graph |= (
            pynutil.delete("key_the_ordinal: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynini.accep(" ")
            + pynutil.delete("integer: \"")
            + pynini.closure(pynutil.insert(""), 0, 1)  # Hindi doesn't use "the" prefix
            + ordinal
            + pynutil.delete("\"")
        ).optimize()

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

