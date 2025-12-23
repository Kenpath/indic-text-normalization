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
from pynini.examples import plurals
from pynini.lib import pynutil

from indic_text_normalization.ml.graph_utils import (
    MINUS,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    insert_space,
)
from indic_text_normalization.ml.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. fraction { integer: "ഇരുപത്തിമൂന്ന്" numerator: "നാല്" denominator: "छः" }-> ഇരുപത്തിമൂന്ന് നാല് ഭാഗിച്ച് छः
        e.g. fraction { numerator: "നാല്" denominator: "छः" } -> നാല് ഭാഗിച്ച് छः
        e.g. fraction { numerator: "ഒന്ന്" denominator: "രണ്ട്" } -> പകുതി
        e.g. fraction { numerator: "ഒന്ന്" denominator: "നാല്" } -> കാൽ
        e.g. fraction { numerator: "മൂന്ന്" denominator: "നാല്" } -> മൂന്ന് കാൽ


    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", MINUS) + pynutil.delete(" "), 0, 1)

        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
        denominator = pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        
        insert_aur = pynutil.insert(" ഉം ")
        insert_bata = pynutil.insert(" ഭാഗിച്ച് ")
        
        # Special cases (like English)
        # 1/2 -> "പകുതി" (half): numerator "ഒന്ന്" + denominator "രണ്ട്" -> "പകുതി"
        numerator_one_half = (
            pynutil.delete("numerator: \"ഒന്ന്\"") + pynutil.delete("\" ")
            + pynini.cross("denominator: \"രണ്ട്\"", "പകുതി")
        )
        
        # 1/4 -> "കാൽ" (quarter): numerator "ഒന്ന്" + denominator "നാല്" -> "കാൽ"
        numerator_one_quarter = (
            pynutil.delete("numerator: \"ഒന്ന്\"") + pynutil.delete("\" ")
            + pynini.cross("denominator: \"നാല്\"", "കാൽ")
        )
        
        # 3/4 -> "മൂന്ന് കാൽ" (three quarters): numerator "മൂന്ന്" + denominator "നാല്" -> "മൂന്ന് കാൽ"
        three_quarters = (
            pynutil.delete("numerator: \"മൂന്ന്\"") + pynutil.delete("\" ")
            + pynutil.insert("മൂന്ന് ")
            + pynini.cross("denominator: \"നാല്\"", "കാൽ")
        )
        
        # Default: numerator ഭാഗിച്ച് denominator (for all other fractions)
        fraction_default = numerator + insert_bata + denominator
        
        # Use priority union like English to handle special cases first, then default
        fraction_graph = plurals._priority_union(
            numerator_one_half,
            plurals._priority_union(
                numerator_one_quarter,
                plurals._priority_union(three_quarters, fraction_default, NEMO_SIGMA),
                NEMO_SIGMA,
            ),
            NEMO_SIGMA,
        ).optimize()

        # Handle morphosyntactic features (for paune, savva, etc.)
        graph_quarter = (
            pynutil.delete("morphosyntactic_features: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )
        
        # Add integer part with "ഉം" (and) for mixed numbers
        self.graph = (
            optional_sign
            + pynini.closure(integer + insert_space + insert_aur, 0, 1)
            + fraction_graph
        ) | graph_quarter

        graph = self.graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
