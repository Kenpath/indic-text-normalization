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

from indic_text_normalization.gu.graph_utils import (
    MINUS,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    insert_space,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractions, e.g.
        fraction { numerator: "ત્રણ" denominator: "ચાર" } -> ત્રણ બટા ચાર
        fraction { integer_part: "બાર" numerator: "ત્રણ" denominator: "ચાર" } -> બાર અને ત્રણ બટા ચાર
        fraction { negative: "true" numerator: "એક" denominator: "બે" } -> ઋણાત્મક એક બટા બે
        fraction { numerator: "એક" denominator: "બે" } -> અડધું
        fraction { numerator: "એક" denominator: "ચાર" } -> પાવ
        fraction { numerator: "ત્રણ" denominator: "ચાર" } -> ત્રણ પાવ
    
    Following Hindi fraction pattern with Gujarati equivalents:
    - "બટા" (bata) = separator between numerator and denominator
    - "અને" (ane) = "and" for mixed numbers
    - "ઋણાત્મક" (runatmak) = "negative"
    - Special cases: અડધું (half), પાવ (quarter)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        # Handle negative sign: use MINUS constant ("ઋણાત્મક")
        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", MINUS) + pynutil.delete(" "), 0, 1)

        # Integer part extraction
        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")

        # Numerator extraction
        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")

        # Denominator extraction
        denominator = pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        # Insert "બટા" (bata) separator between numerator and denominator
        insert_bata = pynutil.insert(" બટા ")

        # Insert "અને" (ane - "and") for mixed numbers
        insert_ane = pynutil.insert(" અને ")

        # Special cases (like Hindi)
        # 1/2 -> "અડધું" (half): numerator "એક" + denominator "બે" -> "અડધું"
        numerator_one_half = (
            pynutil.delete("numerator: \"એક\"") + pynutil.delete(" ")
            + pynini.cross("denominator: \"બે\"", "અડધું")
        )

        # 1/4 -> "પાવ" (quarter): numerator "એક" + denominator "ચાર" -> "પાવ"
        numerator_one_quarter = (
            pynutil.delete("numerator: \"એક\"") + pynutil.delete(" ")
            + pynini.cross("denominator: \"ચાર\"", "પાવ")
        )

        # 3/4 -> "ત્રણ પાવ" (three quarters): numerator "ત્રણ" + denominator "ચાર" -> "ત્રણ પાવ"
        three_quarters = (
            pynutil.delete("numerator: \"ત્રણ\"") + pynutil.delete(" ")
            + pynutil.insert("ત્રણ ")
            + pynini.cross("denominator: \"ચાર\"", "પાવ")
        )

        # Default: numerator બટા denominator (for all other fractions)
        fraction_default = numerator + insert_bata + denominator

        # Use priority union like Hindi to handle special cases first, then default
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

        # Add integer part with "અને" (and) for mixed numbers
        # Format: integer અને numerator બટા denominator
        self.graph = (
            optional_sign
            + pynini.closure(integer + insert_space + insert_ane, 0, 1)
            + fraction_graph
        ) | graph_quarter

        graph = self.graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
