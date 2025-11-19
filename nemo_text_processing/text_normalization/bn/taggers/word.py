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
    MIN_NEG_WEIGHT,
    NEMO_NOT_SPACE,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.bn.taggers.punctuation import PunctuationFst


class WordFst(GraphFst):
    """
    Finite state transducer for classifying Bengali words.
        e.g. সোনা -> tokens { name: "সোনা" }

    Args:
        punctuation: PunctuationFst
        deterministic: if True will provide a single transduction option,
            for False multiple transductions are generated (used for audio-based normalization)
    """

    def __init__(self, punctuation: PunctuationFst, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)

        # Define Bengali characters and symbols using pynini.union
        BENGALI_CHAR = pynini.union(
            *[chr(i) for i in range(0x0980, 0x0983 + 1)],  # Bengali vowels and consonants
            *[chr(i) for i in range(0x0985, 0x099E + 1)],  # More Bengali characters
            *[chr(i) for i in range(0x09E0, 0x09EF + 1)],  # Bengali digits
            *[chr(i) for i in range(0x09BE, 0x09CD + 1)],  # Bengali diacritics
        ).optimize()

        # Include punctuation in the graph
        punct = punctuation.graph
        default_graph = pynini.closure(pynini.difference(NEMO_NOT_SPACE, punct.project("input")), 1)
        symbols_to_exclude = (pynini.union("$", "€", "₩", "£", "¥", "#", "%") | punct).optimize()

        # Use BENGALI_CHAR in the graph
        graph = pynini.closure(pynini.difference(BENGALI_CHAR, symbols_to_exclude), 1)
        graph = pynutil.add_weight(graph, MIN_NEG_WEIGHT) | default_graph

        # Ensure no spaces around punctuation
        graph = pynini.closure(graph + pynini.closure(punct + graph, 0, 1))

        self.graph = convert_space(graph)
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()

