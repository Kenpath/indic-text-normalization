# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, NEMO_HI_DIGIT
from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Hindi ordinals, e.g.
        १०वां -> ordinal { integer: "दसवां" }
        २१वीं -> ordinal { integer: "इक्कीसवीं" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        suffixes_list = pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv"))
        suffixes_map = pynini.string_file(get_abs_path("data/ordinal/suffixes_map.tsv"))
        suffixes_fst = pynini.union(suffixes_list, suffixes_map)
        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))

        # Build graph similar to English: input pattern @ cardinal graph
        # Pattern: Hindi digits followed by ordinal suffixes (like English: digits + "th")
        hindi_digits = pynini.closure(NEMO_HI_DIGIT, 1)
        cardinal_graph = cardinal.final_graph
        
        # Create pattern: (Hindi digits + suffix) -> (Hindi digits) by deleting suffix
        # Then compose with cardinal to get: (Hindi digits + suffix) -> (Hindi words)
        # Then add suffix to output: (Hindi digits + suffix) -> (Hindi words + suffix)
        ordinal_pattern = hindi_digits + pynutil.delete(suffixes_fst)
        # Compose with cardinal: (digits + suffix) -> digits -> words
        graph = ordinal_pattern @ cardinal_graph
        # Add suffix to output: words -> words + suffix
        graph = graph + suffixes_fst
        
        exceptions = pynutil.add_weight(exceptions, -0.1)
        graph = pynini.union(exceptions, graph)

        # Store graph before tokenization (needed for serial tagger)
        # This graph has Hindi digits + suffixes on input, Hindi words + suffix on output
        self.graph = graph.optimize()

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
