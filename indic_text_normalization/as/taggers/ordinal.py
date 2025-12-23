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

from ..graph_utils import GraphFst
from ..taggers.cardinal import CardinalFst
from ..utils import get_abs_path


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
        # Only match non-empty suffixes (exclude empty string)
        non_empty_suffixes = pynini.difference(suffixes_list, pynini.accep("")).optimize()
        suffixes_fst = pynini.union(non_empty_suffixes, suffixes_map)
        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))

        # Ordinals: cardinal number + ordinal suffix
        # e.g., १० + वाँ -> दस + वाँ -> दसवाँ
        # The cardinal.final_graph handles both Hindi (०-९) and Arabic (0-9) digits
        graph = cardinal.final_graph + suffixes_fst
        exceptions = pynutil.add_weight(exceptions, -0.1)
        graph = pynini.union(exceptions, graph)

        # Store graph before tokenization (needed for serial tagger)
        self.graph = graph.optimize()

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
