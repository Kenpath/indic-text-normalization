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

from nemo_text_processing.text_normalization.te.graph_utils import GraphFst
from nemo_text_processing.text_normalization.te.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.te.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu ordinals, e.g.
        ౧౦వ -> ordinal { integer: "పదవ" }
        ౨౧వ -> ordinal { integer: "ఇరవై ఒకటి వ" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        from nemo_text_processing.text_normalization.te.graph_utils import NEMO_TE_DIGIT, NEMO_DIGIT
        
        suffixes_list = pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv"))
        suffixes_map = pynini.string_file(get_abs_path("data/ordinal/suffixes_map.tsv"))
        # Only match non-empty suffixes (exclude empty string)
        non_empty_suffixes = pynini.difference(suffixes_list, pynini.accep("")).optimize()
        suffixes_fst = pynini.union(non_empty_suffixes, suffixes_map)
        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))

        # Ordinals should only match when there's a clear suffix (like "వ") after the number
        # The suffix itself acts as a boundary, so we don't need additional word boundary logic
        # The tokenizer will handle word boundaries correctly
        graph = cardinal.final_graph + suffixes_fst
        exceptions = pynutil.add_weight(exceptions, -0.1)
        graph = pynini.union(exceptions, graph)

        final_graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()

