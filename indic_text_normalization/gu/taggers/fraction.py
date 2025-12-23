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
from pynini.lib import pynutil

from indic_text_normalization.gu.graph_utils import GraphFst, NEMO_SPACE


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions, e.g.
        ૩/૪ -> fraction { numerator: "ત્રણ" denominator: "ચાર" }
        ૧૨ ૩/૪ -> fraction { integer_part: "બાર" numerator: "ત્રણ" denominator: "ચાર" }
        3/4 -> fraction { numerator: "ત્રણ" denominator: "ચાર" }
        -૧/૨ -> fraction { negative: "true" numerator: "એક" denominator: "બે" }
    
    Following Hindi fraction tagger pattern.
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        # Use cardinal.final_graph like Hindi uses cardinal.final_graph
        cardinal_graph = cardinal.final_graph

        # Handle negative sign
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + pynutil.insert(NEMO_SPACE), 0, 1
        )

        integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        numerator = (
            pynutil.insert("numerator: \"") 
            + cardinal_graph 
            + (pynini.cross("/", "\" ") | pynini.cross(" / ", "\" "))
        )
        denominator = pynutil.insert("denominator: \"") + cardinal_graph + pynutil.insert("\"")

        # Basic fraction: [integer ] numerator/denominator
        graph = pynini.closure(integer + pynini.accep(" "), 0, 1) + (numerator + denominator)
        graph = optional_graph_negative + graph

        self.graph = graph
        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
