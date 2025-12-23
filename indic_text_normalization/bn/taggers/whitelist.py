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

from indic_text_normalization.bn.graph_utils import GraphFst, convert_space


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        "ডঃ" -> tokens { name: "ডক্টর" }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        input_file: path to a file with whitelist replacements
    """

    def __init__(self, input_case: str, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        if input_file:
            whitelist = pynini.string_file(input_file).optimize()
            graph = pynutil.insert("name: \"") + convert_space(whitelist) + pynutil.insert("\"")
            self.fst = self.add_tokens(graph).optimize()
        else:
            self.fst = pynini.cdrewrite(pynini.cross("", ""), "", "", pynini.closure(pynini.union("a", "z"))).optimize()

