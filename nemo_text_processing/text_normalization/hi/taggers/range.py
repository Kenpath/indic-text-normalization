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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_DIGIT, GraphFst, convert_space


class RangeFst(GraphFst):
    """
    This class is a composite class of two other class instances

    Args:
        time: composed tagger and verbalizer
        date: composed tagger and verbalizer
        cardinal: tagger
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
        lm: whether to use for hybrid LM
    """

    def __init__(
        self,
        time: GraphFst,
        date: GraphFst,
        cardinal: GraphFst,
        deterministic: bool = True,
        lm: bool = False,
    ):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)

        approx = pynini.cross("~", "लगभग")

        # TIME
        time_graph = time + delete_space + pynini.cross("-", " से ") + delete_space + time
        self.graph = time_graph | (approx + time)

        # Use final_graph for cardinal numbers
        cardinal_graph = cardinal.final_graph
        # YEAR
        date_year_four_digit = (NEMO_DIGIT**4 + pynini.closure(pynini.accep("s"), 0, 1)) @ date
        date_year_two_digit = (NEMO_DIGIT**2 + pynini.closure(pynini.accep("s"), 0, 1)) @ date
        year_to_year_graph = (
            date_year_four_digit
            + delete_space
            + pynini.cross("-", " से ")
            + delete_space
            + (date_year_four_digit | date_year_two_digit | (NEMO_DIGIT**2 @ cardinal_graph))
        )
        mid_year_graph = pynini.accep("mid") + pynini.cross("-", " ") + (date_year_four_digit | date_year_two_digit)

        self.graph |= year_to_year_graph
        self.graph |= mid_year_graph

        # ADDITION
        range_graph = cardinal_graph + pynini.closure(pynini.cross("+", " प्लस ") + cardinal_graph, 1)
        range_graph |= cardinal_graph + pynini.closure(pynini.cross(" + ", " प्लस ") + cardinal_graph, 1)
        range_graph |= approx + cardinal_graph
        range_graph |= cardinal_graph + (pynini.cross("...", " ... ") | pynini.accep(" ... ")) + cardinal_graph

        if not deterministic or lm:
            # cardinal ----
            cardinal_to_cardinal_graph = (
                cardinal_graph + delete_space + pynini.cross("-", pynini.union(" से ", " माइनस ")) + delete_space + cardinal_graph
            )

            range_graph |= cardinal_to_cardinal_graph | (
                cardinal_graph + delete_space + pynini.cross(":", " से ") + delete_space + cardinal_graph
            )

            # MULTIPLY
            for x in [" x ", "x"]:
                range_graph |= cardinal_graph + pynini.cross(x, pynini.union(" बाई ", " गुणा ")) + cardinal_graph

            # 40x -> "40 times" ("40 x" cases is covered in serial)
            for x in [" x", "x"]:
                range_graph |= cardinal_graph + pynini.cross(x, " गुणा")

                # 5x to 7x-> five to seven x/times
                range_graph |= (
                    cardinal_graph
                    + pynutil.delete(x)
                    + pynini.union(" से ", "-", " - ")
                    + cardinal_graph
                    + pynini.cross(x, pynini.union(" x", " गुणा"))
                )

            for x in ["*", " * "]:
                range_graph |= cardinal_graph + pynini.closure(pynini.cross(x, " गुणा ") + cardinal_graph, 1)

            # supports "No. 12" -> "Number 12"
            range_graph |= (
                (pynini.cross(pynini.union("NO", "No"), "नंबर") | pynini.cross("no", "नंबर"))
                + pynini.closure(pynini.union(". ", " "), 0, 1)
                + cardinal_graph
            )

            for x in ["/", " / "]:
                range_graph |= cardinal_graph + pynini.closure(pynini.cross(x, " भाग ") + cardinal_graph, 1)

            # 10% to 20% -> ten to twenty percent
            range_graph |= (
                cardinal_graph
                + pynini.closure(pynini.cross("%", " प्रतिशत") | pynutil.delete("%"), 0, 1)
                + pynini.union(" से ", "-", " - ")
                + cardinal_graph
                + pynini.cross("%", " प्रतिशत")
            )

        self.graph |= range_graph

        self.graph = self.graph.optimize()
        graph = pynutil.insert("name: \"") + convert_space(self.graph).optimize() + pynutil.insert("\"")
        self.fst = graph.optimize()

