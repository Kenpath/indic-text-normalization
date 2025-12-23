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
from pynini.examples import plurals
from pynini.lib import pynutil

from indic_text_normalization.text_normalization.hi.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)
from indic_text_normalization.text_normalization.hi.utils import get_abs_path, load_labels


class SerialFst(GraphFst):
    """
    This class is a composite class of two other class instances

    Args:
        cardinal: tagger
        ordinal: tagger
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
        lm: whether to use for hybrid LM
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="integer", kind="classify", deterministic=deterministic)

        """
        Finite state transducer for classifying serial (handles only cases without delimiters,
        values with delimiters are handled by default).
            The serial is a combination of digits, letters and dashes, e.g.:
            c325b -> tokens { cardinal { integer: "c तीन दो पाँच b" } }
        """
        if deterministic:
            # For Hindi, use final_graph for numbers
            num_graph = pynini.compose(NEMO_DIGIT ** (6, ...), cardinal.final_graph).optimize()
            num_graph |= pynini.compose(NEMO_DIGIT ** (1, 5), cardinal.final_graph).optimize()
            # to handle numbers starting with zero
            num_graph |= pynini.compose(
                pynini.accep("0") + pynini.closure(NEMO_DIGIT), cardinal.final_graph
            ).optimize()
        else:
            num_graph = cardinal.final_graph

        # TODO: "#" doesn't work from the file
        symbols_graph = pynini.string_file(get_abs_path("data/whitelist/abbreviations.tsv")).optimize() | pynini.cross(
            "#", "hash"
        )
        num_graph |= symbols_graph

        if not self.deterministic and not lm:
            num_graph |= cardinal.final_graph
            # also allow double digits to be pronounced as integer in serial number
            num_graph |= pynutil.add_weight(
                NEMO_DIGIT**2 @ cardinal.final_graph, weight=0.0001
            )

        # add space between letter and digit/symbol
        symbols = [x[0] for x in load_labels(get_abs_path("data/whitelist/abbreviations.tsv"))]
        symbols = pynini.union(*symbols)
        digit_symbol = NEMO_DIGIT | symbols

        graph_with_space = pynini.compose(
            pynini.cdrewrite(pynutil.insert(" "), NEMO_ALPHA | symbols, digit_symbol, NEMO_SIGMA),
            pynini.cdrewrite(pynutil.insert(" "), digit_symbol, NEMO_ALPHA | symbols, NEMO_SIGMA),
        )

        # serial graph with delimiter
        delimiter = pynini.accep("-") | pynini.accep("/") | pynini.accep(" ")
        if not deterministic:
            delimiter |= pynini.cross("-", " dash ") | pynini.cross("/", " slash ")

        alphas = pynini.closure(NEMO_ALPHA, 1)
        letter_num = alphas + delimiter + num_graph
        num_letter = pynini.closure(num_graph + delimiter, 1) + alphas
        next_alpha_or_num = pynini.closure(delimiter + (alphas | num_graph))
        next_alpha_or_num |= pynini.closure(
            delimiter
            + num_graph
            + plurals._priority_union(pynini.accep(" "), pynutil.insert(" "), NEMO_SIGMA).optimize()
            + alphas
        )

        serial_graph = letter_num + next_alpha_or_num
        serial_graph |= num_letter + next_alpha_or_num
        # numbers only with 2+ delimiters
        serial_graph |= (
            num_graph + delimiter + num_graph + delimiter + num_graph + pynini.closure(delimiter + num_graph)
        )
        # 2+ symbols
        serial_graph |= pynini.compose(NEMO_SIGMA + symbols + NEMO_SIGMA, num_graph + delimiter + num_graph)

        # exclude ordinal numbers from serial options
        # Note: In Hindi, ordinals use Hindi digits (०-९) while serials use ASCII digits (0-9)
        # So there's minimal overlap. The exclusion is attempted but skipped if it fails.
        try:
            ordinal_input = ordinal_input.rmepsilon()
            ordinal_input = pynini.determinize(ordinal_input)
            ordinal_input = pynini.arcmap(ordinal_input, map_type="rmweight")
            ordinal_input = pynini.project(ordinal.graph, "input")
            if ordinal_input.num_states() > 0:
                ordinal_exclusion = pynini.difference(NEMO_SIGMA, ordinal_input)
                serial_graph = pynini.compose(ordinal_exclusion, serial_graph).optimize()
        except:
            # Skip exclusion if difference operation fails (alphabet mismatch)
            # This is acceptable since Hindi ordinals and serials use different digit systems
            pass

        serial_graph = pynutil.add_weight(serial_graph, 0.0001)
        serial_graph |= (
            pynini.closure(NEMO_NOT_SPACE, 1)
            + (pynini.cross("^2", " वर्ग") | pynini.cross("^3", " घन")).optimize()
        )

        # at least one serial graph with alpha numeric value and optional additional serial/num/alpha values
        serial_graph = (
            pynini.closure((serial_graph | num_graph | alphas) + delimiter)
            + serial_graph
            + pynini.closure(delimiter + (serial_graph | num_graph | alphas))
        )

        serial_graph |= pynini.compose(graph_with_space, serial_graph.optimize()).optimize()
        serial_graph = pynini.compose(pynini.closure(NEMO_NOT_SPACE, 2), serial_graph).optimize()

        # this is not to verbolize "/" as "slash" in cases like "import/export"
        serial_graph = pynini.compose(
            pynini.difference(
                NEMO_SIGMA, pynini.closure(NEMO_ALPHA, 1) + pynini.accep("/") + pynini.closure(NEMO_ALPHA, 1)
            ),
            serial_graph,
        )

        self.graph = serial_graph.optimize()
        graph = pynutil.insert("name: \"") + convert_space(self.graph).optimize() + pynutil.insert("\"")
        self.fst = graph.optimize()

