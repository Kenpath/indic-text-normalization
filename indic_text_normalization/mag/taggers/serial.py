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

from indic_text_normalization.mag.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_MAG_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)
from indic_text_normalization.mag.taggers.cardinal import arabic_to_magadhi_digit
from indic_text_normalization.mag.utils import get_abs_path, load_labels


class SerialFst(GraphFst):
    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="integer", kind="classify", deterministic=deterministic)

        any_digit = NEMO_DIGIT | NEMO_MAG_DIGIT
        single_magadhi_digit = pynini.compose(arabic_to_magadhi_digit, cardinal.digit | cardinal.zero) | (
            cardinal.digit | cardinal.zero
        )
        digit_by_digit = single_magadhi_digit + pynini.closure(pynutil.insert(" ") + single_magadhi_digit)

        if deterministic:
            num_graph = pynini.compose(any_digit ** (1, ...), digit_by_digit).optimize()
            num_graph |= pynini.compose(
                (pynini.accep("0") | pynini.accep("०")) + pynini.closure(any_digit),
                digit_by_digit,
            ).optimize()
        else:
            num_graph = cardinal.final_graph

        symbols_graph = pynini.string_file(get_abs_path("data/whitelist/abbreviations.tsv")).optimize() | pynini.cross(
            "#", "hash"
        )
        num_graph |= symbols_graph

        if not self.deterministic and not lm:
            num_graph |= cardinal.final_graph
            num_graph |= pynutil.add_weight(any_digit**2 @ cardinal.final_graph, weight=0.0001)

        symbols = [x[0] for x in load_labels(get_abs_path("data/whitelist/abbreviations.tsv")) if len(x) > 0]
        symbols = pynini.union(*symbols)
        digit_symbol = any_digit | symbols

        graph_with_space = pynini.compose(
            pynini.cdrewrite(pynutil.insert(" "), NEMO_ALPHA | symbols, digit_symbol, NEMO_SIGMA),
            pynini.cdrewrite(pynutil.insert(" "), digit_symbol, NEMO_ALPHA | symbols, NEMO_SIGMA),
        )

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
        serial_graph |= num_graph + delimiter + num_graph + delimiter + num_graph + pynini.closure(delimiter + num_graph)
        serial_graph |= pynini.compose(NEMO_SIGMA + symbols + NEMO_SIGMA, num_graph + delimiter + num_graph)

        serial_graph = pynutil.add_weight(serial_graph, 0.0001)
        serial_graph |= (
            pynini.closure(NEMO_NOT_SPACE, 1)
            + (pynini.cross("^2", " वर्ग") | pynini.cross("^3", " घन")).optimize()
        )

        serial_graph = (
            pynini.closure((serial_graph | num_graph | alphas) + delimiter)
            + serial_graph
            + pynini.closure(delimiter + (serial_graph | num_graph | alphas))
        )

        serial_graph |= pynini.compose(graph_with_space, serial_graph.optimize()).optimize()
        serial_graph = pynini.compose(pynini.closure(NEMO_NOT_SPACE, 2), serial_graph).optimize()
        serial_graph = pynini.compose(
            pynini.difference(
                NEMO_SIGMA, pynini.closure(NEMO_ALPHA, 1) + pynini.accep("/") + pynini.closure(NEMO_ALPHA, 1)
            ),
            serial_graph,
        )

        self.graph = serial_graph.optimize()
        graph = pynutil.insert("name: \"") + convert_space(self.graph).optimize() + pynutil.insert("\"")
        self.fst = graph.optimize()
