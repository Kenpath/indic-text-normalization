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

major_minor_currencies = {
    "రూపాయి": "పైస",
    "పౌండ్": "పెన్స్",
    "వాన్": "జియోన్",
    "డాలర్": "సెంట్",
    "లీరా": "కురుస్",
    "టకా": "పైస",
    "యెన్": "సెన్",
    "నైరా": "కొబొ",
    "యూరో": "సెంట్",
}
from indic_text_normalization.text_normalization.te.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "పన్నెండు" currency_maj: "రూపాయి" } -> పన్నెండు రూపాయి
        money { integer_part: "పన్నెండు" currency_maj: "రూపాయి" fractional_part: "యాభై" currency_min: "centiles" } -> పన్నెండు రూపాయి యాభై పైస
        money { currency_maj: "రూపాయి" integer_part: "సున్నా" fractional_part: "యాభై" currency_min: "centiles" } -> యాభై పైస

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        currency_major = pynutil.delete('currency_maj: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        integer_part = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        fractional_part = (
            pynutil.delete('fractional_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )

        # Handles major denominations only
        graph_major_only = integer_part + pynini.accep(NEMO_SPACE) + currency_major

        # Handles both major and minor denominations
        major_minor_graphs = []

        # Handles minor denominations only
        minor_graphs = []

        # Logic for handling minor denominations
        for major, minor in major_minor_currencies.items():
            graph_major = pynutil.delete('currency_maj: "') + pynini.accep(major) + pynutil.delete('"')
            graph_minor = pynutil.delete('currency_min: "') + pynini.cross("centiles", minor) + pynutil.delete('"')
            graph_major_minor_partial = (
                integer_part
                + pynini.accep(NEMO_SPACE)
                + graph_major
                + pynini.accep(NEMO_SPACE)
                + fractional_part
                + pynini.accep(NEMO_SPACE)
                + graph_minor
            )
            major_minor_graphs.append(graph_major_minor_partial)

            graph_minor_partial = (
                pynutil.delete('integer_part: "సున్నా"')
                + pynutil.delete(NEMO_SPACE)
                + pynutil.delete('currency_maj: "')
                + pynutil.delete(major)
                + pynutil.delete('"')
                + pynutil.delete(NEMO_SPACE)
                + fractional_part
                + pynini.accep(NEMO_SPACE)
                + graph_minor
            )
            minor_graphs.append(graph_minor_partial)

        graph_major_minor = pynini.union(*major_minor_graphs)
        graph_minor_only = pynini.union(*minor_graphs)

        graph = graph_major_only | graph_major_minor | pynutil.add_weight(graph_minor_only, -0.1)

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

