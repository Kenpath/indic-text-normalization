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

from indic_text_normalization.text_normalization.ta.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_TA_DIGIT,
    NEMO_NON_BREAKING_SPACE,
    NEMO_SPACE,
    TO_LOWER,
    GraphFst,
    convert_space,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)
from indic_text_normalization.text_normalization.ta.utils import get_abs_path

# Convert Arabic digits (0-9) to Tamil digits (௦-௯)
arabic_to_tamil_digit = pynini.string_map([
    ("0", "௦"), ("1", "௧"), ("2", "௨"), ("3", "௩"), ("4", "௪"),
    ("5", "௫"), ("6", "௬"), ("7", "௭"), ("8", "௮"), ("9", "௯")
]).optimize()
arabic_to_tamil_number = pynini.closure(arabic_to_tamil_digit).optimize()


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, e.g.
        -12kg -> measure { negative: "true" cardinal { integer: "பன்னிரண்டு" } units: "கிலோகிராம்" }
        12.5kg -> measure { decimal { integer_part: "பன்னிரண்டு" fractional_part: "ஐந்து" } units: "கிலோகிராம்" }
        ௧௨ kg -> measure { cardinal { integer: "பன்னிரண்டு" } units: "கிலோகிராம்" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        self.deterministic = deterministic

        cardinal_graph = cardinal.final_graph

        # Support both Tamil and Arabic digits
        tamil_number_input = pynini.closure(NEMO_TA_DIGIT, 1)
        tamil_number_graph = pynini.compose(tamil_number_input, cardinal_graph).optimize()

        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_tamil_number @ cardinal_graph
        ).optimize()

        cardinal_graph_combined = tamil_number_graph | arabic_number_graph

        # Add range support (e.g., 2-3, 2x3)
        cardinal_with_range = cardinal_graph_combined | self.get_range(cardinal_graph_combined)

        # Load unit graph
        unit_graph = pynini.string_file(get_abs_path("data/measure/unit.tsv"))

        # Support lowercase unit names
        unit_graph |= pynini.compose(
            pynini.closure(TO_LOWER, 1) + (NEMO_ALPHA | TO_LOWER) + pynini.closure(NEMO_ALPHA | TO_LOWER),
            unit_graph,
        ).optimize()

        # Convert spaces to non-breaking spaces
        unit_graph = convert_space(unit_graph)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "),
            0,
            1,
        )

        # "per" unit support (e.g., km/h)
        graph_unit2 = (
            pynini.cross("/", "ஒரு") + delete_zero_or_one_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + unit_graph
        )

        optional_graph_unit2 = pynini.closure(
            delete_zero_or_one_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit2,
            0,
            1,
        )

        unit_component = (
            pynutil.insert('units: "') + (unit_graph + optional_graph_unit2 | graph_unit2) + pynutil.insert('"')
        )

        # Cardinal + unit (e.g., 12 kg)
        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + cardinal_with_range
            + pynutil.insert('"')
            + pynutil.insert(" } ")
            + delete_space
            + unit_component
        )

        # Decimal + unit (e.g., 12.5 kg)
        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + pynutil.insert(" } ")
            + delete_space
            + unit_component
        )

        # Unit graph standalone (e.g., /kg -> per kilogram)
        unit_graph_standalone = (
            pynutil.insert('cardinal { integer: "-" } units: "')
            + ((pynini.cross("/", "ஒரு") + delete_zero_or_one_space) | (pynini.accep("ஒரு") + pynutil.delete(" ")))
            + pynutil.insert(NEMO_NON_BREAKING_SPACE)
            + unit_graph
            + pynutil.insert('" preserve_order: true')
        )

        # Decimal dash alpha (e.g., 12.5-kg)
        decimal_dash_alpha = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynini.cross("-", "")
            + pynutil.insert(' } units: "')
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert('"')
        )

        # Decimal times (e.g., 12.5x)
        decimal_times = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(' } units: "')
            + (pynini.cross(pynini.union("x", "X"), "மடங்கு") | pynini.cross(pynini.union("*"), "மடங்கு"))
            + pynutil.insert('"')
        )

        # Symbol handling (x, X, *)
        symbol_graph = pynini.string_map([
            ("x", "மடங்கு"),
            ("X", "மடங்கு"),
            ("*", "மடங்கு"),
        ])

        # Cardinal with symbol (e.g., 2x3)
        graph_exceptions = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph_combined
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + symbol_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("} }")
            + insert_space
            + pynutil.insert("tokens { cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph_combined
            + pynutil.insert("\"")
        )

        # Combine all patterns
        final_graph = (
            pynutil.add_weight(subgraph_decimal, 0.1)
            | pynutil.add_weight(subgraph_cardinal, 0.1)
            | unit_graph_standalone
            | decimal_dash_alpha
            | decimal_times
            | pynutil.add_weight(graph_exceptions, 0.2)
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def get_range(self, cardinal: GraphFst):
        """
        Returns range forms for measure tagger, e.g. 2-3, 2x3, 2*2

        Args:
            cardinal: cardinal GraphFst
        """
        range_graph = cardinal + pynini.cross(pynini.union("-", " - "), " முதல் ") + cardinal

        for x in [" x ", "x"]:
            range_graph |= cardinal + pynini.cross(x, " மடங்கு ") + cardinal
            if not self.deterministic:
                range_graph |= cardinal + pynini.cross(x, " மடங்கு ") + pynini.closure(cardinal, 0, 1)

        for x in ["*", " * "]:
            range_graph |= cardinal + pynini.cross(x, " மடங்கு ") + cardinal

        return range_graph.optimize()

