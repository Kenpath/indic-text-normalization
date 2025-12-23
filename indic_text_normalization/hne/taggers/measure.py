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

from indic_text_normalization.hne.graph_utils import (
    CG_DEDH,
    CG_DHAI,
    CG_PAUNE,
    CG_SADHE,
    CG_SAVVA,
    NEMO_DIGIT,
    NEMO_CG_DIGIT,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from indic_text_normalization.hne.utils import get_abs_path

# Convert Arabic digits (0-9) to Chhattisgarhi digits (०-९)
arabic_to_cg_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_cg_number = pynini.closure(arabic_to_cg_digit).optimize()

CG_POINT_FIVE = ".५"  # .5
CG_ONE_POINT_FIVE = "१.५"  # 1.5
CG_TWO_POINT_FIVE = "२.५"  # 2.5
CG_DECIMAL_25 = ".२५"  # .25
CG_DECIMAL_75 = ".७५"  # .75

digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
teens_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -१२kg -> measure { negative: "true" cardinal { integer: "बारह" } units: "किलोग्राम" }
        -१२.२kg -> measure { decimal { negative: "true"  integer_part: "बारह"  fractional_part: "दो"} units: "किलोग्राम" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        # Support both Chhattisgarhi and Arabic digits
        cg_number_input = pynini.closure(NEMO_CG_DIGIT, 1)
        cg_cardinal_graph = pynini.compose(cg_number_input, cardinal.final_graph).optimize()

        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_cardinal_graph = pynini.compose(
            arabic_number_input,
            arabic_to_cg_number @ cardinal.final_graph
        ).optimize()

        cardinal_graph_combined = cg_cardinal_graph | arabic_cardinal_graph

        # Keep the original cardinal_graph for internal use (quarterly units, etc.)
        cardinal_graph = (
            cardinal.zero
            | cardinal.digit
            | cardinal.teens_and_ties
            | cardinal.graph_hundreds
            | cardinal.graph_thousands
            | cardinal.graph_ten_thousands
            | cardinal.graph_lakhs
            | cardinal.graph_ten_lakhs
        )
        point = pynutil.delete(".")
        decimal_integers = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        decimal_graph = decimal_integers + point + insert_space + decimal.graph_fractional
        unit_graph = pynini.string_file(get_abs_path("data/measure/unit.tsv"))

        # Load quarterly units from separate files: map (FST) and list (FSA)
        quarterly_units_map = pynini.string_file(get_abs_path("data/measure/quarterly_units_map.tsv"))
        quarterly_units_list = pynini.string_file(get_abs_path("data/measure/quarterly_units_list.tsv"))
        quarterly_units_graph = pynini.union(quarterly_units_map, quarterly_units_list)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Define the quarterly measurements
        quarter = pynini.string_map(
            [
                (CG_POINT_FIVE, CG_SADHE),
                (CG_ONE_POINT_FIVE, CG_DEDH),
                (CG_TWO_POINT_FIVE, CG_DHAI),
            ]
        )
        quarter_graph = pynutil.insert("integer_part: \"") + quarter + pynutil.insert("\"")

        # Define the unit handling
        unit = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + unit_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )
        units = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + quarterly_units_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        # Handling symbols like x, X, *
        symbol_graph = pynini.string_map(
            [
                ("x", "बाई"),
                ("X", "बाई"),
                ("*", "बाई"),
            ]
        )

        graph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_space
            + unit
        )

        dedh_dhai = pynini.string_map([(CG_ONE_POINT_FIVE, CG_DEDH), (CG_TWO_POINT_FIVE, CG_DHAI)])
        dedh_dhai_graph = pynutil.insert("integer: \"") + dedh_dhai + pynutil.insert("\"")

        savva_numbers = cardinal_graph + pynini.cross(CG_DECIMAL_25, "")
        savva_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(CG_SAVVA)
            + pynutil.insert(NEMO_SPACE)
            + savva_numbers
            + pynutil.insert("\"")
        )

        sadhe_numbers = cardinal_graph + pynini.cross(CG_POINT_FIVE, "")
        sadhe_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(CG_SADHE)
            + pynutil.insert(NEMO_SPACE)
            + sadhe_numbers
            + pynutil.insert("\"")
        )

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(CG_DECIMAL_75, "")
        paune_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(CG_PAUNE)
            + pynutil.insert(NEMO_SPACE)
            + paune_numbers
            + pynutil.insert("\"")
        )

        graph_dedh_dhai = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + dedh_dhai_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_savva = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + savva_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_sadhe = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + sadhe_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_paune = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + paune_graph
            + pynutil.insert(" }")
            + delete_space
            + units
        )

        graph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph_combined
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + unit
        )

        # Handling cardinal clubbed with symbol as single token
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

        graph = (
            pynutil.add_weight(graph_decimal, 0.1)
            | pynutil.add_weight(graph_cardinal, 0.1)
            | pynutil.add_weight(graph_exceptions, 0.1)
            | pynutil.add_weight(graph_dedh_dhai, -0.2)
            | pynutil.add_weight(graph_savva, -0.1)
            | pynutil.add_weight(graph_sadhe, -0.1)
            | pynutil.add_weight(graph_paune, -0.5)
        )
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
