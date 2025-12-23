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

import os
import pynini
from pynini.lib import pynutil

from indic_text_normalization.gu.graph_utils import (
    GU_DODH,
    GU_ADHI,
    GU_PONA,
    GU_SADA,
    GU_SAVA,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_GU_DIGIT,
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    NEMO_SPACE,
    TO_LOWER,
    GraphFst,
    convert_space,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)
from indic_text_normalization.gu.utils import get_abs_path

# Convert Arabic digits (0-9) to Gujarati digits (૦-૯)
arabic_to_gujarati_digit = pynini.string_map([
    ("0", "૦"), ("1", "૧"), ("2", "૨"), ("3", "૩"), ("4", "૪"),
    ("5", "૫"), ("6", "૬"), ("7", "૭"), ("8", "૮"), ("9", "૯")
]).optimize()
arabic_to_gujarati_number = pynini.closure(arabic_to_gujarati_digit).optimize()

GU_POINT_FIVE = ".૫"  # .5
GU_ONE_POINT_FIVE = "૧.૫"  # 1.5
GU_TWO_POINT_FIVE = "૨.૫"  # 2.5
GU_DECIMAL_25 = ".૨૫"  # .25
GU_DECIMAL_75 = ".૭૫"  # .75

digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
teens_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -૧૨kg -> measure { negative: "true" cardinal { integer: "બાર" } units: "કિલોગ્રામ" }
        -૧૨.૨kg -> measure { decimal { negative: "true"  integer_part: "બાર"  fractional_part: "બે"} units: "કિલોગ્રામ" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst (optional, for fraction support)
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst = None, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        self.deterministic = deterministic

        # Get cardinal graph with range support
        # Support both Gujarati and Arabic digits
        # Gujarati digits path
        gujarati_cardinal_graph_base = (
            cardinal.zero
            | cardinal.digit
            | cardinal.teens_and_ties
            | cardinal.graph_hundreds
            | cardinal.graph_thousands
            | cardinal.graph_ten_thousands
            | cardinal.graph_lakhs
            | cardinal.graph_ten_lakhs
        )
        
        # Arabic digits path - convert to Gujarati first, then compose with cardinal
        # Gujarati number input
        gujarati_number_input = pynini.closure(NEMO_GU_DIGIT, 1)
        gujarati_number_graph = pynini.compose(gujarati_number_input, gujarati_cardinal_graph_base).optimize()
        
        # Arabic number input - convert to Gujarati, then compose with cardinal
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_gujarati_number @ gujarati_cardinal_graph_base
        ).optimize()
        
        # Combined cardinal graph (supports both Gujarati and Arabic digits)
        cardinal_graph_base = gujarati_number_graph | arabic_number_graph
        
        # Add range support (e.g., 2-3, 2x3, 2*2)
        cardinal_graph = cardinal_graph_base | self.get_range(cardinal_graph_base)
        
        point = pynutil.delete(".")
        decimal_integers = pynutil.insert("integer_part: \"") + cardinal_graph_base + pynutil.insert("\"")
        decimal_graph = decimal_integers + point + insert_space + decimal.graph_fractional
        
        # Load unit graph with alternatives support
        unit_graph = pynini.string_file(get_abs_path("data/measure/unit.tsv"))
        # Check if unit_alternatives.tsv exists and load it
        unit_alternatives_path = get_abs_path("data/measure/unit_alternatives.tsv")
        if not deterministic and os.path.exists(unit_alternatives_path):
            unit_graph |= pynini.string_file(unit_alternatives_path)
        
        # Support lowercase unit names
        unit_graph |= pynini.compose(
            pynini.closure(TO_LOWER, 1) + (NEMO_ALPHA | TO_LOWER) + pynini.closure(NEMO_ALPHA | TO_LOWER),
            unit_graph,
        ).optimize()
        
        # Convert spaces to non-breaking spaces
        unit_graph = convert_space(unit_graph)

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
                (GU_POINT_FIVE, GU_SADA),
                (GU_ONE_POINT_FIVE, GU_DODH),
                (GU_TWO_POINT_FIVE, GU_ADHI),
            ]
        )
        quarter_graph = pynutil.insert("integer_part: \"") + quarter + pynutil.insert("\"")

        # Define unit handling with "per" support
        graph_unit2 = (
            pynini.cross("/", "પ્રતિ") + delete_zero_or_one_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + unit_graph
        )
        
        optional_graph_unit2 = pynini.closure(
            delete_zero_or_one_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit2,
            0,
            1,
        )
        
        # Unit singular and plural
        # Note: Gujarati units typically don't change for plural
        unit_singular = (
            pynutil.insert('units: "') + (unit_graph + optional_graph_unit2 | graph_unit2) + pynutil.insert('"')
        )
        unit_plural = unit_singular  # Gujarati units typically don't change for plural
        
        # Define the unit handling (keeping existing structure for quarterly units)
        unit = (
            pynutil.insert(NEMO_SPACE)
            + unit_singular
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
                ("x", "બાય"),
                ("X", "બાય"),
                ("*", "બાય"),
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

        dodh_adhi = pynini.string_map([(GU_ONE_POINT_FIVE, GU_DODH), (GU_TWO_POINT_FIVE, GU_ADHI)])
        dodh_adhi_graph = pynutil.insert("integer: \"") + dodh_adhi + pynutil.insert("\"")

        sava_numbers = cardinal_graph_base + pynini.cross(GU_DECIMAL_25, "")
        sava_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(GU_SAVA)
            + pynutil.insert(NEMO_SPACE)
            + sava_numbers
            + pynutil.insert("\"")
        )

        sada_numbers = cardinal_graph_base + pynini.cross(GU_POINT_FIVE, "")
        sada_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(GU_SADA)
            + pynutil.insert(NEMO_SPACE)
            + sada_numbers
            + pynutil.insert("\"")
        )

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(GU_DECIMAL_75, "")
        pona_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(GU_PONA)
            + pynutil.insert(NEMO_SPACE)
            + paune_numbers
            + pynutil.insert("\"")
        )

        graph_dodh_adhi = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + dodh_adhi_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_sava = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + sava_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_sada = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + sada_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_pona = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pona_graph
            + pynutil.insert(" }")
            + delete_space
            + units
        )

        # Cardinal with plural units (for numbers != 1)
        # Exclude "1" and "૧" from matching (they will be handled by singular pattern)
        exclude_one = (NEMO_SIGMA - "1" - "૧")
        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + pynini.compose(exclude_one, cardinal_graph)
            + delete_space
            + pynutil.insert('"')
            + pynutil.insert(" } ")
            + unit_plural
        )
        
        # Cardinal with singular unit (for number == 1)
        # Handle both ASCII "1" and Gujarati "૧"
        one_pattern = pynini.cross("1", "એક") | pynini.cross("૧", "એક")
        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + one_pattern
            + delete_space
            + pynutil.insert('"')
            + pynutil.insert(" } ")
            + unit_singular
        )
        
        # Keep existing graph_cardinal for backward compatibility
        graph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph_base
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
            + cardinal_graph_base
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
            + cardinal_graph_base
            + pynutil.insert("\"")
        )

        # Additional patterns
        # Unit graph (standalone "per" unit)
        unit_graph_standalone = (
            pynutil.insert('cardinal { integer: "-" } units: "')
            + ((pynini.cross("/", "પ્રતિ") + delete_zero_or_one_space) | (pynini.accep("પ્રતિ") + pynutil.delete(" ")))
            + pynutil.insert(NEMO_NON_BREAKING_SPACE)
            + unit_graph
            + pynutil.insert('" preserve_order: true')
        )
        
        # Decimal dash alpha (e.g., "12.5-kg")
        decimal_dash_alpha = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynini.cross("-", "")
            + pynutil.insert(' } units: "')
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert('"')
        )
        
        # Decimal times (e.g., "12.5x")
        decimal_times = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(' } units: "')
            + (pynini.cross(pynini.union("x", "X"), "x") | pynini.cross(pynini.union("x", "X"), " બાય"))
            + pynutil.insert('"')
        )
        
        # Alpha dash decimal (e.g., "kg-12.5")
        alpha_dash_decimal = (
            pynutil.insert('units: "')
            + pynini.closure(NEMO_ALPHA, 1)
            + pynini.accep("-")
            + pynutil.insert('"')
            + pynutil.insert(" decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" } preserve_order: true")
        )
        
        # Fraction support
        subgraph_fraction = None
        if fraction is not None:
            subgraph_fraction = (
                pynutil.insert("fraction { ") + fraction.graph + delete_space + pynutil.insert(" } ") + unit_plural
            )
        
        # Math operations support
        math_operations_path = get_abs_path("data/math_operations.tsv")
        math = None
        try:
            if os.path.exists(math_operations_path):
                math_operations = pynini.string_file(math_operations_path)
                delimiter = pynini.accep(" ") | pynutil.insert(" ")
                
                math_expr = (
                    (cardinal_graph | NEMO_ALPHA)
                    + delimiter
                    + math_operations
                    + (delimiter | NEMO_ALPHA)
                    + cardinal_graph
                    + delimiter
                    + pynini.cross("=", "બરાબર")
                    + delimiter
                    + (cardinal_graph | NEMO_ALPHA)
                )
                
                math_expr |= (
                    (cardinal_graph | NEMO_ALPHA)
                    + delimiter
                    + pynini.cross("=", "બરાબર")
                    + delimiter
                    + (cardinal_graph | NEMO_ALPHA)
                    + delimiter
                    + math_operations
                    + delimiter
                    + cardinal_graph
                )
                
                math = (
                    pynutil.insert('units: "math" cardinal { integer: "') + math_expr + pynutil.insert('" } preserve_order: true')
                )
        except Exception:
            math = None
        
        # Build final graph
        graph = (
            pynutil.add_weight(graph_decimal, 0.1)
            | pynutil.add_weight(subgraph_cardinal, 0.1)
            | pynutil.add_weight(graph_cardinal, 0.1)
            | pynutil.add_weight(graph_exceptions, 0.1)
            | pynutil.add_weight(graph_dodh_adhi, -0.2)
            | pynutil.add_weight(graph_sava, -0.1)
            | pynutil.add_weight(graph_sada, -0.1)
            | pynutil.add_weight(graph_pona, -0.5)
        )
        
        # Add additional patterns
        if unit_graph_standalone is not None:
            graph |= unit_graph_standalone
        if decimal_dash_alpha is not None:
            graph |= decimal_dash_alpha
        if decimal_times is not None:
            graph |= decimal_times
        if alpha_dash_decimal is not None:
            graph |= alpha_dash_decimal
        if subgraph_fraction is not None:
            graph |= subgraph_fraction
        if math is not None:
            graph |= math
        
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
    
    def get_range(self, cardinal: GraphFst):
        """
        Returns range forms for measure tagger, e.g. 2-3, 2x3, 2*2

        Args:
            cardinal: cardinal GraphFst
        """
        range_graph = cardinal + pynini.cross(pynini.union("-", " - "), " થી ") + cardinal

        for x in [" x ", "x"]:
            range_graph |= cardinal + pynini.cross(x, " બાય ") + cardinal
            if not self.deterministic:
                range_graph |= cardinal + pynini.cross(x, " બાય ") + pynini.closure(cardinal, 0, 1)

        for x in ["*", " * "]:
            range_graph |= cardinal + pynini.cross(x, " બાય ") + cardinal
        return range_graph.optimize()
