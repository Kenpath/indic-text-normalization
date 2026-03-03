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

from indic_text_normalization.te.graph_utils import GraphFst, NEMO_DIGIT, NEMO_TE_DIGIT, insert_space
from indic_text_normalization.te.utils import get_abs_path

# Convert Arabic digits (0-9) to Telugu digits (౦-౯)
arabic_to_telugu_digit = pynini.string_map([
    ("0", "౦"), ("1", "౧"), ("2", "౨"), ("3", "౩"), ("4", "౪"),
    ("5", "౫"), ("6", "౬"), ("7", "౭"), ("8", "౮"), ("9", "౯")
]).optimize()
arabic_to_telugu_number = pynini.closure(arabic_to_telugu_digit).optimize()

# Create a graph that deletes commas from digit sequences
# This handles Indian number format where commas are separators (e.g., 1,000,001 or ౧,౦౦౦,౦౦౧)
any_digit = pynini.union(NEMO_DIGIT, NEMO_TE_DIGIT)
# Pattern: digit (comma? digit)* - accepts digits with optional commas, deletes commas
# This creates a transducer: input (with commas) -> output (without commas)
delete_commas = (
    any_digit
    + pynini.closure(pynini.closure(pynutil.delete(","), 0, 1) + any_digit)
).optimize()

# Indian comma format pattern (e.g. 12,34,567)
comma = pynini.accep(",")
three_digits = any_digit + any_digit + any_digit
indian_comma_pattern = (
    pynini.closure(any_digit, 1, 2)
    + pynini.closure(comma + any_digit + any_digit, 1)
    + pynini.closure(comma + three_digits, 0, 1)
).optimize()


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        -౨౩ -> cardinal { negative: "true"  integer: "ఇరవై మూడు" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

        self.digit = digit
        self.zero = zero
        self.teens_and_ties = teens_and_ties

        def create_graph_suffix(digit_graph, suffix, zeros_counts):
            zero = pynutil.add_weight(pynutil.delete("౦"), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix

            return digit_graph + (zero**zeros_counts) + suffix

        def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
            insert_space = pynutil.insert(" ")
            zero = pynutil.add_weight(pynutil.delete("౦"), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix + insert_space + sub_graph

            return digit_graph + suffix + (zero**zeros_counts) + insert_space + sub_graph

        # Hundred graph
        suffix_hundreds = pynutil.insert(" వంద")
        graph_hundreds = create_graph_suffix(digit, suffix_hundreds, 2)
        graph_hundreds |= create_larger_number_graph(digit, suffix_hundreds, 1, digit)
        graph_hundreds |= create_larger_number_graph(digit, suffix_hundreds, 0, teens_ties)
        graph_hundreds.optimize()
        self.graph_hundreds = graph_hundreds

        # Transducer for eleven hundred -> 1100 or twenty one hundred eleven -> 2111
        graph_hundreds_as_thousand = create_graph_suffix(teens_and_ties, suffix_hundreds, 2)
        graph_hundreds_as_thousand |= create_larger_number_graph(teens_and_ties, suffix_hundreds, 1, digit)
        graph_hundreds_as_thousand |= create_larger_number_graph(teens_and_ties, suffix_hundreds, 0, teens_ties)
        self.graph_hundreds_as_thousand = graph_hundreds_as_thousand

        # Thousands and Ten thousands graph
        suffix_thousands = pynutil.insert(" వెయ్యి")
        graph_thousands = create_graph_suffix(digit, suffix_thousands, 3)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 2, digit)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 1, teens_ties)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 0, graph_hundreds)
        graph_thousands.optimize()
        self.graph_thousands = graph_thousands

        graph_ten_thousands = create_graph_suffix(teens_and_ties, suffix_thousands, 3)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 2, digit)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 1, teens_ties)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 0, graph_hundreds)
        graph_ten_thousands.optimize()
        self.graph_ten_thousands = graph_ten_thousands

        # Lakhs graph and ten lakhs graph
        suffix_lakhs = pynutil.insert(" లక్షం")
        graph_lakhs = create_graph_suffix(digit, suffix_lakhs, 5)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 4, digit)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 3, teens_ties)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 2, graph_hundreds)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 1, graph_thousands)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 0, graph_ten_thousands)
        graph_lakhs.optimize()
        self.graph_lakhs = graph_lakhs

        graph_ten_lakhs = create_graph_suffix(teens_and_ties, suffix_lakhs, 5)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 4, digit)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 3, teens_ties)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 2, graph_hundreds)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 1, graph_thousands)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 0, graph_ten_thousands)
        graph_ten_lakhs.optimize()
        self.graph_ten_lakhs = graph_ten_lakhs

        # Crores graph ten crores graph
        suffix_crores = pynutil.insert(" కోటి")
        graph_crores = create_graph_suffix(digit, suffix_crores, 7)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 6, digit)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 5, teens_ties)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 4, graph_hundreds)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 3, graph_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 2, graph_ten_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 1, graph_lakhs)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 0, graph_ten_lakhs)
        graph_crores.optimize()

        graph_ten_crores = create_graph_suffix(teens_and_ties, suffix_crores, 7)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 6, digit)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 5, teens_ties)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 4, graph_hundreds)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 3, graph_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 2, graph_ten_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 1, graph_lakhs)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 0, graph_ten_lakhs)
        graph_ten_crores.optimize()
        self.graph_ten_crores = graph_ten_crores
        self.graph_crores = graph_crores

        # Arabs graph and ten arabs graph
        suffix_arabs = pynutil.insert(" శతకోటి")
        graph_arabs = create_graph_suffix(digit, suffix_arabs, 9)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 8, digit)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 7, teens_ties)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 6, graph_hundreds)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 5, graph_thousands)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 4, graph_ten_thousands)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 3, graph_lakhs)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 2, graph_ten_lakhs)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 1, graph_crores)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 0, graph_ten_crores)
        graph_arabs.optimize()

        graph_ten_arabs = create_graph_suffix(teens_and_ties, suffix_arabs, 9)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 8, digit)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 7, teens_ties)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 6, graph_hundreds)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 5, graph_thousands)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 4, graph_ten_thousands)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 3, graph_lakhs)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 2, graph_ten_lakhs)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 1, graph_crores)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 0, graph_ten_crores)
        graph_ten_arabs.optimize()
        self.graph_ten_arabs = graph_ten_arabs
        self.graph_arabs = graph_arabs

        # Only match exactly 2 digits to avoid interfering with telephone numbers, decimals, etc.
        # e.g., "౦౫" -> "సున్నా ఐదు"
        single_digit = digit | zero
        graph_leading_zero = zero + insert_space + single_digit
        graph_leading_zero = pynutil.add_weight(graph_leading_zero, 0.5)

        # Combine all number patterns efficiently
        # Support both Telugu digits and Arabic digits
        # Telugu digits go directly to final_graph
        telugu_final_graph = (
            digit
            | zero
            | teens_and_ties
            | graph_hundreds
            | graph_thousands
            | graph_ten_thousands
            | graph_lakhs
            | graph_ten_lakhs
            | graph_crores
            | graph_ten_crores
            | graph_arabs
            | graph_ten_arabs
            | graph_leading_zero
        ).optimize()

        # Strict international comma handling:
        #   X,YYY -> X వెయ్యి YYY
        #   X,YYY,ZZZ -> X మిలియన్ YYY వెయ్యి ZZZ
        #   X,YYY,ZZZ,WWW -> X బిలియన్ YYY మిలియన్ ZZZ వెయ్యి WWW
        #   X,YYY,ZZZ,WWW,VVV -> X ట్రిలియన్ YYY బిలియన్ ZZZ మిలియన్ WWW వెయ్యి VVV
        te_1_3 = pynini.closure(NEMO_TE_DIGIT, 1, 3)
        ar_1_3 = pynini.closure(NEMO_DIGIT, 1, 3)
        te_3 = NEMO_TE_DIGIT + NEMO_TE_DIGIT + NEMO_TE_DIGIT
        ar_3 = NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT
        te_3_nonzero = pynini.difference(te_3, pynini.accep("౦౦౦")).optimize()
        ar_3_nonzero = pynini.difference(ar_3, pynini.accep("000")).optimize()

        up_to_999 = (graph_hundreds | teens_and_ties | digit | zero).optimize()
        leading_zero_strip = pynini.closure(pynutil.delete("౦"), 0, 2)

        group_1_3 = (
            pynini.compose(te_1_3, telugu_final_graph)
            | pynini.compose(ar_1_3, arabic_to_telugu_number @ telugu_final_graph)
        ).optimize()
        group_3 = (
            pynini.compose(te_3, leading_zero_strip + up_to_999)
            | pynini.compose(ar_3, arabic_to_telugu_number @ (leading_zero_strip + up_to_999))
        ).optimize()
        group_3_nonzero = (
            pynini.compose(te_3_nonzero, leading_zero_strip + up_to_999)
            | pynini.compose(ar_3_nonzero, arabic_to_telugu_number @ (leading_zero_strip + up_to_999))
        ).optimize()

        delete_comma = pynutil.delete(",")
        intl_thousand = (
            group_1_3 + delete_comma + pynutil.insert(" వెయ్యి ") + group_3
        ).optimize()
        intl_thousand_zero_tail = (
            group_1_3 + delete_comma + pynutil.insert(" వెయ్యి") + pynutil.delete(pynini.union("౦౦౦", "000"))
        ).optimize()
        intl_million = (
            group_1_3
            + delete_comma
            + pynutil.insert(" మిలియన్ ")
            + group_3_nonzero
            + delete_comma
            + pynutil.insert(" వెయ్యి ")
            + group_3
        ).optimize()
        intl_million_zero_thousand = (
            group_1_3
            + delete_comma
            + pynutil.insert(" మిలియన్ ")
            + pynutil.delete(pynini.union("౦౦౦", "000"))
            + delete_comma
            + group_3
        ).optimize()
        intl_billion = (
            group_1_3
            + delete_comma
            + pynutil.insert(" బిలియన్ ")
            + group_3_nonzero
            + delete_comma
            + pynutil.insert(" మిలియన్ ")
            + group_3_nonzero
            + delete_comma
            + pynutil.insert(" వెయ్యి ")
            + group_3
        ).optimize()
        intl_trillion = (
            group_1_3
            + delete_comma
            + pynutil.insert(" ట్రిలియన్ ")
            + group_3_nonzero
            + delete_comma
            + pynutil.insert(" బిలియన్ ")
            + group_3_nonzero
            + delete_comma
            + pynutil.insert(" మిలియన్ ")
            + group_3_nonzero
            + delete_comma
            + pynutil.insert(" వెయ్యి ")
            + group_3
        ).optimize()
        strict_intl_with_commas = (
            pynutil.add_weight(intl_million_zero_thousand, -0.1)
            | pynutil.add_weight(intl_thousand_zero_tail, -0.1)
            | intl_trillion
            | intl_billion
            | intl_million
            | intl_thousand
        ).optimize()

        # Indian comma/default handling.
        telugu_with_commas = (pynini.compose(indian_comma_pattern, delete_commas) @ telugu_final_graph).optimize()
        telugu_final_with_commas = (
            pynutil.add_weight(strict_intl_with_commas, -0.1)
            | pynutil.add_weight(telugu_with_commas, -0.1)
            | telugu_final_graph
        )

        # Arabic digits: convert to Telugu, then apply the same graph.
        arabic_digit_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_final_graph = pynini.compose(arabic_digit_input, arabic_to_telugu_number @ telugu_final_graph).optimize()
        arabic_with_commas = (
            pynini.compose(indian_comma_pattern, delete_commas) @ arabic_to_telugu_number @ telugu_final_graph
        ).optimize()
        arabic_final_with_commas = (
            pynutil.add_weight(strict_intl_with_commas, -0.1)
            | pynutil.add_weight(arabic_with_commas, -0.1)
            | arabic_final_graph
        )

        # Combine both Telugu and Arabic digit paths (both with comma support)
        final_graph = telugu_final_with_commas | arabic_final_with_commas

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.final_graph = final_graph.optimize()
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph
