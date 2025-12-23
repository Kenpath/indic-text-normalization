# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from indic_text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    insert_space,
)
from indic_text_normalization.en.taggers.date import get_four_digit_year_graph
from indic_text_normalization.en.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        self.lm = lm
        self.deterministic = deterministic
        # TODO replace to have "oh" as a default for "0"
        graph = pynini.Far(get_abs_path("data/number/cardinal_number_name.far")).get_fst()
        graph_au = pynini.Far(get_abs_path("data/number/cardinal_number_name_au.far")).get_fst()
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            pynini.closure(NEMO_DIGIT, 2, 3) | pynini.difference(NEMO_DIGIT, pynini.accep("0"))
        ) @ graph

        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        single_digits_graph = pynini.invert(graph_digit | graph_zero)
        self.single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)

        if not deterministic:
            # for a single token allow only the same normalization
            # "007" -> {"oh oh seven", "zero zero seven"} not {"oh zero seven"}
            single_digits_graph_zero = pynini.invert(graph_digit | graph_zero)
            single_digits_graph_oh = pynini.invert(graph_digit) | pynini.cross("0", "oh")

            self.single_digits_graph = single_digits_graph_zero + pynini.closure(
                insert_space + single_digits_graph_zero
            )
            self.single_digits_graph |= single_digits_graph_oh + pynini.closure(insert_space + single_digits_graph_oh)

            single_digits_graph_with_commas = pynini.closure(
                self.single_digits_graph + insert_space, 1, 3
            ) + pynini.closure(
                pynutil.delete(",")
                + single_digits_graph
                + insert_space
                + single_digits_graph
                + insert_space
                + single_digits_graph,
                1,
            )

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        graph = (
            pynini.closure(NEMO_DIGIT, 1, 3)
            + (pynini.closure(pynutil.delete(",") + NEMO_DIGIT**3) | pynini.closure(NEMO_DIGIT**3))
        ) @ graph

        self.graph = graph
        self.graph_with_and = self.add_optional_and(graph)

        if deterministic:
            # Exclude phone-number-like patterns (7-10 digits) from proper number normalization
            # But preserve round numbers (ending in many zeros) which should be normalized properly
            # Phone numbers are typically 7-10 digits that don't end in many zeros
            phone_number_pattern_7 = NEMO_DIGIT ** 7  # Exactly 7 digits
            phone_number_pattern_8 = NEMO_DIGIT ** 8  # Exactly 8 digits  
            phone_number_pattern_9 = NEMO_DIGIT ** 9  # Exactly 9 digits
            # 10 digits: exclude unless it's a round number (ends with at least 6 zeros)
            phone_number_pattern_10 = (
                NEMO_DIGIT ** 4 + pynini.difference(NEMO_DIGIT ** 6, "000000")
            )  # 4 digits + 6 digits that aren't all zeros
            phone_number_pattern = (
                phone_number_pattern_7 
                | phone_number_pattern_8 
                | phone_number_pattern_9 
                | phone_number_pattern_10
            ).optimize()
            # Get the input domain of the proper number graph and exclude phone number patterns
            graph_input_domain = pynini.project(self.graph_with_and, "input")
            non_phone_inputs = pynini.difference(graph_input_domain, phone_number_pattern).optimize()
            # Filter the graph: compose with non_phone_inputs (which acts as an identity FST for allowed inputs)
            # This keeps only paths where the input is in non_phone_inputs
            graph_excluding_phone = pynini.compose(non_phone_inputs, self.graph_with_and).optimize()
            
            # Give priority to proper number names (like "one hundred thousand", "one billion")
            # over digit-by-digit conversion for long numbers
            long_numbers = pynini.compose(NEMO_DIGIT ** (5, ...), self.single_digits_graph).optimize()
            # Use weighted union: lower weight = higher priority
            # Proper number names (excluding phone patterns) get weight 0.1 (higher priority)
            # Digit-by-digit gets weight 1.0 (lower priority, fallback)
            self.long_numbers = (
                pynutil.add_weight(graph_excluding_phone, 0.1) 
                | pynutil.add_weight(long_numbers, 1.0)
            ).optimize()
            cardinal_with_leading_zeros = pynini.compose(
                pynini.accep("0") + pynini.closure(NEMO_DIGIT), self.single_digits_graph
            )
            final_graph = self.long_numbers | cardinal_with_leading_zeros
            final_graph |= self.add_optional_and(graph_au)
        else:
            leading_zeros = pynini.compose(pynini.closure(pynini.accep("0"), 1), self.single_digits_graph)
            cardinal_with_leading_zeros = (
                leading_zeros + pynutil.insert(" ") + pynini.compose(pynini.closure(NEMO_DIGIT), self.graph_with_and)
            )
            self.long_numbers = self.graph_with_and | pynutil.add_weight(self.single_digits_graph, 0.0001)
            # add small weight to non-default graphs to make sure the deterministic option is listed first
            final_graph = (
                self.long_numbers
                | get_four_digit_year_graph()  # allows e.g. 4567 be pronounced as forty five sixty seven
                | pynutil.add_weight(single_digits_graph_with_commas, 0.0001)
                | cardinal_with_leading_zeros
            ).optimize()

            one_to_a_replacement_graph = (
                pynini.cross("one hundred", "a hundred")
                | pynini.cross("one thousand", "thousand")
                | pynini.cross("one million", "a million")
            )
            final_graph |= pynini.compose(final_graph, one_to_a_replacement_graph.optimize() + NEMO_SIGMA).optimize()
            # remove commas for 4 digits numbers
            four_digit_comma_graph = (NEMO_DIGIT - "0") + pynutil.delete(",") + NEMO_DIGIT**3
            final_graph |= pynini.compose(four_digit_comma_graph.optimize(), final_graph).optimize()

        self.final_graph = final_graph
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def add_optional_and(self, graph):
        graph_with_and = graph

        if not self.lm:
            graph_with_and = pynutil.add_weight(graph, 0.00001)
            not_quote = pynini.closure(NEMO_NOT_QUOTE)
            no_thousand_million = pynini.difference(
                not_quote, not_quote + pynini.union("thousand", "million") + not_quote
            ).optimize()
            integer = (
                not_quote + pynutil.add_weight(pynini.cross("hundred ", "hundred and ") + no_thousand_million, -0.0001)
            ).optimize()

            no_hundred = pynini.difference(NEMO_SIGMA, not_quote + pynini.accep("hundred") + not_quote).optimize()
            integer |= (
                not_quote + pynutil.add_weight(pynini.cross("thousand ", "thousand and ") + no_hundred, -0.0001)
            ).optimize()

            optional_hundred = pynini.compose((NEMO_DIGIT - "0") ** 3, graph).optimize()
            optional_hundred = pynini.compose(optional_hundred, NEMO_SIGMA + pynini.cross(" hundred", "") + NEMO_SIGMA)
            graph_with_and |= pynini.compose(graph, integer).optimize()
            graph_with_and |= optional_hundred
        return graph_with_and
