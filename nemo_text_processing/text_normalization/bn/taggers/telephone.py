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

from nemo_text_processing.text_normalization.bn.graph_utils import GraphFst, NEMO_DIGIT, NEMO_BN_DIGIT, insert_space
from nemo_text_processing.text_normalization.bn.utils import get_abs_path

# Convert Arabic digits (0-9) to Bengali digits (০-৯)
arabic_to_bengali_digit = pynini.string_map([
    ("0", "০"), ("1", "১"), ("2", "২"), ("3", "৩"), ("4", "৪"),
    ("5", "৫"), ("6", "৬"), ("7", "৭"), ("8", "৮"), ("9", "৯")
]).optimize()
arabic_to_bengali_number = pynini.closure(arabic_to_bengali_digit).optimize()

digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        "৯৮৭৬৫৪৩২১০" -> telephone { number_part: "নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        from nemo_text_processing.text_normalization.bn.graph_utils import NEMO_SPACE

        digit_graph = digit | zero
        # Require minimum 10 digits for telephone numbers to avoid matching short numbers like "123"
        min_digits = 10
        
        bengali_digit_input = pynini.closure(NEMO_BN_DIGIT, min_digits)
        arabic_digit_input = pynini.closure(NEMO_DIGIT, min_digits)

        # Support both Bengali and Arabic digits, but require minimum length
        bengali_path = pynini.compose(bengali_digit_input, pynini.closure(digit_graph + insert_space, min_digits)).optimize()
        arabic_path = pynini.compose(
            arabic_digit_input,
            arabic_to_bengali_number @ pynini.closure(digit_graph + insert_space, min_digits)
        ).optimize()

        number_graph = bengali_path | arabic_path

        # Handle country code with + sign (e.g., +91)
        # Convert + to Bengali "প্লাস"
        plus_sign = pynini.cross("+", "প্লাস")
        country_code_digits = pynini.closure(pynini.union(NEMO_BN_DIGIT, NEMO_DIGIT), 1, 3)
        country_code_graph = (
            pynutil.insert("country_code: \"")
            + plus_sign
            + insert_space
            + pynini.compose(
                country_code_digits,
                arabic_to_bengali_number @ pynini.closure(digit_graph + insert_space, 1, 3)
            )
            + pynutil.insert("\" ")
        )

        # Phone number with country code
        phone_with_country = country_code_graph + pynutil.insert("number_part: \"") + number_graph + pynutil.insert("\"")
        
        # Phone number without country code
        phone_without_country = pynutil.insert("number_part: \"") + number_graph + pynutil.insert("\"")

        graph = phone_with_country | phone_without_country

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()

