# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from indic_text_normalization.ta.graph_utils import (
    MIN_NEG_WEIGHT,
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone numbers in Tamil.

    Examples:
        telephone { country_code: "பிளஸ் ஒன்பது ஒன்று" number_part: "ஒன்பது எட்டு ஏழு ஆறு ஐந்து நான்கு மூன்று இரண்டு ஒன்று சுழியம்" }
        -> பிளஸ் ஒன்பது ஒன்று ஒன்பது எட்டு ஏழு ஆறு ஐந்து நான்கு மூன்று இரண்டு ஒன்று சுழியம்

        telephone { number_part: "சுழியம் ஒன்பது எட்டு ஏழு ஆறு ஐந்து நான்கு மூன்று இரண்டு ஒன்று சுழியம்" }
        -> சுழியம் ஒன்பது எட்டு ஏழு ஆறு ஐந்து நான்கு மூன்று இரண்டு ஒன்று சுழியம்

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        # Optional country code
        optional_country_code = pynini.closure(
            pynutil.delete("country_code: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + insert_space,
            0,
            1,
        )

        # Number part (required)
        number_part = (
            pynutil.delete("number_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynini.closure(pynutil.add_weight(pynutil.delete(NEMO_SPACE), MIN_NEG_WEIGHT), 0, 1)
            + pynutil.delete("\"")
        )

        # Optional extension
        optional_extension = pynini.closure(
            delete_space
            + insert_space
            + pynutil.delete("extension: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\""),
            0,
            1,
        )

        graph = optional_country_code + number_part + optional_extension
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

