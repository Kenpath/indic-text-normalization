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
from pynini.examples import plurals
from pynini.lib import pynutil

from indic_text_normalization.text_normalization.ta.graph_utils import (
    MINUS,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractions, e.g.
        fraction { numerator: "மூன்று" denominator: "நான்கு" } -> நான்கில் மூன்று
        fraction { integer_part: "பன்னிரண்டு" numerator: "மூன்று" denominator: "நான்கு" } -> பன்னிரண்டு மற்றும் நான்கில் மூன்று
    
    Following English fraction verbalizer pattern.
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        # Integer part extraction
        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")

        # Special denominators (like English half, quarter)
        denominator_half = pynini.cross("denominator: \"இரண்டு\"", "அரை")
        denominator_quarter = pynini.cross("denominator: \"நான்கு\"", "கால்")

        # Denominator with "இல்" suffix - map cardinal to "இல்" form
        denominator_il_suffix = pynini.string_map([
            ("ஒன்று", "ஒன்றில்"),
            ("இரண்டு", "இரண்டில்"),
            ("மூன்று", "மூன்றில்"),
            ("நான்கு", "நான்கில்"),
            ("ஐந்து", "ஐந்தில்"),
            ("ஆறு", "ஆறில்"),
            ("ஏழு", "ஏழில்"),
            ("எட்டு", "எட்டில்"),
            ("ஒன்பது", "ஒன்பதில்"),
            ("பத்து", "பத்தில்"),
            ("பதினொன்று", "பதினொன்றில்"),
            ("பன்னிரண்டு", "பன்னிரண்டில்"),
            ("பதிமூன்று", "பதிமூன்றில்"),
            ("பதினான்கு", "பதினான்கில்"),
            ("பதினைந்து", "பதினைந்தில்"),
            ("பதினாறு", "பதினாறில்"),
            ("பதினேழு", "பதினேழில்"),
            ("பதினெட்டு", "பதினெட்டில்"),
            ("பத்தொன்பது", "பத்தொன்பதில்"),
            ("இருபது", "இருபதில்"),
            ("இருபத்தொன்று", "இருபத்தொன்றில்"),
            ("இருபத்திரண்டு", "இருபத்திரண்டில்"),
            ("இருபத்துமூன்று", "இருபத்துமூன்றில்"),
            ("இருபத்துநான்கு", "இருபத்துநான்கில்"),
            ("இருபத்தைந்து", "இருபத்தைந்தில்"),
            ("இருபத்தாறு", "இருபத்தாறில்"),
            ("இருபத்தேழு", "இருபத்தேழில்"),
            ("இருபத்தெட்டு", "இருபத்தெட்டில்"),
            ("இருபத்தொன்பது", "இருபத்தொன்பதில்"),
            ("முப்பது", "முப்பதில்"),
            ("நாற்பது", "நாற்பதில்"),
            ("ஐம்பது", "ஐம்பதில்"),
            ("அறுபது", "அறுபதில்"),
            ("எழுபது", "எழுபதில்"),
            ("எண்பது", "எண்பதில்"),
            ("தொண்ணூறு", "தொண்ணூறில்"),
            ("நூறு", "நூறில்"),
        ]).optimize()

        denominator_rest = (
            pynutil.delete("denominator: \"") 
            + pynini.closure(NEMO_NOT_QUOTE) @ denominator_il_suffix 
            + pynutil.delete("\"")
        )

        # Priority union: half > quarter > rest (like English)
        denominators = plurals._priority_union(
            denominator_half,
            plurals._priority_union(denominator_quarter, denominator_rest, NEMO_SIGMA),
            NEMO_SIGMA,
        ).optimize()

        # Numerator when it's "one" (ஒன்று) - special case like English
        numerator_one = pynutil.delete("numerator: \"") + pynini.accep("ஒன்று") + pynutil.delete("\" ")
        numerator_one = numerator_one + insert_space + denominators

        # Numerator for other values
        numerator_rest = (
            pynutil.delete("numerator: \"")
            + (pynini.closure(NEMO_NOT_QUOTE) - pynini.accep("ஒன்று"))
            + pynutil.delete("\" ")
        )
        numerator_rest = denominators + insert_space + numerator_rest

        graph = numerator_one | numerator_rest

        # Add "மற்றும்" (and) for mixed numbers like English "and"
        conjunction = pynutil.insert(" மற்றும் ")
        integer = pynini.closure(integer + conjunction, 0, 1)

        graph = integer + graph

        # Handle negative: add MINUS at the beginning
        optional_sign = pynini.closure(
            pynini.cross("negative: \"true\" ", MINUS),
            0,
            1,
        )
        graph = optional_sign + graph

        self.graph = graph
        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()

