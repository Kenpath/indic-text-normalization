# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from indic_text_normalization.te.graph_utils import (
    MIN_NEG_WEIGHT,
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone numbers, e.g.
        telephone { country_code: "ప్లస్ తొమ్మిది ఒకటి", number_part: "తొమ్మిది రెండు ఒకటి సున్నా ఐదు ఒకటి ఐదు ఆరు సున్నా ఆరు" } ->  ప్లస్ తొమ్మిది ఒకటి తొమ్మిది రెండు ఒకటి సున్నా ఐదు ఒకటి ఐదు ఆరు సున్నా ఆరు
        telephone { number_part: "సున్నా ఒకటి మూడు ఏడు నాలుగు మూడు సున్నా తొమ్మిది తొమ్మిది ఎనిమిది ఎనిమిది" } -> సున్నా ఒకటి మూడు ఏడు నాలుగు మూడు సున్నా తొమ్మిది తొమ్మిది ఎనిమిది ఎనిమిది

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        optional_country_code = pynini.closure(
            pynutil.delete("country_code: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + insert_space,
            0,
            1,
        )

        number_part = (
            pynutil.delete("number_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynini.closure(pynutil.add_weight(pynutil.delete(NEMO_SPACE), MIN_NEG_WEIGHT), 0, 1)
            + pynutil.delete("\"")
        )

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

