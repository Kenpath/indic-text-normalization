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

from indic_text_normalization.ta.graph_utils import GraphFst
from indic_text_normalization.ta.verbalizers.cardinal import CardinalFst
from indic_text_normalization.ta.verbalizers.date import DateFst
from indic_text_normalization.ta.verbalizers.decimal import DecimalFst
from indic_text_normalization.ta.verbalizers.fraction import FractionFst
from indic_text_normalization.ta.verbalizers.math import MathFst
from indic_text_normalization.ta.verbalizers.measure import MeasureFst
from indic_text_normalization.ta.verbalizers.money import MoneyFst
from indic_text_normalization.ta.verbalizers.ordinal import OrdinalFst
from indic_text_normalization.ta.verbalizers.telephone import TelephoneFst
from indic_text_normalization.ta.verbalizers.time import TimeFst
from indic_text_normalization.ta.verbalizers.whitelist import WhiteListFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)

        cardinal = CardinalFst(deterministic=deterministic)
        cardinal_graph = cardinal.fst

        decimal = DecimalFst(deterministic=deterministic)
        decimal_graph = decimal.fst

        time = TimeFst(cardinal=cardinal)
        time_graph = time.fst

        date = DateFst(deterministic=deterministic)
        date_graph = date.fst

        money = MoneyFst()
        money_graph = money.fst

        measure = MeasureFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
        measure_graph = measure.fst

        fraction = FractionFst(deterministic=deterministic)
        fraction_graph = fraction.fst

        ordinal = OrdinalFst(deterministic=deterministic)
        ordinal_graph = ordinal.fst

        math = MathFst(deterministic=deterministic)
        math_graph = math.fst

        telephone = TelephoneFst(deterministic=deterministic)
        telephone_graph = telephone.fst

        whitelist_graph = WhiteListFst(deterministic=deterministic).fst

        graph = (
            cardinal_graph
            | decimal_graph
            | time_graph
            | date_graph
            | money_graph
            | measure_graph
            | fraction_graph
            | ordinal_graph
            | math_graph
            | telephone_graph
            | whitelist_graph
        )

        self.fst = graph

