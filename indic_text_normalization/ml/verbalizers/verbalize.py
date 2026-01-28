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

from indic_text_normalization.ml.graph_utils import GraphFst
from indic_text_normalization.ml.verbalizers.abbreviation import AbbreviationFst
from indic_text_normalization.ml.verbalizers.cardinal import CardinalFst
from indic_text_normalization.ml.verbalizers.date import DateFst
from indic_text_normalization.ml.verbalizers.decimal import DecimalFst
from indic_text_normalization.ml.verbalizers.electronic import ElectronicFst
from indic_text_normalization.ml.verbalizers.fraction import FractionFst
from indic_text_normalization.ml.verbalizers.math import MathFst
from indic_text_normalization.ml.verbalizers.measure import MeasureFst
from indic_text_normalization.ml.verbalizers.money import MoneyFst
from indic_text_normalization.ml.verbalizers.ordinal import OrdinalFst
from indic_text_normalization.ml.verbalizers.roman import RomanFst
from indic_text_normalization.ml.verbalizers.telephone import TelephoneFst
from indic_text_normalization.ml.verbalizers.time import TimeFst
from indic_text_normalization.ml.verbalizers.whitelist import WhiteListFst
from indic_text_normalization.ml.verbalizers.power import PowerFst
from indic_text_normalization.ml.verbalizers.scientific import ScientificFst


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

        fraction = FractionFst(cardinal=cardinal, deterministic=deterministic)
        fraction_graph = fraction.fst

        date = DateFst()
        date_graph = date.fst

        time = TimeFst(cardinal=cardinal)
        time_graph = time.fst

        measure = MeasureFst(cardinal=cardinal, decimal=decimal)
        measure_graph = measure.fst

        money = MoneyFst()
        money_graph = money.fst

        telephone = TelephoneFst()
        telephone_graph = telephone.fst
        ordinal = OrdinalFst(deterministic=deterministic)
        ordinal_graph = ordinal.fst

        math = MathFst(deterministic=deterministic)
        math_graph = math.fst

        whitelist_graph = WhiteListFst(deterministic=deterministic).fst

        electronic_graph = ElectronicFst(deterministic=deterministic).fst
        power_graph = PowerFst(deterministic=deterministic).fst
        scientific_graph = ScientificFst(deterministic=deterministic).fst

        graph = (
            cardinal_graph
            | decimal_graph
            | fraction_graph
            | date_graph
            | time_graph
            | measure_graph
            | money_graph
            | ordinal_graph
            | math_graph
            | scientific_graph
            | whitelist_graph
            | telephone_graph
            | electronic_graph
            | power_graph
        )

        roman_graph = RomanFst(deterministic=deterministic).fst
        graph |= roman_graph

        if not deterministic:
            abbreviation_graph = AbbreviationFst(deterministic=deterministic).fst
            graph |= abbreviation_graph

        self.fst = graph
