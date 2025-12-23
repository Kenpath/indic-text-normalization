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

"""
Script to generate roman_to_spoken.tsv for Malayalam.
This maps Roman numerals to Malayalam number words.
"""

import csv
import os


def get_abs_path(rel_path):
    """Get absolute path relative to this script's directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)


def load_labels(abs_path):
    """Load labels from a TSV file."""
    with open(abs_path, encoding="utf-8") as f:
        labels = list(csv.reader(f, delimiter="\t"))
    return labels


def int_to_roman(n):
    """Convert integer to Roman numeral."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            roman_num += syms[i]
            n -= val[i]
        i += 1
    return roman_num


def num_to_hindi(num):
    """Convert number to Malayalam words using data files."""
    if num == 0:
        return "शून्य"

    # Load data files
    digits = load_labels(get_abs_path("../numbers/digit.tsv"))
    teens_and_ties = load_labels(get_abs_path("../numbers/teens_and_ties.tsv"))
    hundreds = load_labels(get_abs_path("../numbers/hundred.tsv"))
    thousands = load_labels(get_abs_path("../numbers/thousands.tsv"))

    # Create lookup dictionaries
    digit_map = {int(row[0]): row[1] for row in digits}
    teens_ties_map = {int(row[0]): row[1] for row in teens_and_ties}
    hundred_map = {int(row[0]): row[1] for row in hundreds}
    thousand_map = {int(row[0]): row[1] for row in thousands}

    parts = []

    # Handle thousands (1000-1999)
    if num >= 1000:
        thousands_val = num // 1000
        if thousands_val == 1:
            parts.append("एक हज़ार")
        else:
            # For 2-9 thousands, use digit + हज़ार
            if thousands_val in digit_map:
                parts.append(digit_map[thousands_val] + " हज़ार")
            else:
                # For 10+ thousands, use teens_ties + हज़ार
                if thousands_val in teens_ties_map:
                    parts.append(teens_ties_map[thousands_val] + " हज़ार")
                else:
                    # Handle complex thousands (e.g., 11, 12, etc.)
                    parts.append(teens_ties_map.get(thousands_val, str(thousands_val)) + " हज़ार")
        num %= 1000

    # Handle hundreds (100-999)
    if num >= 100:
        hundreds_val = num // 100
        if hundreds_val == 1:
            parts.append("एक सौ")
        else:
            if hundreds_val in digit_map:
                parts.append(digit_map[hundreds_val] + " सौ")
            else:
                parts.append(str(hundreds_val) + " सौ")
        num %= 100

    # Handle tens and units (1-99)
    if num > 0:
        if num in teens_ties_map:
            parts.append(teens_ties_map[num])
        elif num in digit_map:
            parts.append(digit_map[num])
        else:
            # This shouldn't happen for numbers 1-99, but handle it
            parts.append(str(num))

    return " ".join(parts).strip()


def generate_roman_to_spoken_tsv(output_file):
    """Generate roman_to_spoken.tsv file with Roman numeral to Malayalam word mappings."""
    generated_mappings = []
    
    # Generate mappings for numbers 1 to 2000 (same as English)
    for i in range(1, 2001):
        roman_numeral = int_to_roman(i)
        hindi_word = num_to_hindi(i)
        generated_mappings.append([roman_numeral, hindi_word])

    # Write to file
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in generated_mappings:
            writer.writerow(row)
    
    print(f"Generated {output_file} with {len(generated_mappings)} entries")


if __name__ == "__main__":
    output_file = get_abs_path("roman_to_spoken.tsv")
    generate_roman_to_spoken_tsv(output_file)
