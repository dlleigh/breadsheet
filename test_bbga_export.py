#!/usr/bin/env python3
"""Test BBGA Excel export with bagel recipe"""

import sys
sys.path.append('.')
from breadsheet_utils import PrefermentCalculator, create_formula, export_bbga_excel

# Setup bagel recipe
calc = PrefermentCalculator(
    num_loaves=24,
    weight_grams=133)

pre_fermented_flour_ratio = 0.514

dough = create_formula({
    "high-gluten-flour": 100,
    "water": 57,
    "salt": 2,
    "yeast (IDY)": 0.47,
    "malt": 0.94
})

sponge = create_formula({
    "high-gluten-flour": 100,
    "water": 57 / pre_fermented_flour_ratio,
    "yeast (IDY)": 0.6
})

# Calculate recipe
sponge_weights, final_dough = calc.calculate_sponge_recipe(dough, sponge, pre_fermented_flour_ratio)

# Export to BBGA Excel format
export_bbga_excel(
    formula_df=final_dough,
    calc=calc,
    sponge_df=sponge_weights,
    pre_fermented_flour_ratio=pre_fermented_flour_ratio,
    output_path='bagel_formula_bbga.xlsx',
    title='Bagels'
)

print("Done! Check bagel_formula_bbga.xlsx")