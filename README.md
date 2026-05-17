# breadsheet

## What is it?

Bread recipe calculators as Jupyter notebooks. Each recipe uses baker's percentages and `breadsheet_utils` to calculate ingredient weights.

## Why?

- I created this repo in 2018 after 10+ years of using spreadsheets to manage my breadmaking formulas. The simple reason was that I hate using spreadsheets, and I love using jupyter notebooks.
- After several years' experimentation with different approaches, I created the `breadsheet_utils` module in 2025 and standardized the formula calculator and formatter according to [BBGA](https://bbga.org) formula guidelines.
- My workflow for using these formulas:
    1. Open the bread's formula in VSCode or (on iPad) on a GitHub Codespace.
    1. Adjust quantity and run the notebook
    1. Copy the final output cell and paste into an [Obsidian](https://obsidian.md/) note.
    1. View the Obsidian note on my phone, and annotate as needed (e.g. scheduling, temperatures, results)
- I use this repo every time I'm baking - often multiple times per week. 
- I'm happy if others find it useful, and feedback/suggestions are most welcome, however this is primarily for my own use.

## Approach

The calculations follow the Bread Bakers' Guild of America (BBGA) formula formatting guidelines (2009), available online to BBGA members.

Key points:

- All ingredient quantities are expressed as baker's percentages, where total flour = 100%.
- A formula's *prefermented flour ratio* determines the quantity of preferment used.
- For prefermented doughs, the preferment's ingredients (flour, water, yeast, etc.) are included in the bread's overall formula. The calculator shows three views: the overall formula, the preferment breakdown, and the final dough (overall minus what went into the preferment). 
- Note that many breadmaking books do not include preferment components in the overall formula.  Instead, those books list  the preferment in the formula based on its percentage of the flour weight, where the flour of the preferment is *not* included in the total flour weight. This (and the fact that the water in the preferment is not included in the water component of these formulas) makes it more difficult to see the dough's hydration, and results in odd % values for salt and other components.
- An exception is sourdough seed/starter culture: its flour and water components are *not* included in the overall formula. The seed is treated as a single ingredient with its own baker's percentage. 
- For dividing, loaf sizes are described in a range from 95%-105% of the desired loaf weight.

## Limitations

- The calculator currently does not handle formulas which contain multiple types of preferments.

## Using breadsheet_utils

### Straight dough (no preferment)

```python
from breadsheet_utils import RecipeCalculator, create_formula, format_and_display

formula = create_formula({
    "flour": 100,
    "water": 66,
    "yeast": 0.44,
    "salt": 2,
})

calc = RecipeCalculator(num_loaves=2, weight_grams=900)
calc.print_batch_info()

final_dough = calc.calculate_straight_dough(formula)

format_and_display(final_dough, calc, title="My Bread")
```

Output:

```
My Bread

2 loaves at 855-945 grams
overall formula total = 168.4%

|       |   baker% |    grams |    oz |
|:------|---------:|---------:|------:|
| flour |   100.00 | 1,069.00 | 37.70 |
| water |    66.00 |   705.00 | 24.90 |
| yeast |     0.44 |     4.70 |  0.17 |
| salt  |     2.00 |    21.40 |  0.75 |
```

### Prefermented dough

```python
from breadsheet_utils import RecipeCalculator, create_formula, format_and_display

formula = create_formula({
    "flour": 100,
    "water": 66,
    "yeast": 0.44,
    "salt": 2,
})

poolish = create_formula({
    "flour": 100,
    "water": 100,
    "yeast": 0.27,
})

calc = RecipeCalculator(num_loaves=4, weight_grams=350)
calc.print_batch_info()

poolish_weights, final_dough = calc.calculate(
    formula, poolish_df=poolish, pre_fermented_flour_ratio=0.33
)

format_and_display(final_dough, calc, poolish=poolish_weights, title="Poolish Bread")
```

Output:

```
Poolish Bread

4 loaves at 332-368 grams
overall formula total = 168.4%

Overall Formula:

|       |   baker% |   grams |    oz |
|:------|---------:|--------:|------:|
| flour |   100.00 |  831.00 | 29.30 |
| salt  |     2.00 |   16.60 |  0.59 |
| water |    66.00 |  549.00 | 19.40 |
| yeast |     0.44 |    3.66 |  0.13 |

Poolish:

|       |   baker% |   grams |   oz |
|:------|---------:|--------:|-----:|
| flour |   100.00 |  274.00 | 9.68 |
| water |   100.00 |  274.00 | 9.68 |
| yeast |     0.27 |    0.74 | 0.03 |

Final Dough:

|         |   baker% |   grams |    oz |
|:--------|---------:|--------:|------:|
| flour   |   100.00 |  557.00 | 19.60 |
| water   |    66.00 |  274.00 |  9.68 |
| yeast   |     0.44 |    2.92 |  0.10 |
| salt    |     2.00 |   16.60 |  0.59 |
| poolish |    66.10 |  549.00 | 19.40 |
```

`calc.calculate()` supports all preferment types via keyword arguments: `poolish_df`, `sponge_df`, `levain_df`, `pate_fermentee_df`, `desem_df`.

### Weight parameters

You can specify target weight per unit with any of: `weight_grams`, `weight_ounces`, or `weight_pounds`. Add `waste_factor` (e.g., `0.02` for 2%) to account for dough lost during handling.

### Display options

`format_and_display` accepts optional parameters:

- `title` — recipe name, printed as a header
- `steps` — markdown string of instructions, printed before the tables
- `reserved_seed_grams` — extra grams to add to the preferment display for maintaining a sourdough starter (display only, doesn't affect the dough calculation)

### Cost calculation

`calculate_cost` computes ingredient costs from a pricing CSV file.

```python
from breadsheet_utils import format_and_display_cost

cost_df = calc.calculate_cost('costs.csv', final_dough)
format_and_display_cost(cost_df, calc)
```

The CSV needs columns: `ingredient`, `quantity`, `unit_of_measure`, `cost`. Units can be grams, ounces, or pounds. Ingredient names are matched case-insensitively against the recipe. For prefermented recipes, pass the preferment DataFrame so its ingredients get costed individually (otherwise the lump "poolish" row in `final_dough` has no match in the pricing CSV and gets skipped):

```python
cost_df = calc.calculate_cost('costs.csv', final_dough, preferment_df=poolish_weights)
```

