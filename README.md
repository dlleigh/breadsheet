# breadsheet

Bread recipe calculators as Jupyter notebooks. Each recipe uses baker's percentages and `breadsheet_utils` to calculate ingredient weights.

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

### Display options

`format_and_display` accepts optional parameters:

- `title` — recipe name, printed as a header
- `steps` — markdown string of instructions, printed before the tables
- `reserved_seed_grams` — extra grams to add to the preferment display for maintaining a sourdough starter (display only, doesn't affect the dough calculation)

### Weight parameters

You can specify target weight per unit with any of: `weight_grams`, `weight_ounces`, or `weight_pounds`. Add `waste_factor` (e.g., `0.02` for 2%) to account for dough lost during handling.
