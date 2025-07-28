# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Breadsheet is a collection of bread recipe calculators implemented as Jupyter notebooks. Each recipe is a Python notebook that calculates ingredient quantities based on desired batch sizes using baker's percentages.

## Architecture

- **Structure**: Recipe categories organized in `formulas/` subdirectories (bagel, baguette, ciabatta, conchas, crackers, focaccia, grissini, pain-rustique, pita, pizza, sourdough)
- **Format**: Each recipe is a Jupyter notebook (.ipynb) containing:
  - Recipe instructions and timing as markdown
  - Python code using pandas DataFrames for ingredient calculations
  - Baker's percentage formulas converted to weights in grams and ounces
- **Dependencies**: Python environment with pandas, numpy (see requirements.txt)

## Common Operations

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running Notebooks
- Use Jupyter Lab/Notebook or compatible environment
- Available via Binder: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/dlleigh/breadsheet/master)

### Recipe Pattern
Most notebooks follow this structure:
1. Set batch parameters (num_loaves, weight per unit)
2. Define baker's percentages in pandas DataFrames
3. Calculate ingredient weights in grams and ounces
4. Display formatted results using pandas styling

### Key Variables
- `grams_per_ounce = 28.3495` - conversion constant
- `num_loaves` - batch size
- `loaf_weight` - individual unit weight
- `baker%` - baker's percentages (flour = 100%)

## Recipe Categories

- **Pizza**: Multiple variations with poolish pre-ferments (4ct, 6ct, 8ct, 20ct versions)
- **Sourdough**: Various loaf counts (4, 6, 8, 12, 16 loaves)
- **Bagels**: Including specialty versions (cinnamon-raisin, 2x batch)
- **Italian**: Focaccia (1x, 2x, 4x), ciabattini, grissini
- **Other**: Baguette, conchas, crackers, pain rustique, pita