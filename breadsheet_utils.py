import pandas as pd
from typing import Dict, Optional
import math

GRAMS_PER_OUNCE = 28.3495
GRAMS_PER_POUND = GRAMS_PER_OUNCE * 16
GRAMS_PER_TBSP = 14.787
GRAMS_PER_TSP = GRAMS_PER_TBSP / 3
GRAMS_PER_LARGE_EGG = 50

# Grams per cup for common baking ingredients (used for volume column)
GRAMS_PER_CUP = {
    'water': 236.6,
    'oil': 218,
    'sugar': 200,
    'brown sugar': 220,
    'salt': 288,
    'honey': 340,
    'milk': 242,
    'butter': 227,
    'baking powder': 192,
    'baking soda': 192,
    'yeast': 150,
    'extract': 283,
    'vanilla': 283,
    'cocoa': 86,
    'cinnamon': 125,
    'almonds': 145,
}

# Ingredient classification sets (lowercase for case-insensitive matching)
SEED_CULTURE_NAMES = {'seed', 'starter', 'levain', 'desem'}
PREFERMENT_NAMES = {'poolish', 'sponge', 'levain', 'pâte fermentée', 'desem', 'biga'}

def _get_grams_per_cup(ingredient_name: str) -> Optional[float]:
    """
    Look up grams-per-cup for an ingredient by substring match.
    Returns None if no match or if the ingredient contains 'flour'.
    """
    name = ingredient_name.lower().replace('-', ' ')
    if 'flour' in name:
        return None
    # Try longest key first to prefer 'brown sugar' over 'sugar', etc.
    for key in sorted(GRAMS_PER_CUP.keys(), key=len, reverse=True):
        if key in name:
            return GRAMS_PER_CUP[key]
    return None


def _nearest_fraction(value: float, denominator: int = 8) -> tuple[int, int, int]:
    """
    Snap a float to the nearest fraction with the given denominator.
    Returns (whole, numerator, denominator) where numerator < denominator.
    """
    total_parts = round(value * denominator)
    whole = total_parts // denominator
    numerator = total_parts % denominator
    # Simplify the fraction
    if numerator > 0:
        from math import gcd
        g = gcd(numerator, denominator)
        numerator //= g
        denominator //= g
    return whole, numerator, denominator


def _format_fraction(value: float, denominator: int = 8) -> str:
    """Format a float as a fraction string (e.g., '1 3/4', '1/2', '3')."""
    whole, num, den = _nearest_fraction(value, denominator)
    if num == 0:
        return str(whole) if whole > 0 else '0'
    if whole == 0:
        return f"{num}/{den}"
    return f"{whole} {num}/{den}"


def _grams_to_volume_str(grams: float, grams_per_cup: float) -> Optional[str]:
    """
    Convert grams to a volume string using the most natural unit.
    Uses the ingredient's density (grams_per_cup) for all unit conversions.
    Returns None if the result exceeds 1 cup.
    """
    cups = grams / grams_per_cup
    if cups > 1.0:
        return None

    grams_per_tbsp = grams_per_cup / 16
    grams_per_tsp = grams_per_cup / 48

    # Try cups (if >= 1/4 cup)
    if cups >= 0.25:
        return _format_fraction(cups) + " cup"

    # Try tablespoons (if >= 1/2 tbsp)
    tbsp = grams / grams_per_tbsp
    if tbsp >= 0.5:
        return _format_fraction(tbsp) + " tbsp"

    # Use teaspoons, round to nearest 1/8
    tsp = grams / grams_per_tsp
    return _format_fraction(tsp) + " tsp"


def _add_volume_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'volume' column with human-readable volume measurements."""
    result = df.copy()
    volumes = []
    for ingredient in result.index:
        name = ingredient.lower().replace('-', ' ')
        if 'egg' in name:
            count = round(result.at[ingredient, 'grams'] / GRAMS_PER_LARGE_EGG)
            label = 'egg' if count == 1 else 'eggs'
            volumes.append(f"{count} {label}")
        else:
            gpc = _get_grams_per_cup(ingredient)
            if gpc is None:
                volumes.append('')
            else:
                vol_str = _grams_to_volume_str(result.at[ingredient, 'grams'], gpc)
                volumes.append(vol_str if vol_str else '')
    result['volume'] = volumes
    return result


def _name_matches(index: pd.Index, names: set) -> pd.Series:
    """Check if index values exactly match any name in the set (case-insensitive)"""
    return pd.Series(index.str.lower().isin(names), index=index)

def format_significant_digits(value, sig_digits=3):
    """
    Format number to specified significant digits with smart decimal placement

    Args:
        value: Number to format
        sig_digits: Number of significant digits to preserve
        max_decimal_places: Maximum number of digits after decimal point (optional)

    Returns:
        Formatted string with appropriate precision
    """
    if abs(value) < 10 ** (-sig_digits):
        x = 0
    else:
        # Get the integer part
        int_part = int(abs(value))

        # If integer part has sig_digits or more digits, no decimal places needed
        if int_part >= 10 ** (sig_digits - 1):
            x = round(value)
        else:
            # Otherwise, calculate how many decimal places we need
            decimal_places = sig_digits - len(str(int_part))
            x = round(value, decimal_places)

    return f"{x:,.{2}f}"
    

DEFAULT_FORMATTER = {
    'grams': lambda x: format_significant_digits(x),
    'oz': lambda x: format_significant_digits(x),
    'baker%': lambda x: format_significant_digits(x)
}

class RecipeCalculator:
    """Base calculator for all bread recipes"""
    
    def __init__(self, num_loaves: int, weight_pounds: int = 0,
                 weight_ounces: float = 0, weight_grams: float = 0,
                 waste_factor: float = 0.0, num_batches: int = 1,
                 preferment_waste_factor: float = 0.0):
        """
        Initialize recipe calculator

        Args:
            num_loaves: Number of loaves/units per dough batch
            weight_pounds: Weight per unit in pounds
            weight_ounces: Weight per unit in ounces
            weight_grams: Weight per unit in grams
            waste_factor: Extra dough fraction for shaping losses (e.g. 0.02 = 2%)
            num_batches: Number of separate dough mixes sharing one preferment
            preferment_waste_factor: Extra preferment fraction for bucket losses (e.g. 0.03 = 3%)
        """
        self.num_loaves = num_loaves
        self.num_batches = num_batches
        self.preferment_waste_factor = preferment_waste_factor
        self.loaf_weight = (weight_pounds * 16 + weight_ounces) * GRAMS_PER_OUNCE + weight_grams
        self.total_weight = num_loaves * self.loaf_weight * (1 + waste_factor)
    
    def get_batch_info(self):
        min_loaf_weight = self.loaf_weight * 0.95
        max_loaf_weight = self.loaf_weight * 1.05
        if self.num_batches > 1:
            total_loaves = self.num_loaves * self.num_batches
            return (f"{total_loaves:,.0f} loaves in {self.num_batches} "
                    f"dough batches of {self.num_loaves:,.0f} "
                    f"at {min_loaf_weight:,.0f}-{max_loaf_weight:,.0f} grams")
        return f"{self.num_loaves:,.0f} loaves at {min_loaf_weight:,.0f}-{max_loaf_weight:,.0f} grams"

    def print_batch_info(self):
        """Print batch size information"""
        print(self.get_batch_info())
        print(f"total weight: {self.total_weight:,.2f} grams")

    def get_flour_pct(self, formula):
        """
        Calculate total flour percentage from a formula DataFrame,
        including flour contributed by seed/starter ingredients.

        Seed/starter is assumed to have the same flour-to-water ratio
        as the other ingredients in the formula.

        Args:
            formula: DataFrame with ingredients and baker's percentages

        Returns:
            Total flour percentage (including seed/starter flour contribution)
        """
        flour_pct = formula[formula.index.str.contains('flour', case=False)]['baker%'].sum()

        seed_mask = _name_matches(formula.index, SEED_CULTURE_NAMES)
        seed_pct = formula[seed_mask]['baker%'].sum()

        if seed_pct > 0:
            non_seed_total = formula[~seed_mask]['baker%'].sum()
            flour_ratio = flour_pct / non_seed_total
            flour_pct += seed_pct * flour_ratio

        return flour_pct
    
    def calculate_straight_dough(self, formula_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weights for straight dough recipes (no preferments)

        Args:
            formula_df: DataFrame with ingredients and baker's percentages

        Returns:
            DataFrame with baker%, grams, and oz columns
        """
        _, result = self.calculate(formula_df)
        return result

    def calculate(self, formula_df: pd.DataFrame,
                  poolish_df: Optional[pd.DataFrame] = None,
                  sponge_df: Optional[pd.DataFrame] = None,
                  levain_df: Optional[pd.DataFrame] = None,
                  pate_fermentee_df: Optional[pd.DataFrame] = None,
                  desem_df: Optional[pd.DataFrame] = None,
                  pre_fermented_flour_ratio: float = 0) -> tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """
        Universal calculator that handles both straight doughs and prefermented doughs

        Args:
            formula_df: Main recipe formula
            poolish_df: Optional poolish ingredients and percentages
            sponge_df: Optional sponge ingredients and percentages
            levain_df: Optional levain ingredients and percentages
            pate_fermentee_df: Optional pâte fermentée ingredients and percentages
            desem_df: Optional desem ingredients and percentages
            pre_fermented_flour_ratio: Fraction of total flour in preferment (0-1)

        Returns:
            Tuple of (preferment_df_with_weights, final_dough_df) if preferment exists,
            or (None, formula_df_with_weights) for straight dough
        """
        # Determine preferment type and name
        preferment_df = None
        preferment_name = None

        if poolish_df is not None:
            preferment_df = poolish_df
            preferment_name = "poolish"
        elif sponge_df is not None:
            preferment_df = sponge_df
            preferment_name = "sponge"
        elif levain_df is not None:
            preferment_df = levain_df
            preferment_name = "levain"
        elif pate_fermentee_df is not None:
            preferment_df = pate_fermentee_df
            preferment_name = "pâte fermentée"
        elif desem_df is not None:
            preferment_df = desem_df
            preferment_name = "desem"

        # Straight dough (no preferment)
        if preferment_df is None:
            formula_total = formula_df['baker%'].sum()
            result_df = formula_df.copy()
            result_df['grams'] = result_df['baker%'] * self.total_weight / formula_total
            result_df['oz'] = result_df['grams'] / GRAMS_PER_OUNCE
            return None, result_df

        formula_total = formula_df['baker%'].sum()
        preferment_total = preferment_df['baker%'].sum()

        # Calculate preferment weight
        flour_weight = self.get_flour_pct(formula_df) / formula_total * self.total_weight
        preferment_flour_weight = flour_weight * pre_fermented_flour_ratio
        preferment_weight = preferment_total / self.get_flour_pct(preferment_df) * preferment_flour_weight

        # Calculate preferment ingredients
        preferment_result = preferment_df.copy()
        preferment_result['grams'] = preferment_result['baker%'] * preferment_weight / preferment_total
        preferment_result['oz'] = preferment_result['grams'] / GRAMS_PER_OUNCE

        # Calculate final dough (subtract preferment ingredients)
        final_dough = formula_df.copy()
        final_dough['grams'] = final_dough['baker%'] * self.total_weight / formula_total
        final_dough.loc[preferment_name, 'grams'] = preferment_weight
        final_dough.loc[preferment_name, 'baker%'] = preferment_weight / flour_weight * 100

        # Subtract preferment ingredients from final dough
        # Decompose seed/starter into flour and water components first
        seed_mask = _name_matches(preferment_result.index, SEED_CULTURE_NAMES)
        if seed_mask.any():
            seed_grams = preferment_result[seed_mask]['grams'].sum()
            non_seed = preferment_result[~seed_mask]
            # Distribute seed weight proportionally among non-seed ingredients
            for idx, row in non_seed.iterrows():
                proportion = row['grams'] / non_seed['grams'].sum()
                seed_contribution = seed_grams * proportion
                if idx in final_dough.index:
                    final_dough.at[idx, 'grams'] -= seed_contribution

        for index, row in preferment_result.iterrows():
            if index in final_dough.index and index.lower() not in SEED_CULTURE_NAMES:
                final_dough.at[index, 'grams'] -= row['grams']

        final_dough['oz'] = final_dough['grams'] / GRAMS_PER_OUNCE

        return preferment_result, final_dough

    def calculate_cost(self, pricing_csv_path: str,
                       final_dough_df: pd.DataFrame,
                       preferment_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate ingredient costs for a recipe

        Args:
            pricing_csv_path: Path to CSV with columns: ingredient, quantity, unit_of_measure, cost
            final_dough_df: Final dough DataFrame from calculate() with grams column
            preferment_df: Optional preferment DataFrame from calculate() with grams column

        Returns:
            DataFrame with columns: ingredient, grams, unit_cost_per_gram, total_cost
        """
        # Load pricing data
        pricing_df = pd.read_csv(pricing_csv_path)

        # Convert all pricing to cost per gram
        pricing_df['cost_per_gram'] = pricing_df.apply(
            lambda row: self._convert_to_cost_per_gram(row['cost'], row['quantity'], row['unit_of_measure']),
            axis=1
        )

        # Create a lookup dict for ingredient costs (case-insensitive)
        cost_lookup = {ing.lower(): cost for ing, cost in
                      zip(pricing_df['ingredient'], pricing_df['cost_per_gram'])}

        # Combine all ingredients from recipe
        all_ingredients = {}

        # Add preferment ingredients if present
        if preferment_df is not None:
            for idx, row in preferment_df.iterrows():
                ingredient = idx.lower()
                all_ingredients[ingredient] = all_ingredients.get(ingredient, 0) + row['grams']

        # Add final dough ingredients
        for idx, row in final_dough_df.iterrows():
            ingredient = idx.lower()
            # Skip preferment rows (they're already counted above)
            if preferment_df is not None and idx in ['poolish', 'sponge', 'levain', 'pâte fermentée', 'desem']:
                continue
            all_ingredients[ingredient] = all_ingredients.get(ingredient, 0) + row['grams']

        # Calculate costs
        cost_data = []
        for ingredient, grams in all_ingredients.items():
            if ingredient in cost_lookup:
                cost_per_gram = cost_lookup[ingredient]
                total_cost = grams * cost_per_gram
                cost_data.append({
                    'ingredient': ingredient,
                    'grams': grams,
                    'unit_cost_per_gram': cost_per_gram,
                    'total_cost': total_cost
                })
            else:
                print(f"Warning: No pricing found for '{ingredient}' - skipping")

        cost_df = pd.DataFrame(cost_data)

        # Add summary rows
        total_batch_cost = cost_df['total_cost'].sum()
        per_loaf_cost = total_batch_cost / self.num_loaves

        # Append summary rows directly to avoid concat deprecation warning
        summary_rows = [
            {'ingredient': 'TOTAL BATCH', 'grams': float('nan'), 'unit_cost_per_gram': float('nan'), 'total_cost': total_batch_cost},
            {'ingredient': 'PER LOAF', 'grams': float('nan'), 'unit_cost_per_gram': float('nan'), 'total_cost': per_loaf_cost}
        ]

        cost_df = pd.DataFrame(cost_data + summary_rows)

        return cost_df

    def _convert_to_cost_per_gram(self, cost: float, quantity: float, unit: str) -> float:
        """Convert ingredient pricing to cost per gram"""
        unit = unit.lower().strip()

        if unit in ['gram', 'grams', 'g']:
            return cost / quantity
        elif unit in ['ounce', 'ounces', 'oz']:
            grams = quantity * GRAMS_PER_OUNCE
            return cost / grams
        elif unit in ['pound', 'pounds', 'lb', 'lbs']:
            grams = quantity * GRAMS_PER_POUND
            return cost / grams
        else:
            raise ValueError(f"Unsupported unit of measure: {unit}. Use grams, ounces, or pounds.")

def create_formula(ingredients_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Create DataFrame from ingredients dictionary
    
    Args:
        ingredients_dict: Dictionary mapping ingredient names to baker's percentages
        
    Returns:
        DataFrame with ingredients as index and baker% column
    """
    return pd.DataFrame.from_dict(ingredients_dict, orient='index', columns=["baker%"])

def _format_table(df: pd.DataFrame, formatter: Dict) -> pd.DataFrame:
    """Apply formatter to a DataFrame for display"""
    display = df.copy()
    for col, fmt in formatter.items():
        if col in display.columns:
            display[col] = display[col].apply(fmt)
    return display

def _print_table(df: pd.DataFrame, formatter: Dict, label: str = ""):
    """Format and print a DataFrame as a markdown table"""
    display = _format_table(df, formatter)
    if label:
        print(f"{label}:\n")
    colalign = ["left"] + ["right"] * len(display.columns)
    print(display.to_markdown(floatfmt=".2f", colalign=colalign))

def format_and_display(formula: pd.DataFrame, calc: RecipeCalculator, poolish: pd.DataFrame = None,
                       sponge: pd.DataFrame = None,
                       levain: pd.DataFrame = None,
                       pate_fermentee: pd.DataFrame = None,
                       desem: pd.DataFrame = None,
                       formatter: Dict = None, title: str = "", steps: str = "",
                       reserved_seed_grams: float = 0,
                       show_volume: bool = False) -> pd.DataFrame:
    """
    Format DataFrame and display as markdown

    Args:
        formula: DataFrame to format and display
        formatter: Dictionary of column formatters (uses DEFAULT_FORMATTER if None)
        title: Optional title to print before the table
        reserved_seed_grams: Extra preferment to make for maintaining starter (display only)

    Returns:
        Formatted DataFrame (for display purposes)
    """
    if formatter is None:
        formatter = DEFAULT_FORMATTER

    # Extract soaker ingredients from formula
    soaker_mask = formula.index.str.contains('soaker', case=False)
    soaker_ingredients = formula[soaker_mask].copy()
    formula_without_soaker = formula[~soaker_mask].copy()

    if title:
        print(f"{title}\n")
    if steps:
        print(f"{steps}\n")

    print(calc.get_batch_info())

    # Identify preferment
    preferment_map = {
        'Poolish': poolish, 'sponge': sponge, 'levain': levain,
        'Pâte Fermentée': pate_fermentee, 'Desem': desem
    }
    preferment_name, preferment = next(
        ((name, df) for name, df in preferment_map.items() if df is not None),
        (None, None)
    )

    # Build and display overall formula when a preferment is present
    if preferment is not None:
        final_dough_ingredients = formula_without_soaker[
            ~formula_without_soaker.index.str.lower().isin(PREFERMENT_NAMES)
        ].copy()
        overall = pd.concat([final_dough_ingredients[['grams']], preferment[['grams']]])
        overall = overall.groupby(overall.index).sum()

        flour_mask = overall.index.str.contains('flour', case=False)
        total_flour_grams = overall[flour_mask]['grams'].sum()
        overall['baker%'] = overall['grams'] / total_flour_grams * 100
        overall['oz'] = overall['grams'] / GRAMS_PER_OUNCE

        # Reorder: flours first, then rest
        overall = pd.concat([overall[flour_mask], overall[~flour_mask]])

        # Scale grams/oz for multi-batch (baker% stays the same)
        if calc.num_batches > 1:
            overall['grams'] = overall['grams'] * calc.num_batches
            overall['oz'] = overall['grams'] / GRAMS_PER_OUNCE

        print(f"overall formula total = {overall['baker%'].sum():.1f}%\n")
        overall_display = overall[['baker%', 'grams', 'oz']]
        if show_volume:
            overall_display = _add_volume_column(overall_display)
        _print_table(overall_display, formatter, "Overall Formula")
        print()
    else:
        print(f"overall formula total = {formula_without_soaker['baker%'].sum():.1f}%\n")

    # Display soaker ingredients if present
    if not soaker_ingredients.empty:
        soaker_display = soaker_ingredients
        if show_volume:
            soaker_display = _add_volume_column(soaker_display)
        _print_table(soaker_display, formatter, "Soaker")
        print("\n")

    # Display preferment (scaled for multi-batch, reserved seed, and waste)
    if preferment is not None:
        display_preferment = preferment
        num_batches = calc.num_batches
        preferment_waste_factor = calc.preferment_waste_factor

        needs_scaling = (num_batches > 1 or reserved_seed_grams > 0
                         or preferment_waste_factor > 0)
        if needs_scaling:
            base_total = preferment['grams'].sum()
            scaled_total = ((base_total * num_batches + reserved_seed_grams)
                            * (1 + preferment_waste_factor))
            scale = scaled_total / base_total
            display_preferment = preferment.copy()
            display_preferment['grams'] = preferment['grams'] * scale
            display_preferment['oz'] = display_preferment['grams'] / GRAMS_PER_OUNCE

        preferment_label = preferment_name
        if num_batches > 1:
            preferment_label = f"{preferment_name} (all {num_batches} batches)"

        if show_volume:
            display_preferment = _add_volume_column(display_preferment)
        _print_table(display_preferment, formatter, preferment_label)

        if num_batches > 1:
            print(f"\nFinal Dough (per batch):\n")
        else:
            print(f"\nFinal Dough:\n")

    # Display final dough (excluding zero-weight rows)
    final_display = formula_without_soaker
    if show_volume:
        final_display = _add_volume_column(final_display)
    display_formula = _format_table(final_display, formatter)
    display_formula = display_formula[display_formula['grams'].str.replace(',', '').astype(float) != 0]
    colalign = ["left"] + ["right"] * len(display_formula.columns)
    print(display_formula.to_markdown(floatfmt=".2f", colalign=colalign))

def format_and_display_cost(cost_df: pd.DataFrame,
                            calc: RecipeCalculator,
                            title: str = "Cost Breakdown") -> pd.DataFrame:
    """
    Format and display cost breakdown as markdown

    Args:
        cost_df: DataFrame from calculate_cost() with cost information
        calc: RecipeCalculator instance for batch info
        title: Optional title to print before the table

    Returns:
        Formatted DataFrame (for display purposes)
    """
    display_df = cost_df.copy()

    # Format numeric columns
    if 'grams' in display_df.columns:
        display_df['grams'] = display_df['grams'].apply(
            lambda x: format_significant_digits(x) if pd.notna(x) else ''
        )
    if 'unit_cost_per_gram' in display_df.columns:
        display_df['unit_cost_per_gram'] = display_df['unit_cost_per_gram'].apply(
            lambda x: f"${x:.6f}" if pd.notna(x) else ''
        )
    if 'total_cost' in display_df.columns:
        display_df['total_cost'] = display_df['total_cost'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else ''
        )

    if title:
        print(f"\n{title}\n")

    print(calc.get_batch_info())
    print()

    print(display_df.to_markdown(index=False, colalign=["left", "right", "right", "right"]))