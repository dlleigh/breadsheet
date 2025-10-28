import pandas as pd
from typing import Dict, Optional
import math

GRAMS_PER_OUNCE = 28.3495
GRAMS_PER_POUND = GRAMS_PER_OUNCE * 16

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
                 waste_factor: float = 0.0):
        """
        Initialize recipe calculator
        
        Args:
            num_loaves: Number of loaves/units to make
            weight_pounds: Weight per unit in pounds
            weight_ounces: Weight per unit in ounces
            weight_grams: Weight per unit in grams
        """
        self.num_loaves = num_loaves
        self.loaf_weight = (weight_pounds * 16 + weight_ounces) * GRAMS_PER_OUNCE + weight_grams
        self.total_weight = num_loaves * self.loaf_weight * (1 + waste_factor)
    
    def get_batch_info(self):
        min_loaf_weight = self.loaf_weight * 0.95
        max_loaf_weight = self.loaf_weight * 1.05
        return f"{self.num_loaves:,.0f} loaves at {min_loaf_weight:,.0f}-{max_loaf_weight:,.0f} grams"

    def print_batch_info(self):
        """Print batch size information"""
        print(self.get_batch_info())
        print(f"total weight: {self.total_weight:,.2f} grams")

    def get_flour_pct(self, formula):
        """
        Calculate total flour percentage from a formula DataFrame
        
        Args:
            formula: DataFrame with ingredients and baker's percentages
            
        Returns:
            Total flour percentage
        """
        return formula[formula.index.str.contains('flour', case=False)]['baker%'].sum()
    
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
        for index, row in preferment_result.iterrows():
            if index in final_dough.index:
                final_dough.at[index, 'grams'] = final_dough.at[index, 'grams'] - row['grams']

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

class PrefermentCalculator(RecipeCalculator):
    """Calculator for recipes with preferments (poolish, levain, sponge)"""
    

    
    def calculate_poolish_recipe(self, formula_df: pd.DataFrame, poolish_df: pd.DataFrame,
                                 pre_fermented_flour_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate poolish and final dough for poolish-based recipes

        Args:
            formula_df: Main recipe formula
            poolish_df: Poolish ingredients and percentages
            pre_fermented_flour_ratio: What fraction of total flour goes into poolish (e.g., 0.2 = 20%)

        Returns:
            Tuple of (poolish_df_with_weights, final_dough_df)
        """
        return self.calculate(formula_df, poolish_df=poolish_df,
                            pre_fermented_flour_ratio=pre_fermented_flour_ratio)

    def calculate_sponge_recipe(self, formula_df: pd.DataFrame, sponge_df: pd.DataFrame,
                                 pre_fermented_flour_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate sponge and final dough for sponge-based recipes

        Args:
            formula_df: Main recipe formula
            sponge_df: Sponge ingredients and percentages
            pre_fermented_flour_ratio: What fraction of total flour goes into sponge (e.g., 0.2 = 20%)

        Returns:
            Tuple of (sponge_df_with_weights, final_dough_df)
        """
        return self.calculate(formula_df, sponge_df=sponge_df,
                            pre_fermented_flour_ratio=pre_fermented_flour_ratio)

    def calculate_sourdough_recipe(self, formula_df: pd.DataFrame, levain_df: pd.DataFrame,
                                 pre_fermented_flour_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate levain and final dough for sourdough recipes

        Args:
            formula_df: Main recipe formula
            levain_df: Levain ingredients and percentages
            pre_fermented_flour_ratio: What fraction of total flour goes into levain (e.g., 0.2 = 20%)

        Returns:
            Tuple of (levain_df_with_weights, final_dough_df)
        """
        return self.calculate(formula_df, levain_df=levain_df,
                            pre_fermented_flour_ratio=pre_fermented_flour_ratio)

def create_formula(ingredients_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Create DataFrame from ingredients dictionary
    
    Args:
        ingredients_dict: Dictionary mapping ingredient names to baker's percentages
        
    Returns:
        DataFrame with ingredients as index and baker% column
    """
    return pd.DataFrame.from_dict(ingredients_dict, orient='index', columns=["baker%"])

def format_and_display(formula: pd.DataFrame, calc: RecipeCalculator, poolish: pd.DataFrame = None,
                       sponge: pd.DataFrame = None,
                       levain: pd.DataFrame = None,
                       pate_fermentee: pd.DataFrame = None,
                       desem: pd.DataFrame = None,
                       formatter: Dict = None, title: str = "", steps: str = "") -> pd.DataFrame:
    """
    Format DataFrame and display as markdown
    
    Args:
        formula: DataFrame to format and display
        formatter: Dictionary of column formatters (uses DEFAULT_FORMATTER if None)
        title: Optional title to print before the table
        
    Returns:
        Formatted DataFrame (for display purposes)
    """
    if formatter is None:
        formatter = DEFAULT_FORMATTER

    # Extract soaker ingredients from formula
    soaker_mask = formula.index.str.contains('soaker', case=False)
    soaker_ingredients = formula[soaker_mask].copy()
    formula_without_soaker = formula[~soaker_mask].copy()

    # Create display copy with formatted values
    display_formula = formula_without_soaker.copy()
    for col, fmt in formatter.items():
        if col in display_formula.columns:
            display_formula[col] = display_formula[col].apply(fmt)

    if title:
        print(f"{title}\n")
    if steps:
        print(f"{steps}\n")

    print(calc.get_batch_info())
    if sponge is None and pate_fermentee is None and desem is None:
        formula_total = formula_without_soaker['baker%'].sum()
    elif sponge is not None:
        formula_total = sponge['baker%'].sum() + formula_without_soaker['baker%'].sum()
    elif pate_fermentee is not None:
        formula_total = pate_fermentee['baker%'].sum() + formula_without_soaker['baker%'].sum()
    elif desem is not None:
        formula_total = desem['baker%'].sum() + formula_without_soaker['baker%'].sum()
    else:
        formula_total = formula_without_soaker['baker%'].sum()
    print(f"overall formula total = {formula_total:.1f}%\n")

    # Display soaker ingredients if present
    if not soaker_ingredients.empty:
        soaker_display = soaker_ingredients.copy()
        for col, fmt in formatter.items():
            if col in soaker_display.columns:
                soaker_display[col] = soaker_display[col].apply(fmt)
        print("Soaker:\n")
        print(soaker_display.to_markdown(floatfmt=".2f",colalign=["left", "right", "right", "right"]))
        print("\n")
        
    if poolish is not None:
        poolish_display = poolish.copy()
        for col, fmt in formatter.items():
            if col in poolish_display.columns:
                poolish_display[col] = poolish_display[col].apply(fmt)
        print("Poolish:\n")
        print(poolish_display.to_markdown(floatfmt=".2f",colalign=["left", "right", "right", "right"]))
        print("\nFinal Dough:\n")
        
    if sponge is not None:
        sponge_display = sponge.copy()
        for col, fmt in formatter.items():
            if col in sponge_display.columns:
                sponge_display[col] = sponge_display[col].apply(fmt)
        print("sponge:\n")
        print(sponge_display.to_markdown(floatfmt=".2f",colalign=["left", "right", "right", "right"]))
        print("\nFinal Dough:\n")

    if levain is not None:
        levain_display = levain.copy()
        for col, fmt in formatter.items():
            if col in levain_display.columns:
                levain_display[col] = levain_display[col].apply(fmt)
        print("levain:\n")
        print(levain_display.to_markdown(floatfmt=".2f",colalign=["left", "right", "right", "right"]))
        print("\nFinal Dough:\n")

    if pate_fermentee is not None:
        pate_fermentee_display = pate_fermentee.copy()
        for col, fmt in formatter.items():
            if col in pate_fermentee_display.columns:
                pate_fermentee_display[col] = pate_fermentee_display[col].apply(fmt)
        print("Pâte Fermentée:\n")
        print(pate_fermentee_display.to_markdown(floatfmt=".2f",colalign=["left", "right", "right", "right"]))
        print("\nFinal Dough:\n")

    if desem is not None:
        desem_display = desem.copy()
        for col, fmt in formatter.items():
            if col in desem_display.columns:
                desem_display[col] = desem_display[col].apply(fmt)
        print("Desem:\n")
        print(desem_display.to_markdown(floatfmt=".2f",colalign=["left", "right", "right", "right"]))
        print("\nFinal Dough:\n")

    display_formula = display_formula[display_formula['grams'].str.replace(',', '').astype(float) != 0]
    print(display_formula.to_markdown(floatfmt=".2f",colalign=["left", "right", "right", "right"]))

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