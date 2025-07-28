import pandas as pd
from typing import Dict, Optional

GRAMS_PER_OUNCE = 28.3495

DEFAULT_FORMATTER = {
    'grams': lambda x: f"{x:,.0f}" if x > 10 else f"{x:.2f}",
    'oz': lambda x: f"{x:,.0f}" if x > 100 else f"{x:.2f}",
    'baker%': lambda x: f"{x:.0f}" if x > 2 else f"{x:.1f}"
}

class RecipeCalculator:
    """Base calculator for all bread recipes"""
    
    def __init__(self, num_loaves: int, weight_pounds: int = 0, 
                 weight_ounces: float = 0, weight_grams: float = 0):
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
        self.total_weight = num_loaves * self.loaf_weight
    
    def get_batch_info(self):
        return f"{self.num_loaves:,.0f} loaves at {self.loaf_weight:,.0f} grams"

    def print_batch_info(self):
        """Print batch size information"""
        print(self.get_batch_info())
        print(f"total weight: {self.total_weight:,.2f} grams")
    
    def calculate_straight_dough(self, formula_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weights for straight dough recipes (no preferments)
        
        Args:
            formula_df: DataFrame with ingredients and baker's percentages
            
        Returns:
            DataFrame with baker%, grams, and oz columns
        """

        formula_total = formula_df['baker%'].sum() 
        # Create a copy to avoid modifying the original
        result_df = formula_df.copy()
        result_df['grams'] = result_df['baker%'] * self.total_weight / formula_total
        result_df['oz'] = result_df['grams'] / GRAMS_PER_OUNCE
        
        return result_df

def create_formula(ingredients_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Create DataFrame from ingredients dictionary
    
    Args:
        ingredients_dict: Dictionary mapping ingredient names to baker's percentages
        
    Returns:
        DataFrame with ingredients as index and baker% column
    """
    return pd.DataFrame.from_dict(ingredients_dict, orient='index', columns=["baker%"])

def format_and_display(df: pd.DataFrame, calc: RecipeCalculator, formatter: Dict = None, title: str = "", steps: str = "") -> pd.DataFrame:
    """
    Format DataFrame and display as markdown
    
    Args:
        df: DataFrame to format and display
        formatter: Dictionary of column formatters (uses DEFAULT_FORMATTER if None)
        title: Optional title to print before the table
        
    Returns:
        Formatted DataFrame (for display purposes)
    """
    if formatter is None:
        formatter = DEFAULT_FORMATTER
    
    # Create display copy with formatted values
    display_df = df.copy()
    for col, fmt in formatter.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt)
    
    if title:
        print(f"{title}\n")
    if steps:
        print(f"{steps}\n")

    print(calc.get_batch_info())
    formula_total = df['baker%'].sum()
    print(f"overall formula total = {formula_total:.1f}%\n")
    print(display_df.to_markdown())
    
    return display_df