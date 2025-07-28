import pandas as pd
from typing import Dict, Optional

GRAMS_PER_OUNCE = 28.3495

DEFAULT_FORMATTER = {
    'grams': lambda x: f"{x:,.0f}" if x > 10 else f"{x:.2f}",
    'oz': lambda x: f"{x:,.0f}" if x > 100 else f"{x:.2f}",
    'baker%': lambda x: f"{x:.0f}" if x > 2 else f"{x:.2f}"
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

class PrefermentCalculator(RecipeCalculator):
    """Calculator for recipes with preferments (poolish, levain, sponge)"""
    
    def get_flour_pct(self, formula):
        """
        Calculate total flour percentage from a formula DataFrame
        
        Args:
            formula: DataFrame with ingredients and baker's percentages
            
        Returns:
            Total flour percentage
        """
        return formula[formula.index.str.contains('flour')]['baker%'].sum()
    
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
        formula_total = formula_df['baker%'].sum()
        poolish_total = poolish_df['baker%'].sum()
        
        # Calculate poolish weight
        flour_weight = self.get_flour_pct(formula_df) / formula_total * self.total_weight
        poolish_flour_weight = flour_weight * pre_fermented_flour_ratio
        poolish_weight = poolish_total  / self.get_flour_pct(poolish_df) * poolish_flour_weight
        
        # Calculate poolish ingredients
        poolish_result = poolish_df.copy()
        poolish_result['grams'] = poolish_result['baker%'] * poolish_weight / poolish_total
        poolish_result['oz'] = poolish_result['grams'] / GRAMS_PER_OUNCE
        
        # Calculate final dough (subtract poolish ingredients)
        final_dough = formula_df.copy()
        final_dough['grams'] = final_dough['baker%'] * self.total_weight / formula_total
        final_dough.loc['poolish', 'grams'] = poolish_weight
        final_dough.loc['poolish', 'baker%'] = poolish_weight / flour_weight * 100
        
        # Subtract poolish ingredients from final dough
        for index, row in poolish_result.iterrows():
            if index in final_dough.index:
                final_dough.at[index, 'grams'] = final_dough.at[index, 'grams'] - row['grams']
        
        final_dough['oz'] = final_dough['grams'] / GRAMS_PER_OUNCE
        
        return poolish_result, final_dough
    
    def calculate_sponge_recipe(self, sponge_df: pd.DataFrame, 
                              dough_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate sponge and final dough for sponge-based recipes (like bagels)
        
        Args:
            sponge_df: Sponge ingredients and percentages
            dough_df: Final dough ingredients and percentages
            
        Returns:
            Tuple of (sponge_df_with_weights, final_dough_df_with_weights)
        """
        sponge_total = sponge_df['baker%'].sum()
        dough_total = dough_df['baker%'].sum()
        
        # Calculate final dough weight based on total percentages
        final_dough_weight = self.total_weight * 100 / (dough_total + sponge_total)
        
        # Calculate sponge weights
        sponge_result = sponge_df.copy()
        sponge_result['grams'] = sponge_result['baker%'] * final_dough_weight / 100
        sponge_result['oz'] = sponge_result['grams'] / GRAMS_PER_OUNCE
        
        # Calculate final dough weights  
        dough_result = dough_df.copy()
        dough_result['grams'] = dough_result['baker%'] * final_dough_weight / 100
        dough_result['oz'] = dough_result['grams'] / GRAMS_PER_OUNCE
        
        return sponge_result, dough_result

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
    
    # Create display copy with formatted values
    display_formula = formula.copy()
    for col, fmt in formatter.items():
        if col in display_formula.columns:
            display_formula[col] = display_formula[col].apply(fmt)

    if title:
        print(f"{title}\n")
    if steps:
        print(f"{steps}\n")

    print(calc.get_batch_info())
    formula_total = formula['baker%'].sum()
    print(f"overall formula total = {formula_total:.1f}%\n")
        
    if poolish is not None:
        poolish_display = poolish.copy()
        for col, fmt in formatter.items():
            if col in poolish_display.columns:
                poolish_display[col] = poolish_display[col].apply(fmt)
        print("Poolish:\n")
        print(poolish_display.to_markdown())
        print("\nFinal Dough:\n")

    print(display_formula.to_markdown())
    
    return display_formula