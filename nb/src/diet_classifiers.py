import json
import sys
from argparse import ArgumentParser
from typing import List
from time import time
import pandas as pd
import re
try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")


from keto_helpers import (parse_ingredient, _as_float, pick_usda_hit_cached, fetch_food_cached, get_macronutrients,
                        override_macronutrients, _unit_to_grams, estimate_weight, _scale_macros)

from CONSTANTS import _ZERO_NET


def is_ingredient_vegan(ingredient: str) -> bool:
    # TODO: Implement
    return False



def np_style_str_to_list(txt: str) -> list[str]:
    """
    Convert a NumPy-style array string to Python list of strings.

    s = "['3 cups flour' '2 cups sugar' '6 tbsp cocoa']"
    np_style_str_to_list(s)
    # → ['3 cups flour', '2 cups sugar', '6 tbsp cocoa']
    """
    return re.findall(r"'([^']+)'", txt)

def is_keto(ingredients, verbose = False):

    try:
        if verbose:
            print(ingredients)
        row_macros = dict(carbs=0.0, protein=0.0, fat=0.0, fiber=0.0, calories=0.0)
    
        for line in ingredients:
            try:
                if verbose:
                    print(line)
                
                start_ = time()
                parsed = parse_ingredient(line)
                qty = _as_float(parsed["quantity"])
                unit = (parsed["unit"] or "").lower()
                ingredient_name = parsed["ingredient"]
                                
                fdc_id, usda_name = pick_usda_hit_cached(ingredient_name)
                info = fetch_food_cached(int(fdc_id))
                macros = get_macronutrients(info)
                macros = override_macronutrients(macros, ingredient_name)

                if ingredient_name in _ZERO_NET:
                    macros = dict(carbs_g=0, protein_g=0, fat_g=0, fiber_g=0, calories=0, basis_g=1, source="override")
        
                g_per_unit = _unit_to_grams(info, unit)
                if g_per_unit is None:
                    g_per_unit = estimate_weight(ingredient_name)
        
                grams_needed = qty * g_per_unit
                scaled = _scale_macros(macros, grams_needed)
        
                row_macros["carbs"] += scaled["carbs_g"]
                row_macros["protein"] += scaled["protein_g"]
                row_macros["fat"] += scaled["fat_g"]
                row_macros["fiber"] += scaled["fiber_g"]
                row_macros["calories"] += scaled["calories"]
                end_ = time()
                if verbose:
                    print("Line Name: ", ingredient_name)
                    print("USDA Name: ", usda_name)
                    print("quantity: ", qty, unit)
                    print("Estimated Weight (g): ", grams_needed)
                    print("Basis (g): ", macros["basis_g"])
                    print("Raw Carbs: ", macros["carbs_g"])
                    print("Scaled Carbs: ", scaled["carbs_g"])
                    print("Raw Protein: ", macros["protein_g"])
                    print("Scaled Protein: ", scaled["protein_g"])
                    print("Raw Fat: ", macros["fat_g"])
                    print("Scaled Fat: ", scaled["fat_g"])
                    print("Raw Fiber: ", macros["fiber_g"])
                    print("Scaled Fiber: ", scaled["fiber_g"])
                    print("Raw Calories: ", macros["calories"])
                    print("Scaled Calories: ", scaled["calories"])
                    print("Time Taken: ", end_ - start_)
                    print()
                    
            except Exception as e:
                print(e)
                continue
    
        net_carbs = max(row_macros["carbs"] - row_macros["fiber"], 0)
        carb_pct = (
            (net_carbs * 4) / row_macros["calories"] * 100 if row_macros["calories"] else 0
        )
        is_this_keto = (carb_pct <= 20)
    
        if verbose:
            print("total carbs: ", row_macros["carbs"])
            print("total protein: ", row_macros["protein"])
            print("total fat: ", row_macros["fat"])
            print("total fiber: ", row_macros["fiber"])
            print("is keto: ", is_this_keto)
            print()
            print()
    
        return is_this_keto
    except Exception as e:
        print(e)
        return False



def is_ingredient_vegan(ingredient):
    # TODO: complete
    return False    


def is_vegan(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_vegan, ingredients))


def main(args):
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)

    try:
        start_time = time()
        ground_truth['keto_pred'] = ground_truth['ingredients'].apply(np_style_str_to_list).apply(is_keto)
        ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(is_vegan)

        end_time = time()
    except Exception as e:
        print(f"Error: {e}")
        return -1

    print("===Keto===")
    print(classification_report(
        ground_truth['keto'], ground_truth['keto_pred']))
    print("===Vegan===")
    print(classification_report(
        ground_truth['vegan'], ground_truth['vegan_pred']))
    print(f"== Time taken: {end_time - start_time} seconds ==")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str,
                        default="/usr/src/data/ground_truth_sample.csv")
    sys.exit(main(parser.parse_args()))
