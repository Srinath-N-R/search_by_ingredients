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


from keto_helpers import is_keto


def is_ingredient_vegan(ingredient: str) -> bool:
    # TODO: Implement
    return False



def np_style_str_to_list(txt):
    '''
    Convert a NumPy-style array string to Python list of strings.

    '['3 cups flour' '2 cups sugar' '6 tbsp cocoa']' ---> ['3 cups flour', '2 cups sugar', '6 tbsp cocoa']
    '''
    return re.findall(r"'([^']+)'", txt)


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
