import re
import unicodedata
import numpy as np
import os

from keto_helpers import parse_ingredient, _embedding, pick_usda_hit_cached, fetch_food_cached


BASE_DIR = os.path.dirname(__file__)
VEGAN_DEP_DIR = os.path.join(BASE_DIR, "vegan_dependencies")

EMBED_ALL = np.load(os.path.join(VEGAN_DEP_DIR, "vegan_embeddings.npy"))
VEGAN_INGREDIENTS = np.load(os.path.join(VEGAN_DEP_DIR, "vegan_ingredients.npy"))
NON_VEGAN_INGREDIENTS = np.load(os.path.join(VEGAN_DEP_DIR, "non_vegan_ingredients.npy"))

ALL_INGREDIENTS = np.concatenate([VEGAN_INGREDIENTS, NON_VEGAN_INGREDIENTS])


def clean_ingredients(txt):
    '''
    Apply regex cleaning to make embedding matching much easier.
    '''
    if not isinstance(txt, str):
        return ""

    txt = txt.lower()

    # Remove bracketed content that contains digits
    txt = re.sub(r"\([^\)]*\d+[^\)]*\)", "", txt)
    txt = re.sub(r"\[[^\]]*\d+[^\]]*\]", "", txt)
    txt = re.sub(r"\{[^\}]*\d+[^\}]*\}", "", txt)

    # Remove any leftover brackets
    txt = re.sub(r"[\[\]\(\)\{\}]", "", txt)

    # Normalize accents: e.g., é → e
    txt = unicodedata.normalize("NFKD", txt).encode("ASCII", "ignore").decode("utf-8")

    # Remove bullets, asterisks, punctuation
    txt = re.sub(r"[•*\.\"\'\:;!?,%-]", "", txt)

    # Remove long digit sequences (e.g., scan codes, years)
    txt = re.sub(r"\b\d{3,}\b", "", txt)

    if len(txt) <= 2:
        txt = ""

    # Normalize whitespace
    txt = re.sub(r"\s+", " ", txt).strip()

    return txt



def _is_ingredient_vegan(name, verbose = False):
    """
    Embedding-based matching for whether a single ingredient is vegan.
    In our embedding matrix, indices < 4310 correspond to vegan examples
    """
    cleaned_name = clean_ingredients(name)
    vec = _embedding(cleaned_name)
    scores = EMBED_ALL @ vec
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if verbose:
        closest = ALL_INGREDIENTS[best_idx]
        print(f"{name}'s closest ingredient is: {closest} | score: {best_score:.3f}")

    # We return an answer only if the score is above a confidence threshold of 0.5
    if best_score >= 0.5:
        return best_idx < 4313
    
    return True


def _usda_ingredient_list(record):
    """
    Normalise the various ways USDA can store an ingredient list.
    """
    if isinstance(record.get("ingredients"), str):
        ingredients = record["ingredients"].split(", ")
        if len(ingredients) == 1:
            ingredients = record["ingredients"].split("; ")
        return ingredients

    if "inputFoods" in record:
        return [item.get("foodDescription", "") for item in record["inputFoods"]]

    return []


def is_vegan(ingredient_lines, verbose = False):
    """
    Decide if all ingredients in a recipe are vegan.

    Workflow (high-level)

    1.  Parse each raw line (qty, unit, ingredient, preperation)
    2.  Query USDA for that name.
        – If USDA returns a record, test every sub-ingredient it lists.
        – If USDA has no record (or gives no list), fall back to the name itself.
    3.  The first non-vegan hit short-circuits the loop and returns False.
    """
    for raw_line in ingredient_lines:
        name = parse_ingredient(raw_line)["ingredient"]

        # USDA lookup (cached)
        fdc_id, _ = pick_usda_hit_cached(name)
        if fdc_id:
            record = fetch_food_cached(int(fdc_id))
            sub_ingredients = _usda_ingredient_list(record) or [name]
        else:
            sub_ingredients = [name]

        # Vegan test for this sub-ingredients
        for sub in sub_ingredients:
            _is_vegan = _is_ingredient_vegan(sub, verbose)
            if verbose:
                if _is_vegan:
                    print(f"---> INGREDIENT: {sub} flagged as vegan")
                else:
                    print(f"---> INGREDIENT: {name} flagged as non-vegan due to '{sub}'")
                print()
            if not _is_vegan:
                return False

    return True