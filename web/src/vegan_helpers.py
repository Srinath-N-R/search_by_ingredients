import re
import logging
import unicodedata
import numpy as np
import os

from keto_helpers import parse_ingredient, _embedding, pick_usda_hit_cached, fetch_food_cached


# Configure logging - keep it simple
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'  # Simplified format
)
# Silence noisy loggers
logging.getLogger('opensearchpy').setLevel(logging.ERROR)  # Only show errors
logging.getLogger('urllib3').setLevel(logging.ERROR)       # Only show errors
logging.getLogger('opensearch').setLevel(logging.ERROR)    # Only show errors
logger = logging.getLogger(__name__)


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
        logger.info(f"{name}'s closest ingredient is: {closest} | score: {best_score:.3f}")

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
        ingredients = [ing for ing in ingredients if len(ing) > 2]
        return ingredients

    if "inputFoods" in record:
        ingredients = [item.get("foodDescription", "") for item in record["inputFoods"]]
        ingredients = [ing for ing in ingredients if len(ing) > 2]
        return ingredients

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
    seen = {}
    for raw_line in ingredient_lines:
        try:
            if verbose:
                logger.info(raw_line)

            name = parse_ingredient(raw_line)["ingredient"]

            # USDA lookup (cached)
            fdc_id, _ = pick_usda_hit_cached(name)
 
            if fdc_id:
                record = fetch_food_cached(int(fdc_id))
                sub_ingredients = _usda_ingredient_list(record) or [name]
                if not sub_ingredients:
                    sub_ingredients = [name]
            else:
                sub_ingredients = [name]

            # Vegan test for this sub-ingredients
            for sub in sub_ingredients:
                if sub not in seen:
                    _is_vegan = _is_ingredient_vegan(sub, verbose)
                    seen[sub] = _is_vegan
                    if verbose:
                        if _is_vegan:
                            logger.info(f"---> INGREDIENT: {sub} flagged as vegan")
                        else:
                            logger.info(f"---> INGREDIENT: {name} flagged as non-vegan due to '{sub}'")
                else:
                    _is_vegan = seen[sub]

                if not _is_vegan:
                    return False

        except Exception as e:
            logger.warning(f'{e}')
    return True