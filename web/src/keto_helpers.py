import fractions
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import os
import json
import logging
import pathlib
import hashlib
from time import time

from CONSTANTS import (_MASS_UNITS, _NUMBER_RE, _FALLBACK_GRAMS, __CAT_WEIGHTS_NAMES, 
                       __CAT_WEIGHTS, __CAT_MACROS_NAMES, __CAT_MACROS, _ZERO_NET)


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

_LOOKUP_DIR = pathlib.Path(BASE_DIR) / "usda_lookup_cache"
_LOOKUP_DIR.mkdir(exist_ok=True)

_CACHE_DIR = pathlib.Path(BASE_DIR) / "usda_json_cache"
_CACHE_DIR.mkdir(exist_ok=True)

API_KEY = "MdvR7gqeWDCMVDsspxgLNGqkjU8HNFNe3yoXlYER"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_EMB_MODEL = SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1024)
def _embedding(text):
    return _EMB_MODEL.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]


def parse_ingredient(line: str) -> dict:
    '''
    Each line consists of quantity, unit, ingredient and preparation. 
    Rather than using NER, we are using a navie regex based approach to parse out quantity, unit, ingredient and preparation.

    Ex:
    '2 (16 ounce) packages fresh mushrooms, stems removed' ----> (quantity='16', unit='ounce', ingredient='fresh mushrooms', preparation='stems removed')
    '3 cups shredded cheddar' ----> (quantity='3', unit='cup', ingredient='shredded cheddar', preparation='')
    '''
    # If there is a parenthetical, see if there are weight spefications inside it.
    inner_qty = inner_unit = None
    m_paren = re.search(r'\(([^)]+)\)', line)
    if m_paren:
        inner_txt = m_paren.group(1).strip()
        m_inner = re.match(r'([\d/.\s]+)\s+(\w+)', inner_txt)
        if m_inner:
            inner_qty, inner_unit = m_inner.groups()

    # We'll parse the text after removing parantheticals
    clean = re.sub(r'\([^)]*\)', '', line)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Parse Quantity
    q_match = re.match(r'^(\d+\s\d+/\d+|\d+/\d+|\d+)', clean)
    outer_qty = q_match.group(0) if q_match else ''
    rest = clean[len(outer_qty):].lstrip() if outer_qty else clean

    # Parse Unit
    outer_unit = ''
    rest_lc = rest.lower()
    for u in _MASS_UNITS:
        if rest_lc.startswith(u + ' '):
            outer_unit, rest = u, rest[len(u):].lstrip()
            break
        if rest_lc == u:
            outer_unit, rest = u, ''
            break

    # Parse Ingredient
    # For parsing out ingredient from preparation, we are simply splitting on the first comma - a naive rule.
    # Of course this is not ideal, but since this is a structured dataset, it gets us most of the way to the answer.
    parts = [p.strip() for p in rest.split(',', 1)]
    ingredient = parts[0]
    preparation = parts[1] if len(parts) > 1 else ''

    # Weight specifications found inside parantheticals can override weight specifications found outside parantheticals.
    quantity = (inner_qty or outer_qty).strip()
    unit     = (inner_unit or outer_unit).lower().rstrip('s')

    return {
        "quantity":    quantity,
        "unit":        unit,
        "ingredient":  ingredient,
        "preparation": preparation,
    }



def search_food(query, page_size = 20, alpha = 0.3):
    '''
    Searches food candidates from USDA API.
    Searches food from SR Legacy item type. These would return generic items and would contain macronutrient values on the basis of 100g.
    Then candidates are ranked using a hybrid approach which combines USDA search score and query-result sentence embedding similarity.
    '''
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params_init = {
        "api_key": API_KEY,
        "query": query,
        "pageSize": page_size,
        "dataType": ["SR Legacy"],
    }


    foods = requests.get(url, params=params_init).json().get("foods", [])

    q_emb = _embedding(query)

    rows = []
    for f in foods:
        sim = float(np.dot(q_emb, _embedding(f["description"])))
        hybr = ((alpha * f["score"]) / 1000) + (1 - alpha) * sim * 100
        rows.append([f["fdcId"], f["description"], f["score"], sim, hybr, f["dataType"]])

    rows.sort(key=lambda r: -r[-2])

    return rows



def search_food_catch_all(query, page_size = 20, alpha = 0.3):
    '''
    Searches food candidates from USDA API.
    Searches food from all item types. These would return all items (including branded items). The macronutrient values for branded items would on the basis of custom       
    serving sizes as opposed to the standard 100 grams.
    Then candidates are ranked using a hybrid approach which combines USDA search score and query-result sentence embedding similarity.
    
    '''
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"

    params_catch_all = {
        "api_key": API_KEY,
        "query": query,
        "pageSize": page_size,
        "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)", "Branded"],
    }

    foods = requests.get(url, params=params_catch_all).json().get("foods", [])

    q_emb = _embedding(query)

    rows = []
    for f in foods:
        sim = float(np.dot(q_emb, _embedding(f["description"])))
        hybr = ((alpha * f["score"]) / 1000) + (1 - alpha) * sim * 100
        rows.append([f["fdcId"], f["description"], f["score"], sim, hybr, f["dataType"]])

    rows.sort(key=lambda r: -r[-2])

    return rows


def pick_usda_hit(query):
    '''
    Returns the best USDA fdc_id, usda_name after searching from USDA API.
    Search is designed to hit SR-Legacy as much as possible. This keeps the search less noisy. And keeps the macronutrient calculation very simple.
    The least common is BRANDED.
    '''

    ranked = search_food(query, page_size=20)

    # If there is no SR_Legacy search results or if the top result isn't similar to the search query, we run the catch all search.
    if not ranked or not ranked[0][-3] > 0.7:
        ranked = search_food_catch_all(query, page_size=20)    

    if not ranked:
        return None, None

    return ranked[0][:2]


def pick_usda_hit_cached(ingredient):
    '''
    Caches the fdc_id and usda_name for the query ingredient after searching it using pick_usda_hit.
    If there is a cache hit, there is no need for an API call.
    If no cache hit, it will call the API and save the result to disk.
    '''

    key = ingredient.strip().lower()
    file = _LOOKUP_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.json"

    if file.exists():
        return tuple(json.loads(file.read_text()))

    fdc_id, desc = pick_usda_hit(ingredient)
    file.write_text(json.dumps([fdc_id, desc]))
    return fdc_id, desc


@lru_cache(maxsize=4096)
def fetch_food(fdc_id: int):
    '''
    For a given USDA fdc_id, we fetch all information on that ingredient using the USDA API.
    '''
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={API_KEY}"
    return requests.get(url, timeout=10).json()


def fetch_food_cached(fdc_id):
    '''
    Caches the result for the fcid after running fetch_food function.
    If there is a cache hit, there is no need for an API call.
    If no cache hit, it will call the API and save the result to disk.
    '''
    path = _CACHE_DIR / f"{fdc_id}.json"
    if path.exists():
        return json.loads(path.read_text())

    info = fetch_food(fdc_id)
    path.write_text(json.dumps(info))
    return info


def get_macronutrients(info: dict) -> dict:
    '''
    Returns a dict with carbs_g, protein_g, fat_g, fiber_g, calories, basis_g.
    SR-Legacy and most item types have caloric data per 100-g.
    Branded item types have caloric data per custom serving sizes. 
    The caloric data per custom serving sizes can be reliably converted to caloric data per grams. 
    '''
    m = {
        "carbs_g":   None,
        "protein_g": None,
        "fat_g":     None,
        "fiber_g":   None,
        "calories":  None,
        "basis_g":   100.0,
        "source":    "foodNutrients",
    }

    # Branded Items have custom serving sizes.
    # We convert these custom serving sizes to gram equivalents.
    # Only Branded Items have labelNutrients.
    if "labelNutrients" in info:
        unit = str(info.get("servingSizeUnit", "")).strip().lower()
        size = float(info.get("servingSize") or 1.0)

        # Convert recognised units to grams, fallbacks to 100 grams for UNK units.
        g_per_unit = _FALLBACK_GRAMS.get(unit)
        if g_per_unit is None:
            if unit in _FALLBACK_GRAMS:
                g_per_unit = _FALLBACK_GRAMS[unit]
            else:
                g_per_unit = 100

        mass_g = size * g_per_unit

        ln = info["labelNutrients"]
        m.update(
            carbs_g   = ln.get("carbohydrates", {}).get("value"),
            protein_g = ln.get("protein", {}).get("value"),
            fat_g     = ln.get("fat", {}).get("value"),
            fiber_g   = ln.get("fiber", {}).get("value"),
            calories  = ln.get("calories", {}).get("value"),
            basis_g   = mass_g,
            source    = "labelNutrients",
        )
        return m

    # Foundation / SR Legacy results which we use for most of the searches – per 100 g 
    for n in info.get("foodNutrients", []):
        name  = n.get("nutrient", {}).get("name", "").lower()
        value = n.get("amount")
        if value is None:
            continue
        if "carbohydrate" in name and m["carbs_g"] is None:
            m["carbs_g"] = value
        elif name == "protein" and m["protein_g"] is None:
            m["protein_g"] = value
        elif (("total lipid" in name) or name == "fat") and m["fat_g"] is None:
            m["fat_g"] = value
        elif "fiber" in name and m["fiber_g"] is None:
            m["fiber_g"] = value
        elif "energy" in name and "kcal" in n["nutrient"].get("unitName", "").lower():
            m["calories"] = value
    return m


def override_macronutrients(macros, ingredient_name):
    '''
    Stuff like water, ice cubes, sweetners, and small spices can be reliable over-ridden to 0 calories with 0 macro-nutrient values.
    '''
    if ingredient_name in _ZERO_NET:
        macros = dict(carbs_g=0, protein_g=0, fat_g=0, fiber_g=0, calories=0, basis_g=1, source="override")
    return macros


# precomputing embeddings for CATEGORIES of items.
_EMBS_MACROS = np.stack([_embedding(CAT) for CAT in __CAT_MACROS_NAMES])

def estimate_macronutrients(item):
    '''
    Estimates the macronutrients of an item when fdc_id is not found for an item. 
    For items where there are no macros provided, we roughly estimate a macronutrient distribution for that item.
    This happens by identifying what product category an item roughly belongs to by computing item-category similarity using sentence embeddings. 
    Then, we return the macronutrients associated with that product category.

    EX:
    "Fresh herb leafy": dict(carbs_g=7,  protein_g=3,  fat_g=0.5,fiber_g=3,  calories=40).
    "Tomato-based sauce": dict(carbs_g=12, protein_g=2,  fat_g=1,  fiber_g=2,  calories=65).
    "Sweet pastry (danish/tart)": dict(carbs_g=55, protein_g=6,  fat_g=20, fiber_g=2,  calories=430).
    '''
    idx = int(np.argmax(_EMBS_MACROS @ _embedding(item)))
    template = __CAT_MACROS[idx]
    return {**template, "basis_g": 100}


def _as_float(q):
    '''
    Converts string numerical values to float values. A simple eval fn cannot be used because the numbers are represnted in fractions.
    '''
    if not q:
        return 1.0

    q = q.strip()
    try:
        if " " in q and "/" in q:
            whole, frac = q.split()
            return float(whole) + float(fractions.Fraction(frac))
        if "/" in q:
            return float(fractions.Fraction(q))
        return float(fractions.Fraction(q)) if _NUMBER_RE.match(q) else float(q)
    except Exception as e:
        logger.info(f"Could not parse quantity {q} because of {e} – defaulting to 1")
        return 1.0


# precomputing embeddings for CATEGORIES of items.
_EMBS_WEIGHTS = np.stack([_embedding(CAT) for CAT in __CAT_WEIGHTS_NAMES])


def estimate_weight(item):
    '''
    Estimates the weight of an item in grams when the unit is not given. 
    For items where there are no units specified, we roughly estimate a weight for that item.
    This happens by identifying what product category an item roughly belongs to by computing item-category similarity using sentence embeddings. 
    Then, we return the weight associated with that product category.

    EX:
    "a pinch of tiny spice like salt, pepper, yeast, cayenne, paprika, cumin, turmeric, nutmeg, cloves, oregano, cardamom, cinnamon": 2 grams.
    "a small piece of meat like chicken breast, pork loin, cod fillet, tofu slab, tempeh slice": 100 grams.
    "one large shellfish like lobster tail, king crab leg": 90 grams.
    '''
    best = int(np.argmax(_EMBS_WEIGHTS @ _embedding(item)))
    return float(__CAT_WEIGHTS[best])


def _unit_to_grams(info, unit):
    '''
    Convert the unit to gram equivalent.
    '''

    # If there is no input unit, we just return None.
    unit_lc = unit.lower().rstrip("s")
    if not unit_lc:
        return None
    
    # If there is a gram weight associated to a unit in the USDA API result, we simply use that
    # This for-loop can be potentially removed and the code would run just fine. This is just a precautionary step.
    for p in info.get("foodPortions", []):
        mu = (p.get("measureUnit", {}) or {}).get("name", "").lower().rstrip("s")
        if unit_lc == mu:
            g = p.get("gramWeight")
            if g:
                return g
        desc = str(p.get("portionDescription", "")).lower()
        if unit_lc in desc.split():
            g = p.get("gramWeight")
            if g:
                return g

    # Else we make assumptions based on known knowledge
    g = _FALLBACK_GRAMS.get(unit_lc)
    if g:
        return g

    return None


def _scale_macros(macros, grams_needed):
    '''
    Convert per basis nutrient values to the actual amount used in the recipe.
    
    If the USDA record lacks an explicit calorie value, calories are back-filled using the classic 4-4-9 rule:
    kcal = carbs*4 + protein*4 + fat*9
    '''
    # scales the macros of the ingredient to the grams observed in the recipe.
    factor = grams_needed / (macros["basis_g"] or 100)
    result = {}
    for k in ("carbs_g", "protein_g", "fat_g", "fiber_g", "calories"):
        v = macros.get(k)
        result[k] = (v * factor) if v is not None else 0.0

    # 4‑4‑9 estimate if kcal missing
    if result["calories"] == 0:
        result["calories"] = (
            result["carbs_g"] * 4 + result["protein_g"] * 4 + result["fat_g"] * 9
        )
    return result


def get_line_macros(line, row_macros, verbose):

    start_ = time()  # performance timing

    # Parse quantity / unit / name
    parsed = parse_ingredient(line)
    qty  = _as_float(parsed["quantity"])
    unit = (parsed["unit"] or "").lower()
    ingredient_name = parsed["ingredient"]

    # USDA lookup (cached)
    fdc_id, usda_name = pick_usda_hit_cached(ingredient_name)

    if not fdc_id:
        info = {}
        macros = estimate_macronutrients(ingredient_name)
    else:
        info   = fetch_food_cached(int(fdc_id))
        macros = get_macronutrients(info)  # per-basis macros dict

    # Overrides (zero-net items)
    macros = override_macronutrients(macros, ingredient_name)

    # Quantity to grams
    g_per_unit = _unit_to_grams(info, unit)
    if g_per_unit is None: # fallback guess
        g_per_unit = estimate_weight(ingredient_name)
    grams_needed = qty * g_per_unit

    # Scale macros to recipe usage
    scaled = _scale_macros(macros, grams_needed)

    # accumulate
    row_macros["carbs"]    += scaled["carbs_g"]
    row_macros["protein"]  += scaled["protein_g"]
    row_macros["fat"]      += scaled["fat_g"]
    row_macros["fiber"]    += scaled["fiber_g"]
    row_macros["calories"] += scaled["calories"]

    if verbose:
        end_ = time()
        logger.info(f"Line Ingr Name: {ingredient_name}")
        logger.info(f"USDA Ingr Name: {usda_name}")
        logger.info(f"quantity: {qty} {unit}")
        logger.info(f"Estimated Weight (g): {grams_needed}")
        logger.info(f"Basis (g): {macros['basis_g']}")
        logger.info(f"Raw Carbs: {macros['carbs_g']}")
        logger.info(f"Scaled Carbs: {scaled['carbs_g']}")
        logger.info(f"Raw Protein: {macros['protein_g']}")
        logger.info(f"Scaled Protein: {scaled['protein_g']}")
        logger.info(f"Raw Fat: {macros['fat_g']}")
        logger.info(f"Scaled Fat: {scaled['fat_g']}")
        logger.info(f"Raw Fiber: {macros['fiber_g']}")
        logger.info(f"Scaled Fiber: {scaled['fiber_g']}")
        logger.info(f"Raw Calories: {macros['calories']}")
        logger.info(f"Scaled Calories: {scaled['calories']}")
        logger.info(f"Time Taken: {end_ - start_:.2f} seconds")
        logger.info("")

    return row_macros


def is_keto(ingredients, verbose = False):
    """
    Determine whether a recipe is ketogenic (≤ 20 % of calories from net carbs).

    Workflow (high-level)

    1. Parse each line with parse_ingredient
       – extracts quantity / unit / ingredient / preparation  
       – weights inside parenthetical overrides outer quantity if present

    2. Resolve the ingredient to a USDA FDC record  
       – re-ranked results from API using a hybrid metric: semantic similarity + search score
       – result cached on disk to avoid redundant API calls

    3. Fetch full nutrient data (cached).

    4. Apply heuristics 
       – _unit_to_grams converts declared unit ---> grams using foodPortions or _FALLBACK_GRAMS ENUMS  
       – if unit absent, estimate_weight guesses a default weight via sentence embedding similarity to category archetypes
       – if fdc_id absent, estimate_macronutrinets guesses a default macronutrinet distribution like estimate_weight function
       – certain zero-impact items (water, salt, artificial sweetners) override macros to 0

    5. Scale the per-basis macros to the grams actually used, aggregate them for the whole recipe, and compute:
         net_carbs = max(total_carbs − total_fiber, 0)  
         carb_pct  = (net_carbs × 4 kcal/g) / total_calories × 100

    6. Classify as keto if `carb_pct` ≤ 20 %.

    Notes
    * The 20 % threshold is deliberately lenient to offset uncertainty in
      weight-guessing and label inaccuracies.
    * Any exception inside the per-ingredient loop is caught and logged so a
      single bad line does not abort the entire evaluation; the outer `try`
      ensures a final Boolean is always returned.
    """

    try:
        if verbose:
            logger.info(str(ingredients))

        # running totals for the whole recipe
        
        row_macros = dict(carbs=0.0, protein=0.0, fat=0.0, fiber=0.0, calories=0.0)

        for line in ingredients:
            try:
                if verbose:
                    logger.info(line)
                row_macros = get_line_macros(line, row_macros, verbose)

            except Exception as e:
                # fail-soft on a single ingredient
                logger.warning(f"Error parsing line '{line}': {e}")
                continue

        # Final keto calculation
        net_carbs = max(row_macros["carbs"] - row_macros["fiber"], 0)
        carb_pct = (
            (net_carbs * 4) / row_macros["calories"] * 100
            if row_macros["calories"]
            else 0
        )
        is_this_keto = carb_pct <= 20

        if verbose:
            logger.info(f'total carbs: {row_macros["carbs"]}')
            logger.info(f'total protein: {row_macros["protein"]}')
            logger.info(f'total fat:  {row_macros["fat"]}')
            logger.info(f'total fiber: {row_macros["fiber"]}')
            logger.info(f'is keto: {is_this_keto}')
            logger.info("")

        return is_this_keto

    except Exception as e:
        # catastrophic failure – classify as non-keto for safety
        print(e)
        return False
