import fractions
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import json
import pathlib
import hashlib


from CONSTANTS import (_MASS_UNITS, _MASS_UNITS_SINGULAR, _NUMBER_RE, _FALLBACK_GRAMS, 
                        _CATS, _CAT_NAMES, _CAT_WEIGHTS, _ZERO_NET)


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
    for u in sorted(_MASS_UNITS_SINGULAR, key=len, reverse=True):
        if rest.lower().startswith(u + 's '):
            outer_unit, rest = u, rest[len(u) + 1:].lstrip()
            break
        elif rest.lower().startswith(u + ' '):
            outer_unit, rest = u, rest[len(u):].lstrip()
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
    return ranked[0][:2]


_LOOKUP_DIR = pathlib.Path("./usda_lookup_cache")
_LOOKUP_DIR.mkdir(exist_ok=True)


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


_CACHE_DIR = pathlib.Path("./usda_json_cache")
_CACHE_DIR.mkdir(exist_ok=True)

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
        print(f"Could not parse quantity {q} because of {e} – defaulting to 1")
        return 1.0


# precomputing embeddings for CATEGORIES of items.
_EMBS = np.stack([_embedding(CAT) for CAT in _CAT_NAMES])


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
    best = int(np.argmax(_EMBS @ _embedding(item)))
    return float(_CAT_WEIGHTS[best])


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
