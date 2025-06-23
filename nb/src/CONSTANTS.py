import re
import numpy as np


_MASS_UNITS_SINGULAR = [
    'teaspoon', 'tablespoon', 'cup', 'ounce', 'pound', 'clove', 'can', 'slice', 
    'head', 'package', 'quart', 'pint', 'liter', 'gram', 'kilogram'
]

_NUMBER_RE = re.compile(r"^[\d\.\u00bc-\u00be\u2150-\u215e/ ]+$")


_FALLBACK_GRAMS = {
    # mass units
    "mg": 0.001,
    "g": 1,
    "gram": 1,
    "grams": 1,
    "kg": 1000,
    "kilogram": 1000,
    "kilograms": 1000,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
    "lb": 453.592,
    "pound": 453.592,
    "pounds": 453.592,

    # volume units (≈ water density)
    "ml": 1,
    "milliliter": 1,
    "millilitre": 1,
    "l": 1000,
    "liter": 1000,
    "litre": 1000,
    "tsp": 5,
    "teaspoon": 5,
    "teaspoons": 5,
    "tbsp": 15,
    "tablespoon": 15,
    "tablespoons": 15,
    "fl oz": 29.5735,
    "fluid ounce": 29.5735,
    "fluid ounces": 29.5735,
    "cup": 240,
    "cups": 240,
    "pint": 473,
    "pt": 473,
    "pints": 473,
    "quart": 946,
    "quarts": 946,
    "qt": 946,

    # recipe-specific heuristics
    "clove": 3,      # average garlic clove
    "cloves": 3,
    "slice": 28,     # bread slice
    "slices": 28,
    "can": 425,      # 15-oz US retail
    "cans": 425,
    "small can": 170,
    "large can": 794,
    "head": 300,     # lettuce/cabbage head
    "heads": 300,
    "package": 500,  # generic “pack” – adjust as needed
    "packages": 500,

    "clove": 3,
    "slice": 28,
    "rib": 40,
    "leaf": 5,
    "pinch": 0.36,
    "dash": 0.60,
    "sprig": 1,
    "bunch": 25,
    "ear":  90, 
    "stalk": 30,
    "stick": 113,
    "fillet": 170,
    "jar": 350,
}

# Fast membership lookup
_MASS_UNITS = {u for u, g in _FALLBACK_GRAMS.items() if g >= 1}   # ≈ all but mg



_CATS = {
    # DRIED / POWDER SPICES
    "a pinch of tiny spice like salt, pepper, yeast, cayenne, paprika, cumin, turmeric, nutmeg, cloves, oregano, cardamom, cinnamon": 2,
    "1 teaspoon of small spice like salt, pepper, yeast, cayenne, paprika, cumin, turmeric, nutmeg, cloves, oregano, cardamom, cinnamon": 5,
    "1 tablespoon of medium spice like salt, pepper, yeast, cayenne, paprika, cumin, turmeric, nutmeg, cloves, oregano, cardamom, cinnamon": 15,

    # FRESH HERBS
    "a handful of herb leaves like basil, cilantro, parsley, dill, mint, chives, rosemary, thyme, oregano, tarragon": 10,
    "half a bunch of herbs like basil, cilantro, parsley, dill, mint, chives, rosemary, thyme, oregano, tarragon": 25,

    # NUTS & SEEDS
    "a sprinkle of nuts like almonds, walnuts, pistachios, cashews, pecans, hazelnuts, macadamias, peanuts": 15,
    "a handful of nuts like almonds, walnuts, pistachios, cashews, pecans, hazelnuts, macadamias, peanuts": 30,

    # CHEESE
    "grated cheese like parmesan, mozzarella, cheddar, feta, pecorino, romano": 30,
    "one cheese slice like cheddar, swiss, provolone, american, gouda, havarti": 40,

    # MEAT / FISH
    "a small piece of meat like chicken breast, pork loin, cod fillet, tofu slab, tempeh slice": 100,
    "a medium piece of meat like salmon fillet, flank steak, turkey cutlet, swordfish steak": 150,
    "a large piece of meat like rib-eye steak, large chicken breast, whole pork chop": 225,
    "one small whole bird like quail, cornish hen": 1200,
    "one large whole bird like chicken, turkey, duck": 2000,
    "one small shellfish like scallop, shrimp": 20,
    "one large shellfish like lobster tail, king crab leg": 90,

    # FRESH MINI-VEG / CHILES
    "one small hot pepper like jalapeño, serrano, habanero, bird’s-eye": 15,

    # BULK VEGETABLES
    "a small sized vegetable like baby carrot, radish, baby potato, shallot": 50,
    "a medium sized vegetable like tomato, carror, cucumber, zucchini, bell pepper, beet, turnip": 100,
    "a large sized vegetable like eggplant, large carrot, large sweet potato, butternut squash": 200,
    "one head of leafy veg like lettuce, cabbage, kale, romaine, napa cabbage": 300,
    "one starchy veg like russet potato, sweet potato, cassava, plantain": 180,

    # FRUIT / MISC PRODUCE
    "a cup of berries like strawberries, blueberries, raspberries, blackberries, mixed berries": 150,
    "citrus fruit like orange, lemon, lime, grapefruit, tangerine": 130,
    "one banana (yellow or plantain)": 120,
    "one apple or pear like gala apple, granny smith, bartlett pear, bosc pear": 180,
    "one mushroom like button mushroom, cremini, shiitake, portobello cap": 18,

    # BAKERY & STARCHES 
    "one bread slice like white bread, whole-wheat, sourdough, rye": 30,
    "one bun or roll like hamburger bun, brioche roll, dinner roll, kaiser roll": 70,
    "one corn tortilla": 25,
    "one flour tortilla": 50,
    "a dry cup of pasta like penne, fusilli, macaroni, spaghetti pieces": 90,
    "a dry cup of rice like basmati, jasmine, long-grain, arborio": 190,

    # LIQUIDS & CANS
    "one liquid teaspoon like vanilla extract, lemon juice, soy sauce, vinegar": 5,
    "one liquid tablespoon like olive oil, sesame oil, maple syrup, honey": 15,
    "one cup of liquid like milk, broth, water, coconut milk": 240,
    "one quart of liquid like chicken stock, vegetable stock, tomato juice": 950,

    "one small can like tuna, tomato paste, diced green chiles": 170,
    "one standard can like black beans, kidney beans, diced tomatoes, chickpeas": 425,
    "one large can like crushed tomatoes, pumpkin purée, hominy": 800,

    # EGGS
    "one egg yolk": 18,
    "one egg white": 30,
    "one egg": 50,

    # PIZZA
    "large pizza crust": 400,
    "12 inch pizza crust": 300,
    "9 inch pizza crust": 250,
    "6 inch pizza crust": 150,
}


_CAT_NAMES   = list(_CATS)
_CAT_WEIGHTS = np.array([_CATS[c] for c in _CAT_NAMES], dtype=float)




_ZERO_NET = {
    # ────────────────────────────────────────────────────────────
    # 1) High-intensity / sugar-alcohol sweeteners & brand names
    # ────────────────────────────────────────────────────────────
    "splenda", "sucralose",
    "erythritol",           # incl. blends like 'Swerve'
    "allulose",
    "stevia", "stevia extract", "reb a", "rebiana",
    "monk fruit", "lo han guo", "luo han guo",
    "truvia", "purevia", "pyure", "lakanto", "swerve",
    "xylitol", "birch sugar",
    "maltitol", "sorbitol", "mannitol",   # optional: has lax net carbs, often counted as 0
    "isomalt", "erylite",
    "aspartame", "acesulfame potassium", "ace k",
    "saccharin", "cyclamate", "advantame", "neotame",

    # ────────────────────────────────────────────────────────────
    # 2) Water & zero-nutrient liquids / solids
    # ────────────────────────────────────────────────────────────
    "water", "tap water", "distilled water",
    "sparkling water", "seltzer", "club soda", "carbonated water",
    "ice", "ice cube", "ice cubes",
    "coffee", "black coffee", "espresso",
    "tea", "black tea", "green tea", "herbal tea",
    "diet soda", "diet cola",  # most have <1 kcal; safe to treat as 0

    # ────────────────────────────────────────────────────────────
    # 3) Pure seasonings & additives that are calorically nil
    # (USDA often shows <1 kcal per serving, but we can skip)
    # ────────────────────────────────────────────────────────────
    "salt", "sea salt", "kosher salt", "pink salt",
    "baking soda", "sodium bicarbonate",
    "cream of tartar", "tartaric acid",
    "baking powder",
    "yeast nutrient",   # tiny amounts
    "citric acid", "ascorbic acid", "potassium sorbate",
    "food coloring", "gel food coloring",
    "vanilla essence", "vanilla extract",  # carbs <0.5 g per tsp
    "almond extract", "peppermint extract",
    "vinegar", "white vinegar", "apple cider vinegar",
    "liquid smoke",


}
