import re
import numpy as np


_NUMBER_RE = re.compile(r"^[\d\.\u00bc-\u00be\u2150-\u215e/ ]+$")


# unit to gram conversion
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

    # volume units
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
    "clove": 3,
    "cloves": 3,
    "slice": 28,
    "slices": 28,
    "can": 425,
    "cans": 425,
    "small can": 170,
    "large can": 794,
    "head": 300,
    "heads": 300,
    "package": 500,
    "packages": 500,
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

# Fast unit lookup
_MASS_UNITS = set(sorted(_FALLBACK_GRAMS.keys(), key=len, reverse=True))


# Product-weight categories
_CAT_WEIGHTS = {
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

    # DRESSINGS
    "one tablespoon of dressing, frosting, glaze, dip, syrup": 18,
    "a quarter cup of sauce, salsa, or topping": 60,
    "one cup of sauce, salsa, gravy, or custard": 240,

    # GRAVY
    "one ladle of gravy": 60,

    # BATTER
    "one cup of batter (pancake, cake, fritter)": 250,
    "one pancake-worth of batter": 60,

    # DUMPLINGS
    "one small dumpling like gyoza, wonton, ravioli": 20,
    "one large dumpling like bao, pierogi": 50,

    # CAKE / PASTRIES
    "one slice of cake like chocolate, vanilla, carrot": 100,
    "one cupcake or muffin": 70,

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

# fast product-weight category lookup
__CAT_WEIGHTS_NAMES = list(_CAT_WEIGHTS)
__CAT_WEIGHTS = [_CAT_WEIGHTS[c] for c in __CAT_WEIGHTS_NAMES]


# Product-macro categories
_CAT_MACROS = {
    # SEASONINGS & CONDIMENTS
    "Dried spice powder": dict(carbs_g=60, protein_g=10, fat_g=6,  fiber_g=25, calories=300),
    "Fresh herb leafy": dict(carbs_g=7,  protein_g=3,  fat_g=0.5,fiber_g=3,  calories=40),
    "Gravy (milk based)": dict(carbs_g=6,  protein_g=3,  fat_g=4,  fiber_g=0,  calories=80),
    "Pure oil or butter": dict(carbs_g=0,  protein_g=0,  fat_g=100,fiber_g=0,  calories=884),
    "Zero-cal sweetener": dict(carbs_g=0,  protein_g=0,  fat_g=0,  fiber_g=0,  calories=0),
    "Plain sugar": dict(carbs_g=100,protein_g=0,  fat_g=0,  fiber_g=0,  calories=400),

    # NUTS, SEEDS & LEGUMES
    "Nuts or seeds": dict(carbs_g=20, protein_g=20, fat_g=50, fiber_g=9,  calories=600),
    "Canned legumes": dict(carbs_g=14, protein_g=6,  fat_g=1,  fiber_g=5,  calories=90),

    # DAIRY
    "Semi-hard cheese": dict(carbs_g=2,  protein_g=25, fat_g=33, fiber_g=0,  calories=400),
    "Milk or stock (unsweet)": dict(carbs_g=5,  protein_g=3,  fat_g=3,  fiber_g=0,  calories=60),

    # EGGS
    "Whole egg": dict(carbs_g=1,  protein_g=13, fat_g=11, fiber_g=0,  calories=155),

    # MEAT & SEAFOOD
    "Lean meat or poultry": dict(carbs_g=0,  protein_g=26, fat_g=7,  fiber_g=0,  calories=165),
    "Fatty meat cut (rib-eye)": dict(carbs_g=0,  protein_g=23, fat_g=35, fiber_g=0,  calories=350),
    "Lean fish (cod/haddock)": dict(carbs_g=0,  protein_g=24, fat_g=1,  fiber_g=0,  calories=105),
    "Fatty fish (salmon/mackerel)": dict(carbs_g=0,  protein_g=20, fat_g=13, fiber_g=0,  calories=208),
    "Average shellfish": dict(carbs_g=1,  protein_g=24, fat_g=1,  fiber_g=0,  calories=106),
    "Processed meat (sausage)": dict(carbs_g=2,  protein_g=14, fat_g=28, fiber_g=0,  calories=310),

    # GRAIN, BREAD & STARCHES
    "Bread or roll (baked)": dict(carbs_g=49, protein_g=9,  fat_g=4,  fiber_g=3.5,calories=265),
    "Corn tortilla": dict(carbs_g=44, protein_g=6,  fat_g=3,  fiber_g=4,  calories=230),
    "Flour tortilla": dict(carbs_g=49, protein_g=8,  fat_g=9,  fiber_g=4,  calories=300),
    "Dry pasta": dict(carbs_g=75, protein_g=13, fat_g=1.5,fiber_g=3,  calories=370),
    "Dry rice": dict(carbs_g=80, protein_g=7,  fat_g=0.7,fiber_g=1,  calories=365),
    "Plain pizza crust": dict(carbs_g=54, protein_g=9,  fat_g=4,  fiber_g=3,  calories=300),
    "Pancake or cake batter": dict(carbs_g=30, protein_g=6,  fat_g=4,  fiber_g=1,  calories=190),
    "Cake or pastry slice": dict(carbs_g=55, protein_g=5,  fat_g=15, fiber_g=1,  calories=380),
    "Filled dumpling": dict(carbs_g=35, protein_g=8,  fat_g=5,  fiber_g=2,  calories=220),

    # PRODUCE
    "Leafy greens": dict(carbs_g=4,  protein_g=3,  fat_g=0,  fiber_g=2,  calories=25),
    "Cruciferous veg": dict(carbs_g=6,  protein_g=3,  fat_g=0,  fiber_g=3,  calories=35),
    "Peppers and chiles": dict(carbs_g=6,  protein_g=1,  fat_g=0,  fiber_g=2,  calories=30),
    "Alliums (onion/garlic)": dict(carbs_g=9,  protein_g=2,  fat_g=0,  fiber_g=1,  calories=40),
    "Non-starchy veg (raw)": dict(carbs_g=7,  protein_g=2,  fat_g=0,  fiber_g=3,  calories=35),
    "Starchy root veg": dict(carbs_g=17, protein_g=2,  fat_g=0,  fiber_g=2,  calories=86),
    "Common fruit": dict(carbs_g=14, protein_g=1,  fat_g=0,  fiber_g=2,  calories=60),
    "Raw berries": dict(carbs_g=14, protein_g=1,  fat_g=0,  fiber_g=5,  calories=60),
    "Banana": dict(carbs_g=23, protein_g=1,  fat_g=0,  fiber_g=3,  calories=96),
    "Raw mushroom": dict(carbs_g=3,  protein_g=3,  fat_g=0,  fiber_g=1,  calories=25),

    # SWEET TOPPINGS / FILLINGS
    "Sweet frosting or glaze": dict(carbs_g=80, protein_g=1,  fat_g=15, fiber_g=0,  calories=460),
    "Syrup or honey": dict(carbs_g=78, protein_g=0,  fat_g=0,  fiber_g=0,  calories=310),
    "Custard or pudding": dict(carbs_g=23, protein_g=4,  fat_g=4,  fiber_g=0,  calories=150),
    
    # SAVOURY SAUCES & DIPS
    "Tomato-based sauce": dict(carbs_g=12, protein_g=2,  fat_g=1,  fiber_g=2,  calories=65),
    "Cream-based sauce": dict(carbs_g=5,  protein_g=3,  fat_g=20, fiber_g=0,  calories=200),
    "Salsa or relish": dict(carbs_g=8,  protein_g=2,  fat_g=1,  fiber_g=2,  calories=50),
    "Savory dip (mayo/ranch)": dict(carbs_g=6,  protein_g=2,  fat_g=35, fiber_g=0,  calories=330),

    # PASTRIES
    "Flaky pastry (croissant/puff)": dict(carbs_g=45, protein_g=8,  fat_g=30, fiber_g=2,  calories=520),
    "Sweet pastry (danish/tart)": dict(carbs_g=55, protein_g=6,  fat_g=20, fiber_g=2,  calories=430),
    "Savory pastry (empanada)": dict(carbs_g=35, protein_g=8,  fat_g=18, fiber_g=3,  calories=380),

    # DUMPLINGS
    "Small dumpling (gyoza)": dict(carbs_g=35, protein_g=8,  fat_g=5,  fiber_g=2,  calories=220),
    "Large dumpling (bao/pierogi)": dict(carbs_g=32, protein_g=7,  fat_g=6,  fiber_g=2,  calories=230),

    # LIQUIDS & CANS
    "Clear broth or stock": dict(carbs_g=1,  protein_g=3,  fat_g=1,  fiber_g=0,  calories=20),
    "Milk unsweetened": dict(carbs_g=5,  protein_g=3,  fat_g=3,  fiber_g=0,  calories=60),
    "Plant milk unsweetened": dict(carbs_g=2,  protein_g=1,  fat_g=2,  fiber_g=0,  calories=30),
    "Fruit juice (100 %)": dict(carbs_g=10, protein_g=0,  fat_g=0,  fiber_g=0,  calories=42),
    "Small canned fish (tuna)": dict(carbs_g=0,  protein_g=26, fat_g=8,  fiber_g=0,  calories=180),
    "Standard canned beans": dict(carbs_g=14, protein_g=6,  fat_g=1,  fiber_g=5,  calories=90),
    "Large canned tomato": dict(carbs_g=4,  protein_g=1,  fat_g=0,  fiber_g=1,  calories=24),
}


# fast product-macro category lookup
__CAT_MACROS_NAMES = list(_CAT_MACROS)
__CAT_MACROS = [_CAT_MACROS[k] for k in __CAT_MACROS_NAMES]


_ZERO_NET = {
    # High-intensity / sugar-alcohol sweeteners & brand names
    "splenda", 
    "sucralose",
    "erythritol",
    "allulose",
    "stevia", 
    "stevia extract", 
    "reb a", 
    "rebiana",
    "monk fruit", 
    "lo han guo", 
    "luo han guo",
    "truvia",
    "purevia", 
    "pyure", 
    "lakanto", 
    "swerve",
    "xylitol", 
    "birch sugar",
    "maltitol", 
    "sorbitol", 
    "mannitol",
    "isomalt", 
    "erylite",
    "aspartame", 
    "acesulfame potassium", 
    "ace k",
    "saccharin", 
    "cyclamate", 
    "advantame", 
    "neotame",

    # Water & zero-nutrient liquids / solids
    "water", 
    "tap water", 
    "distilled water",
    "sparkling water", 
    "seltzer", 
    "ice", 
    "ice cube", 
    "ice cubes",
    "diet soda", 
    "diet cola",

    # Pure seasonings & additives that are calorically nil
    "salt", 
    "sea salt", 
    "kosher salt", 
    "pink salt",
    "baking soda", 
    "sodium bicarbonate",
    "yeast nutrient",
    "citric acid", 
    "ascorbic acid", 
    "potassium sorbate",
    "food coloring", 
    "gel food coloring",
    "liquid smoke",
}
