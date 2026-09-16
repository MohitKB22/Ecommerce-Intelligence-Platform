"""Static taxonomy used to generate the synthetic catalogue.

Everything here is invented for this project. It is NOT derived from, and does
not represent, any real retailer's catalogue or data.
"""
from __future__ import annotations

# category slug -> (display name, parent slug or None, price range, keyword pool)
CATEGORIES: dict[str, dict] = {
    "electronics": {"name": "Electronics", "parent": None, "price": (25, 900), "keywords": []},
    "audio": {"name": "Audio", "parent": "electronics", "price": (19, 480),
              "keywords": ["headphones", "earbuds", "speaker", "soundbar", "microphone", "turntable"]},
    "wearables": {"name": "Wearables", "parent": "electronics", "price": (39, 620),
                  "keywords": ["smartwatch", "fitness tracker", "smart ring", "sleep band"]},
    "cameras": {"name": "Cameras", "parent": "electronics", "price": (89, 1400),
                "keywords": ["action camera", "mirrorless camera", "webcam", "gimbal", "drone camera"]},
    "computers": {"name": "Computers", "parent": None, "price": (35, 2200), "keywords": []},
    "laptops": {"name": "Laptops", "parent": "computers", "price": (420, 2400),
                "keywords": ["ultrabook", "gaming laptop", "notebook", "convertible laptop", "workstation"]},
    "peripherals": {"name": "Peripherals", "parent": "computers", "price": (15, 320),
                    "keywords": ["mechanical keyboard", "wireless mouse", "monitor stand", "docking station", "usb hub"]},
    "storage": {"name": "Storage", "parent": "computers", "price": (22, 480),
                "keywords": ["ssd", "external drive", "memory card", "nas enclosure", "flash drive"]},
    "home-kitchen": {"name": "Home & Kitchen", "parent": None, "price": (12, 700), "keywords": []},
    "kitchen-appliances": {"name": "Kitchen Appliances", "parent": "home-kitchen", "price": (29, 640),
                           "keywords": ["air fryer", "espresso machine", "blender", "rice cooker", "stand mixer"]},
    "home-essentials": {"name": "Home Essentials", "parent": "home-kitchen", "price": (11, 260),
                        "keywords": ["vacuum", "air purifier", "humidifier", "desk lamp", "storage bin"]},
    "sports": {"name": "Sports & Outdoors", "parent": None, "price": (9, 900), "keywords": []},
    "fitness": {"name": "Fitness", "parent": "sports", "price": (14, 780),
                "keywords": ["dumbbell set", "yoga mat", "resistance bands", "treadmill", "kettlebell"]},
    "outdoor": {"name": "Outdoor", "parent": "sports", "price": (18, 700),
                "keywords": ["camping tent", "hiking backpack", "sleeping bag", "trekking poles", "cooler"]},
    "beauty": {"name": "Beauty & Personal Care", "parent": None, "price": (7, 300),
               "keywords": ["face serum", "hair dryer", "electric shaver", "sunscreen", "moisturiser"]},
    "books": {"name": "Books", "parent": None, "price": (6, 60),
              "keywords": ["paperback novel", "cookbook", "technical guide", "biography", "graphic novel"]},
    "toys": {"name": "Toys & Games", "parent": None, "price": (8, 240),
             "keywords": ["building blocks", "board game", "puzzle set", "remote control car", "plush toy"]},
    "office": {"name": "Office", "parent": None, "price": (10, 900),
               "keywords": ["ergonomic chair", "standing desk", "desk organiser", "label printer", "whiteboard"]},
    "pets": {"name": "Pet Supplies", "parent": None, "price": (8, 350),
             "keywords": ["pet bed", "automatic feeder", "dog harness", "cat tree", "grooming kit"]},
    "fashion": {"name": "Fashion", "parent": None, "price": (12, 420),
                "keywords": ["running shoes", "rain jacket", "leather belt", "wool scarf", "backpack"]},
}

LEAF_CATEGORIES = [slug for slug, meta in CATEGORIES.items() if meta["keywords"]]

# Invented brand names - any resemblance to real brands is unintentional.
BRANDS: list[tuple[str, float]] = [
    ("Nordvale", 0.88), ("Auralis", 0.83), ("Kestrelab", 0.79), ("Ferrostar", 0.74), ("Lumenwerk", 0.86),
    ("Brightpath", 0.68), ("Verdant Co", 0.72), ("Ironbloom", 0.61), ("Solvea", 0.77), ("Marloq", 0.55),
    ("Piquant", 0.64), ("Tundrix", 0.7), ("Helioform", 0.81), ("Cobaltine", 0.58), ("Wrenmark", 0.66),
    ("Zephyrra", 0.75), ("Oakspire", 0.69), ("Norsel", 0.62), ("Quillon", 0.73), ("Pallavo", 0.6),
    ("Basalt Works", 0.71), ("Emberline", 0.65), ("Corvane", 0.78), ("Driftwell", 0.57), ("Halcyra", 0.8),
]

MATERIALS = ["aluminium", "recycled ABS", "brushed steel", "carbon composite", "bamboo", "silicone", "tempered glass"]
COLOURS = ["graphite", "arctic white", "midnight blue", "sage", "sandstone", "crimson", "slate"]
QUALIFIERS = ["Pro", "Lite", "Max", "Studio", "Everyday", "Compact", "Elite", "Core", "Plus", "Nova"]
USE_CASES = ["for travel", "for the gym", "for home office", "for everyday use", "for commuting",
             "for professionals", "for beginners", "for outdoor use"]

# aspect -> (positive phrases, negative phrases). Drives review generation AND
# is the ground truth the aspect extractor is evaluated against.
ASPECT_LEXICON: dict[str, dict[str, list[str]]] = {
    "battery_life": {
        "keywords": ["battery", "charge", "charging", "runtime", "power"],
        "positive": ["battery life is excellent and easily lasts a full week",
                     "it holds a charge far longer than I expected",
                     "the runtime between charges is outstanding"],
        "negative": ["the battery drains much faster than advertised",
                     "it needs charging almost every single day",
                     "runtime is disappointing and charging is slow"],
    },
    "sound_quality": {
        "keywords": ["sound", "audio", "bass", "treble", "noise"],
        "positive": ["the sound is crisp with genuinely deep bass",
                     "audio clarity is superb across the whole range",
                     "noise isolation is better than anything at this price"],
        "negative": ["the sound is muddy and the bass is overpowering",
                     "audio quality is thin and tinny at higher volume",
                     "there is a persistent hiss in quiet passages"],
    },
    "comfort": {
        "keywords": ["comfort", "comfortable", "fit", "cushion", "ergonomic", "wear"],
        "positive": ["it stays comfortable even after several hours",
                     "the fit is snug without any pressure points",
                     "the cushions are soft and breathable"],
        "negative": ["the ear cushions are uncomfortable after twenty minutes",
                     "the fit is far too tight and starts to ache",
                     "it becomes hot and sweaty during long sessions"],
    },
    "build_quality": {
        "keywords": ["build", "quality", "material", "construction", "solid", "flimsy"],
        "positive": ["the build quality feels premium and solid",
                     "materials are clearly a step above the competition",
                     "construction is tight with no creaking anywhere"],
        "negative": ["the plastic feels flimsy and creaks constantly",
                     "build quality is poor for the price point",
                     "a hinge cracked within the first month"],
    },
    "value": {
        "keywords": ["price", "value", "money", "cheap", "expensive", "worth"],
        "positive": ["outstanding value for what you pay",
                     "hard to beat at this price",
                     "worth every penny and then some"],
        "negative": ["seriously overpriced for what you actually get",
                     "not worth the money compared to alternatives",
                     "the price is hard to justify"],
    },
    "shipping": {
        "keywords": ["shipping", "delivery", "arrived", "packaging", "package"],
        "positive": ["arrived two days early and beautifully packaged",
                     "delivery was quick and the packaging was secure"],
        "negative": ["the package arrived crushed and late",
                     "shipping took over three weeks with no updates"],
    },
    "durability": {
        "keywords": ["durable", "durability", "broke", "lasted", "wear", "scratch"],
        "positive": ["still looks new after months of heavy use",
                     "it has survived daily abuse without a scratch"],
        "negative": ["it stopped working after just six weeks",
                     "scratches appeared almost immediately"],
    },
    "ease_of_use": {
        "keywords": ["setup", "easy", "intuitive", "instructions", "app", "confusing"],
        "positive": ["setup took under five minutes and was completely intuitive",
                     "the app is clean and genuinely easy to navigate"],
        "negative": ["setup was confusing and the instructions were useless",
                     "the companion app is buggy and crashes often"],
    },
    "design": {
        "keywords": ["design", "look", "style", "colour", "aesthetic", "sleek"],
        "positive": ["the design is sleek and looks great on a desk",
                     "genuinely attractive - it gets compliments"],
        "negative": ["it looks far cheaper in person than in the photos",
                     "the styling is dated and the colour is off"],
    },
    "performance": {
        "keywords": ["performance", "fast", "slow", "lag", "speed", "responsive"],
        "positive": ["performance is fast and completely lag free",
                     "it handles heavy workloads without breaking a sweat"],
        "negative": ["performance is sluggish and it stutters under load",
                     "noticeable lag makes it frustrating to use"],
    },
}

ASPECT_NAMES = list(ASPECT_LEXICON.keys())

# Which aspects are plausible per category - keeps generated reviews coherent.
CATEGORY_ASPECTS: dict[str, list[str]] = {
    "audio": ["sound_quality", "battery_life", "comfort", "build_quality", "value", "design"],
    "wearables": ["battery_life", "comfort", "ease_of_use", "design", "durability", "value"],
    "cameras": ["performance", "build_quality", "ease_of_use", "battery_life", "value", "design"],
    "laptops": ["performance", "battery_life", "build_quality", "design", "value", "durability"],
    "peripherals": ["build_quality", "ease_of_use", "design", "value", "durability", "performance"],
    "storage": ["performance", "durability", "value", "build_quality", "ease_of_use"],
    "kitchen-appliances": ["ease_of_use", "build_quality", "performance", "durability", "value", "design"],
    "home-essentials": ["ease_of_use", "build_quality", "value", "design", "durability"],
    "fitness": ["build_quality", "durability", "comfort", "value", "design"],
    "outdoor": ["durability", "build_quality", "comfort", "value", "shipping"],
    "beauty": ["value", "ease_of_use", "design", "durability", "shipping"],
    "books": ["value", "shipping", "design", "build_quality"],
    "toys": ["durability", "value", "design", "ease_of_use", "shipping"],
    "office": ["comfort", "build_quality", "design", "value", "ease_of_use", "durability"],
    "pets": ["durability", "build_quality", "value", "ease_of_use", "design"],
    "fashion": ["comfort", "durability", "design", "value", "shipping"],
}

REVIEW_OPENERS_POS = ["Really pleased with this purchase.", "Exactly what I was hoping for.",
                      "Bought this a month ago and I have no regrets.", "This exceeded my expectations."]
REVIEW_OPENERS_NEU = ["Mixed feelings about this one.", "It is fine, nothing special.",
                      "Does the job but there are trade-offs.", "Reasonable, with some caveats."]
REVIEW_OPENERS_NEG = ["Disappointed after a few weeks of use.", "I would not buy this again.",
                      "Not what the listing led me to expect.", "Regret this purchase."]
REVIEW_CLOSERS_POS = ["Would happily recommend it.", "Will buy from this brand again.",
                      "Solid recommendation from me."]
REVIEW_CLOSERS_NEU = ["Might suit you depending on priorities.", "Consider the alternatives first.",
                      "Three stars feels about right."]
REVIEW_CLOSERS_NEG = ["Looking into returning it.", "Save your money and look elsewhere.",
                      "Cannot recommend it as it stands."]

SEARCH_INTENT_TEMPLATES = [
    "{kw}", "best {kw}", "{kw} {use_case}", "cheap {kw}", "{brand} {kw}",
    "{kw} under {price}", "{colour} {kw}", "{kw} with long battery life", "durable {kw}", "{kw} for beginners",
]


# Complementary categories drive realistic multi-item baskets, which in turn
# create the co-purchase signal the "frequently bought together" feature needs.
COMPLEMENTARY: dict[str, list[str]] = {
    "audio": ["wearables", "peripherals", "storage"],
    "wearables": ["audio", "fitness", "fashion"],
    "cameras": ["storage", "outdoor", "peripherals"],
    "laptops": ["peripherals", "storage", "office"],
    "peripherals": ["laptops", "storage", "office"],
    "storage": ["laptops", "cameras", "peripherals"],
    "kitchen-appliances": ["home-essentials", "books"],
    "home-essentials": ["kitchen-appliances", "pets", "office"],
    "fitness": ["wearables", "outdoor", "fashion"],
    "outdoor": ["fitness", "cameras", "fashion"],
    "beauty": ["fashion", "home-essentials"],
    "books": ["office", "toys", "kitchen-appliances"],
    "toys": ["books", "pets"],
    "office": ["peripherals", "laptops", "home-essentials"],
    "pets": ["home-essentials", "toys"],
    "fashion": ["beauty", "wearables", "outdoor"],
}
