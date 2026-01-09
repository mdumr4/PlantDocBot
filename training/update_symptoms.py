
import json
import os

# Define the real symptom data map
symptom_map = {
    # APPLE
    "Apple___Apple_scab": "Velvety, olive-green to black spots on leaves and fruit. Leaves may turn yellow and drop early. Fruit may be deformed or cracked.",
    "Apple___Black_rot": "Firm, reddish-brown, circular spots on fruit that turn black and rotten. Leaves show frog-eye leaf spots (circular brown spots with purple margins).",
    "Apple___Cedar_apple_rust": "Bright yellow-orange spots on upper leaf surface, later developing tiny black dots. Underside of leaves may have tube-like structures.",
    "Apple___healthy": "Leaves are vibrant green. No dark spots. No rust. No brown lesions. No yellowing. Fruit is firm and unblemished.",

    # BLUEBERRY
    "Blueberry___healthy": "Leaves are dark green, glossy. No spots. No red discoloration. Stems are sturdy.",

    # CHERRY
    "Cherry_(including_sour)___healthy": "Foliage is green and clean. No white powdery fungus. No curling. No holes.",

    # CORN
    "Corn_(maize)___healthy": "Stalks are strong. Leaves are uniform green. No streaks. No gray spots. No rust pustules.",

    # GRAPE
    "Grape___healthy": "Vines are vigorous. Leaves are green and palm-shaped. No brown spots. No black rot. No shriveling.",

    # ORANGE
    "Orange___Haunglongbing_(Citrus_greening)": "Yellowing of leaf veins or mottling (blotchy patches). Fruit remains small, green, and bitter (misshapen).",

    # PEACH
    "Peach___Bacterial_spot": "Small, water-soaked spots on leaves that turn dark and may fall out ('shot hole' effect). Fruit has surface pitting or cracking.",
    "Peach___healthy": "Leaves are long, lance-shaped, and green. No bacterial spots. No holes. No curling.",

    # PEPPER
    "Pepper,_bell___Bacterial_spot": "Small, water-soaked spots on leaves and fruit. Spots turn brown/black and slightly raised.",
    "Pepper,_bell___healthy": "Leaves are smooth and green. No bacterial spots. No water-soaked lesions. Fruit is glossy.",

    # POTATO
    "Potato___Early_blight": "Dark brown spots with concentric rings ('bullseye' pattern) on older leaves. Yellowing around spots.",
    "Potato___Late_blight": "Large, irregular, dark green to black water-soaked lesions on leaves. White fuzzy mold on undersides in wet weather.",
    "Potato___healthy": "Foliage is bushy and green. No dark brown spots. No concentric rings. No mold.",

    # RASPBERRY
    "Raspberry___healthy": "Canes are brown/green depending on variety. Leaves are clean. No spots. No yellowing.",

    # SOYBEAN
    "Soybean___healthy": "Leaves are trifoliate, green. No pustules. No yellowing. No leaf spots.",

    # SQUASH
    "Squash___Powdery_mildew": "White, talcum-like powder patches on leaves, stems, and fruit. Leaves may yellow and wither.",

    # STRAWBERRY
    "Strawberry___Leaf_scorch": "Irregular, dark purple or reddish spots on leaves. The spots may coalesce, causing the leaf to look scorched or burnt.",
    "Strawberry___healthy": "Leaves are bright green with serrated edges. No purple spots. No red spots. No scorching.",

    # TOMATO
    "Tomato___Bacterial_spot": "Small, dark, greasy-looking spots on leaves and stems. Fruit often has scab-like, raised brown spots.",
    "Tomato___Early_blight": "Brown spots with concentric rings (target-like) on lower leaves first. Leaves turn yellow and drop.",
    "Tomato___Late_blight": "Large, irregular, pale green to brown lesions. White cottony growth on leaf undersides during high humidity.",
    "Tomato___Leaf_Mold": "Pale yellow spots on upper leaf surface. Olive-green to brown velvety mold on the underside of the spots.",
    "Tomato___Septoria_leaf_spot": "Numerous small, circular spots with dark borders and gray centers. Lower leaves affected first.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Leaves look stippled (tiny yellow dots). Fine webbing may be visible. Mites are tiny red/yellow specks.",
    "Tomato___Target_Spot": "Brown to black lesions with concentric rings, similar to early blight but often with a yellow halo.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Leaves curl upward and turn yellow at the edges. Plants are stunted. Flowers may drop.",
    "Tomato___Tomato_mosaic_virus": "Mottled light and dark green pattern on leaves (mosaic). Leaves may be filiform (stringy) or fern-like.",
    "Tomato___healthy": "Vigorous growth. Compound green leaves. No dark spots. No yellowing. No curling. No mosaic pattern. No mold."
}

def update_database():
    if not os.path.exists("disease_info.json"):
        print("Error: disease_info.json not found.")
        return

    with open("disease_info.json", "r") as f:
        data = json.load(f)

    print(f"Updating {len(data)} records with real symptom descriptions...")

    for key, val in data.items():
        # Update/Overwrite the 'description' field with searchable symptoms if available
        if key in symptom_map:
            # We keep the old 'name' and 'treatment' but update 'description' to be symptom-focused
            # Or better, we add a specific 'symptoms' field for the search engine to use.
            val["symptoms"] = symptom_map[key]

            # Also update the display description to be more useful
            val["description"] = symptom_map[key]

    with open("disease_info.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Success! disease_info.json updated.")

if __name__ == "__main__":
    update_database()
