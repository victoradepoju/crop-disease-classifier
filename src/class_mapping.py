# Maps PlantDoc's field-condition classes onto PlantVillage's disease taxonomy so both
# lab and field photos of the same disease end up in one class, directly targeting the
# lab-to-field generalization gap. Generic "<species>_leaf" PlantDoc classes mean healthy
# (confirmed via the PlantDoc paper's class list).
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple_Scab_Leaf": "Apple___Apple_scab",
    "Apple_leaf": "Apple___healthy",
    "Apple_rust_leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper_leaf": "Pepper,_bell___healthy",
    "Bell_pepper_leaf_spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry_leaf": "Blueberry___healthy",
    "Cherry_leaf": "Cherry_(including_sour)___healthy",
    "Corn_Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_rust_leaf": "Corn_(maize)___Common_rust_",
    "Peach_leaf": "Peach___healthy",
    "Potato_leaf_early_blight": "Potato___Early_blight",
    "Potato_leaf_late_blight": "Potato___Late_blight",
    "Raspberry_leaf": "Raspberry___healthy",
    "Soyabean_leaf": "Soybean___healthy",
    "Squash_Powdery_mildew_leaf": "Squash___Powdery_mildew",
    "Strawberry_leaf": "Strawberry___healthy",
    "Tomato_Early_blight_leaf": "Tomato___Early_blight",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_leaf": "Tomato___healthy",
    "Tomato_leaf_bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_leaf_late_blight": "Tomato___Late_blight",
    "Tomato_leaf_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mold_leaf": "Tomato___Leaf_Mold",
    "Tomato_two_spotted_spider_mites_leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape_leaf": "Grape___healthy",
    "grape_leaf_black_rot": "Grape___Black_rot",
}

CASSAVA_LABEL_TO_CLASS = {
    0: "Cassava___Bacterial_Blight",
    1: "Cassava___Brown_Streak_Disease",
    2: "Cassava___Green_Mottle",
    3: "Cassava___Mosaic_Disease",
    4: "Cassava___healthy",
}

RICE_FOLDER_TO_CLASS = {
    "Bacterialblight": "Rice___Bacterial_Blight",
    "Blast": "Rice___Blast",
    "Brownspot": "Rice___Brown_Spot",
    "Tungro": "Rice___Tungro",
}

BANANA_FOLDER_TO_CLASS = {
    "Banana Black Sigatoka Disease": "Banana___Black_Sigatoka",
    "Banana Bract Mosaic Virus Disease": "Banana___Bract_Mosaic_Virus",
    "Banana Healthy Leaf": "Banana___healthy",
    "Banana Insect Pest Disease": "Banana___Insect_Pest",
    "Banana Moko Disease": "Banana___Moko_Disease",
    "Banana Panama Disease": "Banana___Panama_Disease",
    "Banana Yellow Sigatoka Disease": "Banana___Yellow_Sigatoka",
}
