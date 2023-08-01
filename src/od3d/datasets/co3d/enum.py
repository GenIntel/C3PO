from od3d.data import ExtEnum

class CAM_TFORM_OBJ_SOURCES(str, ExtEnum):
    CO3D = 'co3d'
    FIRST_FRAME = 'first_frame'
    FRONT_FRAME = 'front_frame'
    FRONT_FRAME_AND_PCL = 'front_frame_and_pcl'
    KPTS2D_ORIENT_AND_PCL = 'kpts2d_orient_and_pcl'

class CUBOID_SOURCES(str, ExtEnum):
    FRONT_FRAME_AND_PCL = 'front_frame_and_pcl'
    KPTS2D_ORIENT_AND_PCL = 'kpts2d_orient_and_pcl'

class CO3D_FRAME_TYPES(str, ExtEnum):
    DEV_KNOWN = 'dev_known'
    DEV_UNSEEN = 'dev_unseen'
    TEST_KNOWN = 'test_known'
    TEST_UNSEEN = 'test_unseen'
    TRAIN_KNOWN = 'train_known'
    TRAIN_UNSEEN = 'train_unseen'

class CO3D_FRAME_SPLITS(str, ExtEnum):
    MULTISEQUENCE_CAR_DEV_KNOWN = 'multisequence_car_dev_known'
    MULTISEQUENCE_CAR_DEV_UNSEEN = 'multisequence_car_dev_unseen'
    MULTISEQUENCE_CAR_TEST_KNOWN = 'multisequence_car_test_known'
    MULTISEQUENCE_CAR_TEST_UNSEEN = 'multisequence_car_test_unseen'
    MULTISEQUENCE_CAR_TRAIN_KNOWN = 'multisequence_car_train_known'
    MULTISEQUENCE_CAR_TRAIN_UNSEEN = 'multisequence_car_train_unseen'
    SINGLESEQUENCE_CAR_TEST_0_KNOWN = 'singlesequence_car_test_0_known'
    SINGLESEQUENCE_CAR_TEST_0_UNSEEN = 'singlesequence_car_test_0_unseen'

class CO3D_CATEGORIES(str, ExtEnum):
    APPLE = "apple"
    BACKPACK = "backpack"
    BALL = "ball"
    BANANA = "banana"
    BASEBALLBAT = "baseballbat"
    BASEBALLGLOVE = "baseballglove"
    BENCH = "bench"
    BICYCLE = "bicycle"
    BOOK = "book"
    BOTTLE = "bottle"
    BOWL = "bowl"
    BROCCOLI = "broccoli"
    CAKE = "cake"
    CAR = "car"
    CARROT = "carrot"
    CELLPHONE = "cellphone"
    CHAIR = "chair"
    COUCH = "couch"
    CUP = "cup"
    DONUT = "donut"
    FRISBEE = "frisbee"
    HAIRDRYER = "hairdryer"
    HANDBAG = "handbag"
    HOTDOG = "hotdog"
    HYDRANT = "hydrant"
    KEYBOARD = "keyboard"
    KITE = "kite"
    LAPTOP = "laptop"
    MICROWAVE = "microwave"
    MOTORCYCLE = "motorcycle"
    MOUSE = "mouse"
    ORANGE = "orange"
    PARKINGMETER = "parkingmeter"
    PIZZA = "pizza"
    PLANT = "plant"
    REMOTE = "remote"
    SANDWICH = "sandwich"
    SKATEBOARD = "skateboard"
    STOPSIGN = "stopsign"
    SUITCASE = "suitcase"
    TEDDYBEAR = "teddybear"
    TOASTER = "toaster"
    TOILET = "toilet"
    TOYBUS = "toybus"
    TOYPLANE = "toyplane"
    TOYTRAIN = "toytrain"
    TOYTRUCK = "toytruck"
    TV = "tv"
    UMBRELLA = "umbrella"
    VASE = "vase"
    WINEGLASS = "wineglass"

MAP_CO3D_PASCAL3D = {
    "apple" : None,
    "backpack": None,
    "ball": None,
    "banana": None,
    "baseballbat": None,
    "baseballglove": None,
    "bench": None,
    "bicycle": "bicycle",
    "book": None,
    "bottle": "bottle",
    "bowl": None,
    "broccoli": None,
    "cake": None,
    "car": "car",
    "carrot": None,
    "cellphone": None,
    "chair": "chair",
    "couch": "sofa",
    "cup": None,
    "donut": None,
    "frisbee": None,
    "hairdryer": None,
    "handbag": None,
    "hotdog": None,
    "hydrant": None,
    "keyboard": None,
    "kite": None,
    "laptop": None,
    "microwave": None,
    "motorcycle": "motorbike",
    "mouse": None,
    "orange": None,
    "parkingmeter": None,
    "pizza": None,
    "plant": None,
    "remote": None,
    "sandwich": None,
    "skateboard": None,
    "stopsign": None,
    "suitcase": None,
    "teddybear": None,
    "toaster": None,
    "toilet": None,
    "toybus": "bus",
    "toyplane": "aeroplane",
    "toytrain": "train",
    "toytruck": None,
    "tv": "tvmonitor",
    "umbrella": None,
    "vase": None,
    "wineglass": None,
}

MAP_CO3D_PASCAL3D_NOT_NONE = { key: value for key, value in MAP_CO3D_PASCAL3D.items() if value is not None}
# map total: 10
# pascal3d total: 12
# excluded 2: boat, dining table
# questionable 2: toyplane: aeroplane, toytrain: train

MAP_CO3D_OBJECTNET3D = {
    "apple" : None,
    "backpack": "backpack",
    "ball": None,
    "banana": None,
    "baseballbat": None,
    "baseballglove": None,
    "bench": "bench",
    "bicycle": "bicycle",
    "book": None,
    "bottle": "bottle",
    "bowl": None,
    "broccoli": None,
    "cake": None,
    "car": "car",
    "carrot": None,
    "cellphone": "cellphone",
    "chair": "chair",
    "couch": "sofa",
    "cup": "cup",
    "donut": None,
    "frisbee": None,
    "hairdryer": "hair_dryer",
    "handbag": None,
    "hotdog": None,
    "hydrant": None,
    "keyboard": "keyboard",
    "kite": None,
    "laptop": "laptop",
    "microwave": "microwave",
    "motorcycle": "motorbike",
    "mouse": "mouse",
    "orange": None,
    "parkingmeter": None,
    "pizza": None,
    "plant": None,
    "remote": None,
    "sandwich": None,
    "skateboard": None,
    "stopsign": None,
    "suitcase": "suitcase",
    "teddybear": None,
    "toaster": "toaster",
    "toilet": "toilet",
    "toybus": "bus",
    "toyplane": "aeroplane",
    "toytrain": "train",
    "toytruck": None,
    "tv": "tvmonitor",
    "umbrella": None,
    "vase": None,
    "wineglass": None,
}

MAP_CO3D_OBJECTNET3D_NOT_NONE = { key: value for key, value in MAP_CO3D_OBJECTNET3D.items() if value is not None}
# map total: 22
# objectnet3d total: 100
# excluded 78:
# questionable 2: toyplane:aeroplane, toytrain: train


MAP_CO3D_COCO = {
    "apple": "apple",
    "backpack": "backpack",
    "ball": "sports ball",
    "banana": "banana",
    "baseballbat": "baseball bat",
    "baseballglove": "baseball glove",
    "bench": "bench",
    "bicycle": "bicycle",
    "book": "book",
    "bottle": "bottle",
    "bowl": "bowl",
    "broccoli": "broccoli",
    "cake": "cake",
    "car": "car",
    "carrot": "carrot",
    "cellphone": "cell phone",
    "chair": "chair",
    "couch": "couch",
    "cup": "cup",
    "donut": "donut",
    "frisbee": "frisbee",
    "hairdryer": "hair drier",
    "handbag": "handbag",
    "hotdog": "hot dog",
    "hydrant": "fire hydrant",
    "keyboard": "keyboard",
    "kite": "kite",
    "laptop": "laptop",
    "microwave": "microwave",
    "motorcycle": "motorcycle",
    "mouse": "mouse",
    "orange": "orange",
    "parkingmeter": "parking meter",
    "pizza": "pizza",
    "plant": "potted plant",
    "remote": "remote",
    "sandwich": "sandwich",
    "skateboard": "skateboard",
    "stopsign": "stop sign",
    "suitcase": "suitcase",
    "teddybear": "teddy bear",
    "toaster": "toaster",
    "toilet": "toilet",
    "toybus": "bus",
    "toyplane": "airplane",
    "toytrain": "train",
    "toytruck": "truck",
    "tv": "tv",
    "umbrella": "umbrella",
    "vase": "vase",
    "wineglass": "wine glass",
}

MAP_CO3D_COCO_NOT_NONE = { key: value for key, value in MAP_CO3D_COCO.items() if value is not None}
# map total: 51
# coco total: 80
# excluded 29:
# questionable 3: toyplane:aeroplane, toytrain: train, toytruck: truck