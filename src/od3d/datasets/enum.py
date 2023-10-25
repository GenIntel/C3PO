from od3d.data import ExtEnum


class OD3D_CATEGORIES(str, ExtEnum):
    APPLE = "apple" # CO3D + PASCAL3D
    BACKPACK = "backpack"
    BALL = "ball"
    BANANA = "banana"
    BASEBALLBAT = "baseballbat"
    BASEBALLGLOVE = "baseballglove"
    BENCH = "bench"
    BICYCLE = "bicycle"
    BOAT = "boat"
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
    DINING_TABLE = "dining_table"
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
    BUS = "bus" #
    AIRPLANE = "airplane"
    TRAIN = "train"
    TRUCK = "truck"
    TV = "tv"
    UMBRELLA = "umbrella"
    VASE = "vase"
    WINE_GLASS = "wine_glass"
    ASHTRAY = "ashtray" # OBJECTNET3D
    BASKET = "basket"
    BED = "bed"
    BLACKBOARD = "blackboard"
    BOOKSHELF = "bookshelf"
    BUCKET = "bucket"
    CABINET = "cabinet"
    CALCULATOR = "calculator"
    CAMERA = "camera"
    CAN = "can"
    CAP = "cap"
    CLOCK = "clock"
    COFFEE_MAKER = "coffee_maker"
    COMB = "comb"
    COMPUTER = "computer"
    DESK_LAMP = "desk_lamp"
    DISHWASHER = "dishwasher"
    DOOR = "door"
    ERASER = "eraser"
    EYEGLASSES = "eyeglasses"
    FAN = "fan"
    FAUCET = "faucet"
    FILING_CABINET = "filing_cabinet"
    FIRE_EXTINGUISHER = "fire_extinguisher"
    FISH_TANK = "fish_tank"
    FLASHLIGHT = "flashlight"
    FORK = "fork"
    GUITAR = "guitar"
    HAMMER = "hammer"
    HEADPHONE = "headphone"
    HELMET = "helmet"
    IRON = "iron"
    JAR = "jar"
    KETTLE = "kettle"
    KEY = "key"
    KNIFE = "knife"
    LIGHTER = "lighter"
    MAILBOX = "mailbox"
    MICROPHONE = "microphone"
    PAINTBRUSH = "paintbrush"
    PAN = "pan"
    PEN = "pen"
    PENCIL = "pencil"
    PIANO = "piano"
    PILLOW = "pillow"
    PLATE = "plate"
    POT = "pot"
    PRINTER = "printer"
    RACKET = "racket"
    REFRIGERATOR = "refrigerator"
    RIFLE = "rifle"
    ROAD_POLE = "road_pole"
    SATELLITE_DISH = "satellite_dish"
    SCISSORS = "scissors"
    SCREWDRIVER = "screwdriver"
    SHOE = "shoe"
    SHOVEL = "shovel"
    SIGN = "sign"
    SKATE = "skate"
    SLIPPER = "slipper"
    SPEAKER = "speaker"
    SPOON = "spoon"
    STAPLER = "stapler"
    STOVE = "stove"
    TEAPOT = "teapot"
    TELEPHONE = "telephone"
    TOOTHBRUSH = "toothbrush"
    TRASH_BIN = "trash_bin"
    TROPHY = "trophy"
    TUB = "tub"
    VENDING_MACHINE = "vending_machine"
    WASHING_MACHINE = "washing_machine"
    WATCH = "watch"
    WHEELCHAIR = "wheelchair"

OD3D_CATEGORIES_SIZES_IN_M = {
    OD3D_CATEGORIES.BICYCLE: 1.7,
    OD3D_CATEGORIES.TRUCK: 0.15, # toytruck 0.15
    OD3D_CATEGORIES.TRAIN: 0.2, # toytrain 0.2
    OD3D_CATEGORIES.TEDDYBEAR: 0.3,
    OD3D_CATEGORIES.CAR: 4.5,
    OD3D_CATEGORIES.BUS: 0.25, # toybus 0.25
    OD3D_CATEGORIES.MOTORCYCLE: 2.0,
    OD3D_CATEGORIES.KEYBOARD: 0.45,
    OD3D_CATEGORIES.HANDBAG: 0.3,
    OD3D_CATEGORIES.REMOTE: 0.15,
    OD3D_CATEGORIES.AIRPLANE: 0.3, # toyplane 0.3
    OD3D_CATEGORIES.TOILET: 0.75,
    OD3D_CATEGORIES.HAIRDRYER: 0.25,
    OD3D_CATEGORIES.MOUSE: 0.1,
    OD3D_CATEGORIES.TOASTER: 0.25,
    OD3D_CATEGORIES.HYDRANT: 1.0,
    OD3D_CATEGORIES.CHAIR: 0.5,
    OD3D_CATEGORIES.LAPTOP: 0.35,
    OD3D_CATEGORIES.BOOK: 0.2,
    OD3D_CATEGORIES.BACKPACK: 0.45,
    OD3D_CATEGORIES.APPLE: 0.08,
    OD3D_CATEGORIES.BALL: 0.6,
    OD3D_CATEGORIES.BANANA: 0.25,
    OD3D_CATEGORIES.BASEBALLBAT: 0.9,
    OD3D_CATEGORIES.BASEBALLGLOVE: 0.35,
    OD3D_CATEGORIES.BENCH: 1.75,
    OD3D_CATEGORIES.BOAT: 5.,
    OD3D_CATEGORIES.BOTTLE: 0.2,
    OD3D_CATEGORIES.BOWL: 0.225,
    OD3D_CATEGORIES.BROCCOLI: 0.25,
    OD3D_CATEGORIES.CAKE: 0.18,
    OD3D_CATEGORIES.CARROT: 0.17,
    OD3D_CATEGORIES.CELLPHONE: 0.15,
    OD3D_CATEGORIES.COUCH: 2.,
    OD3D_CATEGORIES.CUP: 0.1,
    OD3D_CATEGORIES.DINING_TABLE: 1.2,
    OD3D_CATEGORIES.DONUT: 0.085,
    OD3D_CATEGORIES.FRISBEE: 0.2,
    OD3D_CATEGORIES.HOTDOG: 0.17,
    OD3D_CATEGORIES.KITE: 1.5,
    OD3D_CATEGORIES.MICROWAVE: 0.5,
    OD3D_CATEGORIES.ORANGE: 0.09,
    OD3D_CATEGORIES.PARKINGMETER: 1.25,
    OD3D_CATEGORIES.PIZZA: 0.35,
    OD3D_CATEGORIES.PLANT: 0.4,
    OD3D_CATEGORIES.SANDWICH: 0.125,
    OD3D_CATEGORIES.SKATEBOARD: 0.8,
    OD3D_CATEGORIES.STOPSIGN: 0.75,
    OD3D_CATEGORIES.SUITCASE: 0.7,
    OD3D_CATEGORIES.TV: 1.05,
    OD3D_CATEGORIES.UMBRELLA: 1.1,
    OD3D_CATEGORIES.VASE: 0.35,
    OD3D_CATEGORIES.WINE_GLASS: 0.25,
}

