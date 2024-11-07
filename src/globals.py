RESOURCE_FOLDER="../../Project materials(1)"
MARKOV_SIZE=2
STEPSIZES = [1] #3, 6, 18, 36, 72, 144, 216, 288, 360]
OUTPUT_WINDOW = 1
INPUT_WINDOW = 4
OUTPUT_FORECAST = ["latitude", "longitude","sog", "cog"]
ONE_HOT_COLUMNS = ["navstat_cat"] #rot_cat
DELETEABLE_COLUMNS = ["vesselId", "portId", "etaParsed", "UN_LOCODE", "ISO", "navstat", "rot"]
