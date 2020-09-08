import sys, json, requests, base64
from train_platform import train_function
from test_platform import test_function


TRAIN = 'train'
TEST = 'test'
ADD_NODE = 'add_node'

MAPNIK_SERVICE_URL = "http://125.220.157.225:8383/add_note"


def get_noded_img(location, output_path):
    raw_data_b64 = requests.get("{}?location={}".format(MAPNIK_SERVICE_URL, location)).content
    # print(raw_data_b64)
    img_data = base64.b64decode(raw_data_b64)
    
    with open(output_path, "wb") as fout:
        fout.write(img_data)

def __main__():
    json_file_name = sys.argv[1]

    params = []
    action = ""

    with open(json_file_name, 'rb') as fin:
        opt_json = json.load(fin)

        if "task" in opt_json.keys():
            for k, v in opt_json.items():
                if k == 'task':
                    action = v
                else:
                    params.append("--{}".format(k))
                    params.append("{}".format(v))
        else:
            print("Please assign param 'task'")

    if action == TRAIN:
        train_function(['./train_platform.py'] + params)
    elif action == TEST:
        test_function(['./test_platform.py'] + params)
    elif action == ADD_NODE:
        print("Tagging...")
        get_noded_img(opt_json['data'], opt_json['outPath'])
    else:
        print("param 'task' err")


if __name__ == "__main__":
    __main__()
