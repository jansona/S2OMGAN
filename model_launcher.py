import sys, json
from train_platform import train_function
from test_platform import test_function


TRAIN = 'train'
TEST = 'test'

def __main__():
    json_file_name = sys.argv[1]

    params = []

    with open(json_file_name, 'rb') as fin:
        opt_json = json.load(fin)

        for k, v in opt_json.items():
            if k == 'task':
                action = v
            else:
                params.append("--{}".format(k))
                params.append("{}".format(v))

    print(params)
    if action == TRAIN:
        train_function(['./train_platform.py'] + params)
    elif action == TEST:
        test_function(['./test_platform.py'] + params)
    else:
        print("param 'task' err")


if __name__ == "__main__":
    __main__()
