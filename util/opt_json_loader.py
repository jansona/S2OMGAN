import sys
import json

def get_opt_json():

    with open('./train.json', encoding='utf-8') as fin:
        lines = fin.readlines()
        options_str = '\n'.join(lines)

        options_json = json.loads(options_str)
        sys.argv += options_json
