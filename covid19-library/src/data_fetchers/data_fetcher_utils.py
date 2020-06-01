import json
import urllib.request

def get_raw_data_dict(input_url):
    with urllib.request.urlopen(input_url) as url:
        data_dict = json.loads(url.read().decode())
        return data_dict

def load_regional_metadata(filepath):
    with open(filepath, 'r') as fp:
        return json.load(fp)