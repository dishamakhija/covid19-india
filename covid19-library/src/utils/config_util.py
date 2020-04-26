import simplejson

def read_config_file(config_file_path):
    with open(config_file_path) as config_file:
        config_data = simplejson.load(config_file, object_pairs_hook=simplejson.OrderedDict)
    # logging.info("input config: " + str(config_data))
    return config_data
