import json
import datetime


# from src.main import generate_tuning_data


def read_json(json_file_path: str) -> dict:
    with open(json_file_path) as json_file:
        return json.load(json_file)


def get_properties(obj: dict, parent: str = None):
    attributes = []
    for key, value in obj.items():
        if isinstance(value, dict):
            parent_label = "{}.{}".format(parent, key) if parent is not None else key
            attributes.extend(get_properties(value, parent_label))
        elif isinstance(value, list):
            parent_label = "{}.{}_list".format(parent, key) if parent is not None else key
            if len(value) > 0 and isinstance(value[0], dict):
                attributes.extend(get_properties(value[0], parent_label))
            else:
                parent_label = "{}_list.{}".format(parent, key) if parent is not None else "{}_list".format(key)
                attributes.append({
                    'parent': '{}'.format(parent_label),
                    'key': key,
                    'data_type': type(value[0]).__name__ if len(value) > 0 else "Unknown"
                })
        else:
            attributes.append(
                {
                    'parent': '{}'.format(parent),
                    'key': key,
                    'data_type': type(value).__name__
                }
            )
    return attributes


if __name__ == "__main__":
    print(datetime.date.fromtimestamp(904867200))
    # data = read_json('../resources/data/raw-it_support_proffessional.json')
    # if data[0]['data'] is not None:
    #     attr = get_properties(data[0]['data'])
    #     print(attr)
    # generate_tuning_data()

