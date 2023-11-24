import json


class JsonConfigUpdater:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = self.load_json()

    def load_json(self):
        try:
            with open(self.json_file_path, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            raise Exception(f"File not found: {self.json_file_path}")
        except json.JSONDecodeError as e:
            raise Exception(f"Error decoding JSON file: {e}")

    def update_value(self, key, new_value):
        if key in self.data:
            self.data[key] = new_value
        else:
            raise KeyError(f"Key '{key}' not found in the JSON file")

    def add_new_key_value(self, key, value):
        self.data[key] = value

    def write_to_file(self):
        try:
            with open(self.json_file_path, 'w') as file:
                json.dump(self.data, file, indent=2)
        except FileNotFoundError:
            raise Exception(f"File not found: {self.json_file_path}")
        except json.JSONDecodeError as e:
            raise Exception(f"Error decoding JSON file: {e}")
