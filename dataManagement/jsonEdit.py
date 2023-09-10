import os
import json


def replace_words(text, words_to_replace, replacement):
    for word in words_to_replace:
        text = text.replace(word, replacement)
    return text


def process_json_file(file_path, words_to_replace, replacement):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    if isinstance(data, dict):
        # Iterate through the keys and values in the JSON data
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = replace_words(value, words_to_replace, replacement)
            elif isinstance(value, dict):
                data[key] = process_dict(value, words_to_replace, replacement)

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    elif isinstance(data, list):
        # Process each dictionary within the list
        processed_data = []
        for item in data:
            if isinstance(item, dict):
                processed_data.append(process_dict(item, words_to_replace, replacement))
            else:
                processed_data.append(item)

        with open(file_path, 'w') as json_file:
            json.dump(processed_data, json_file, indent=4)


def process_dict(data, words_to_replace, replacement):
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = replace_words(value, words_to_replace, replacement)
        elif isinstance(value, dict):
            data[key] = process_dict(value, words_to_replace, replacement)
    return data


def main():
    directory_path = r'C:\Users\tomhe\Desktop\json_bearbeiten'
    words_to_replace = ['antrum', 'corpus', 'intermediate', 'other']
    replacement = 'noninflamed'

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                process_json_file(file_path, words_to_replace, replacement)
                print(f"Processed: {file_path}")


if __name__ == "__main__":
    main()
