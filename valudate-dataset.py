import json, os

data_dir = "generated/biotech"
for file in os.listdir(data_dir):
    with open(os.path.join(data_dir, file), "r") as f:
        try:
            qa_pairs = json.load(f)
            # check if the file is empty
            if len(qa_pairs) == 0:
                print(f"file {file} is empty")
        else:
            print(f"file {file} is empty")
