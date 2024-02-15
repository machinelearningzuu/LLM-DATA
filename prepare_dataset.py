import json, os
import pandas as pd

def load_qna_pairs(data_dirs = [
                                "generated/biotech",
                                "generated/biotech-lfs"
                                ]):
    qa_pair_array = []
    total_nodes = 0
    for data_dir in data_dirs:
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file), "r") as f:
                try:
                    qa_pairs = json.load(f)
                    qa_pair_array.extend(qa_pairs)
                    total_nodes += 1
                
                except:
                    print(f"Error reading file {file}")

    return qa_pair_array, total_nodes

qa_pair_array, total_nodes = load_qna_pairs()
df_qa = pd.DataFrame(qa_pair_array)
df_qa.drop_duplicates(
                    subset=['answer', 'context'],  
                    inplace=True,
                    keep='first'
                    )
df_qa = df_qa[~df_qa.question.str.contains("Question 1:")]
df_qa = df_qa[~df_qa.question.str.contains("Question 2:")]
df_qa = df_qa[df_qa.question != ""]

print("=================================")
print("Total number of nodes : {}".format(total_nodes))
print("Total number of QA pairs : {}".format(len(df_qa)))
print("=================================")
