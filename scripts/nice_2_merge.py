import json
import os 
import glob
import csv
import random
dir = "./playground/data/nice_2"
csv_file_path = os.path.join(dir, "candidate_captions.csv")
json_file_list_0 = sorted(glob.glob(dir+"/choices_vgpt_0_*.json"))
json_file_list_0.sort(key=lambda x: os.path.getmtime(x))
json_file_list_1 = sorted(glob.glob(dir+"/choices_vgpt_1_*.json"))
json_file_list_1.sort(key=lambda x: os.path.getmtime(x))
json_file_list_2 = sorted(glob.glob(dir+"/choices_vgpt_2_*.json"))
json_file_list_2.sort(key=lambda x: os.path.getmtime(x))
json_file_list_3 = sorted(glob.glob(dir+"/choices_vgpt_3_*.json"))
json_file_list_3.sort(key=lambda x: os.path.getmtime(x))


def add_samples(json_samples, json_file_list):
    for json_file in json_file_list:
        with open(json_file, 'r') as f:
            json_samples.extend(json.load(f))
    return json_samples

json_samples = []
json_samples = add_samples(json_samples, json_file_list_0)
json_samples = add_samples(json_samples, json_file_list_1)
json_samples = add_samples(json_samples, json_file_list_2)
json_samples = add_samples(json_samples, json_file_list_3)

answers = {}
num_choice = 0
with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        csv_file = csv.DictReader(f)
        for i, row in enumerate(csv_file):
            # json_samples[i]["id"] = int(row["filename"].split('.')[0])
            choices = json_samples[i]["choice"]
            for choice in choices:
                key = "caption"+str(choice+1)
                if len(row[key]) == 0:
                    continue
                json_samples[i][choice] = row[key]
                num_choice += 1
                if num_choice == 5:
                    break

with open(os.path.join(dir, "merge.json"), 'w') as f:
    json.dump(json_samples, f)
    
# with open(os.path.join(dir, "answers.json"), 'w') as f:
#     json.dump(answers, f)



     