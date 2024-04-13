import json
import os 
import glob
import csv
import random
dir = "./playground/data/nice_2"
csv_file_path = os.path.join(dir, "candidate_captions.csv")
json_file_list_0 = sorted(glob.glob(dir+"/choices_0_*.json"))
json_file_list_0.sort(key=lambda x: os.path.getmtime(x))
json_file_list_1 = sorted(glob.glob(dir+"/choices_1_*.json"))
json_file_list_1.sort(key=lambda x: os.path.getmtime(x))


def add_samples(json_samples, json_file_list):
    for json_file in json_file_list:
        with open(json_file, 'r') as f:
            json_samples.extend(json.load(f))
    return json_samples

json_samples = []
json_samples = add_samples(json_samples, json_file_list_0)
json_samples = add_samples(json_samples, json_file_list_1)
answers = {}

with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        csv_file = csv.DictReader(f)
        for i, row in enumerate(csv_file):
            json_samples[i]["id"] = int(row["filename"].split('.')[0])
            choice = json_samples[i]["choice"]
            key = "caption"+str(choice)
            if len(row[key]) == 0:
                key = "caption"+str(random.randint(1,63))
            json_samples[i]["caption"] = row[key]
            answers[json_samples[i]["id"]] = json_samples[i]["caption"]

with open(os.path.join(dir, "merge.json"), 'w') as f:
    json.dump(json_samples, f)
    
with open(os.path.join(dir, "answers.json"), 'w') as f:
    json.dump(answers, f)



     