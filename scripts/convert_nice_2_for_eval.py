import json
import csv
import os

dir = "./playground/data/nice_2"
csv_file_path = os.path.join(dir, "pred.csv")
json_file_path = os.path.join(dir, "answers.json")
save_file_path = os.path.join(dir, "pred_update.csv")

with open(json_file_path, 'r') as f:
    answers = json.load(f)

with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        csv_file = csv.reader(f)
        csv_file = list(csv_file)

for i, row in enumerate(csv_file[1:]):
    key = row[1].split(".")[0]
    row[2] = answers[key]

with open(save_file_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_file)