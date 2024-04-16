import json
import csv
import argparse
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm

def main(args):
    with open(args.origin_candidate_path, 'r', newline='', encoding='utf-8', errors='ignore') as file_o, open(args.aug_candidate_path, 'r') as file_r:
        data_r = json.load(file_r)
        reader = csv.reader(file_o)
        gts_data = {}
        for i, row in enumerate(reader):
            if i == 0:
                continue
            id = row[0]
            if id in data_r.keys():
                gts_data[int(id.strip('.jpg'))] = data_r[id]
                
            else:
                gts_data[int(id.strip('.jpg'))] = row[1:]

    with open(args.pred_path,'r') as file:
        data_list = json.load(file)

    res = {}
    gts = {}
    total_score = {}
    for choice_in_c in range(5):
        for idx, data in tqdm(enumerate(data_list), total=args.num_rows):
            id = data['id']
            res[id] = []
            for key in data.keys():
                if key != 'id' and key != 'choice':
                    res[id].append(data[key]) 
            gts[id] = gts_data[id]
            res[id] = res[id][choice_in_c:choice_in_c + 1]
            if (idx + 1) % args.batch_size == 0 or idx == args.num_rows - 1:
                _ , scores = Cider().compute_score(gts, res)
                for batch_id, captions, score in zip(res.keys(), res.values(), scores):
                    if choice_in_c == 0:
                        total_score[batch_id] = {}
                    total_score[batch_id][captions[0]] = score
                    res = {}
                    gts = {}
    
    selected_dict = {}
    for key in total_score:
        max_score = max(total_score[key].values())
        max_key = [k for k, v in total_score[key].items() if v == max_score][0]   
        selected_dict[key] =  max_key
    
    with open(args.out_path, 'w') as file:
        json.dump(selected_dict, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-candidate-path", type=str, default="/home/cyn/nice2024/candidate_captions.csv")
    parser.add_argument("--aug-candidate-path", type=str, default='NICE72.json')
    parser.add_argument("--pred-path", type=str, default='merge.json')
    parser.add_argument("--out-path", type=str, default='selected.json')
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-rows", type=int, default=20000)
    args = parser.parse_args()
    main(args)