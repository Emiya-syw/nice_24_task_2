import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
import csv

from imp_llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from imp_llava.conversation import conv_templates, SeparatorStyle
from imp_llava.model.builder import load_pretrained_model
from imp_llava.utils import disable_torch_init
from imp_llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader
from imp_llava import make_supervised_data_module

from PIL import Image
import math

from dataclasses import dataclass, field

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

@dataclass # 数据参数
class DataArguments:
    data_path: str = field(default="./playground/data/nice_2/candidate_captions.json",
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    image_folder: str = field(default="./playground/data/")
    image_aspect_ratio: str = 'square'
    mm_use_im_start_end: bool = field(default=False)

def eval_model(args):
    disable_torch_init()
    # 将路径字符串中的波浪线扩展为用户的主目录, 用于跨平台的路径展开功能
    # ~/Documents/file.txt -> /path/to/user/Documents/file.txt
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    data_args=DataArguments()
    data_args.image_processor = image_processor
    tokenizer.pad_token = tokenizer.unk_token
    dataset = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, n=args.num_chunks, k=args.chunk_idx)
    bs = 32
    dataloader = DataLoader(
        dataset = dataset["train_dataset"],
        collate_fn = dataset["data_collator"],
        batch_size = bs,
        shuffle = False
    )
    scores = []
    choices = []
    ids = []
    with open("./playground/data/nice_2/candidate_captions.csv", 'r', encoding='utf-8', errors='ignore') as f:
        csv_file = csv.DictReader(f)
        for i, row in enumerate(csv_file):
            # print(i)
            ids.append(int(row["filename"].split('.')[0])) 
    ids = get_chunk(ids, args.num_chunks, args.chunk_idx)
    interval = 500
    with torch.inference_mode():
        with torch.no_grad():
            # print(dataset["train_dataset"][1])
            for i, batch in tqdm(enumerate(dataloader)):
                output = model(input_ids=batch["input_ids"].to("cuda"),
                            labels=batch["labels"].to("cuda"),
                            images=batch["images"].half().to("cuda"),
                                use_cache=True)
                
                scores.append(torch.sum(output["loss"].reshape(bs, -1), dim=-1))
                
                if (i+1)%(64//bs) == 0 :
                    choice = np.argsort(torch.cat(scores).cpu().numpy()).tolist()
                    # print(choice)
                    choices.append({"id":ids[int((i+1)/(64//bs))-1],"choice":choice})
                    # print(choices)
                    scores = []
                    
                if (i+1)/(64//bs)!=0 and (i+1)/(64//bs)%interval == 0:
                    id = int(int((i+1)/(64//bs))/interval)
                    with open(f'./playground/data/nice_2/choices_vgpt_{args.chunk_idx}_{id}.json', 'w') as f:
                        json.dump(choices, f)
                    choices = []
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--csv-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)