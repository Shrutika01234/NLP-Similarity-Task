import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer, InputExample, losses, util
from tqdm import tqdm
import numpy as np
import pandas as pd
import yaml
import random
import pathlib,os
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    '''
    Seeds everything so as to allow for reproducibility
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dataset(dataset, model):
    '''
    Reading dataset and preprocessing it to get it in the desired forma
    '''
    res = []
    for _, row in dataset.iterrows():
        inp1 = row['text1']
        inp2 = row['text2']
        res.append((inp1, inp2))
    return res


def evaluate(model, iterator):
    '''
    function: Evaluating the model
    Input: model, iterator, optimizer, pad_id
    Returns: epoch_loss, epoch_acc
    '''

    model.eval()
    final_pred = []
    # Predicted value
    with torch.no_grad():
        for inp1, inp2 in tqdm(iterator):
            model.to(device)
            emb1 = model.encode(inp1)
            emb2 = model.encode(inp2)
            
            for i in range(len(emb1)):
                output = util.cos_sim(emb1[i], emb2[i]).numpy()
                final_pred.append(output.item())
    return final_pred


def run(model, root_dir):
    torch.cuda.empty_cache()
    seed_everything(SEED)
    
    final_csv = pd.read_csv(f'{root_dir}/data.csv')
    # final_csv = final_csv.sample(100)

    valid_ds = read_dataset(final_csv, model)
    
    # Dataloader
    valid_loader = DataLoader(
        dataset=valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers)
    
    model = model.to(device)
    
    # Validating
    final_pred = evaluate(model, valid_loader)
    final_csv['similarity'] = final_pred
    final_csv.to_csv('final_pred.csv', index=False)

if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()
    # Path to the config file
    yml_path = f"{base_path}/config/config.yml"
    if not os.path.exists(yml_path):
        print("No such config file exists.")
        exit()
    with open(yml_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    # Extracting parameters from the config file.
    BATCH_SIZE = cfg["params"]["BATCH_SIZE"]
    model_name = cfg["params"]["model_name"]
    device = cfg["params"]["device"]

    SEED = 1234
    num_workers = 2
    # Path to the datset
    root_dir = f"{base_path}/task_data"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    
    # Imports the pretrained model and it's tokenizer
    print("Loading model and weights...")
    model = SentenceTransformer(model_name)
    run(model, root_dir)
