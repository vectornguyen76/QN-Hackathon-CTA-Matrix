from typing_extensions import runtime
from sklearn.metrics import f1_score


import pandas as pd
import hashlib
import sys

import json
import argparse
import requests
from tqdm import tqdm
from datetime import datetime
import time
import logging
import boto3

# Aws config

# Logger configure
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

LABELS_COLUMNS = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']



def load_dataset(path='public.csv'):
    df = pd.read_csv(path)
    df.fillna(0, inplace=True)
    return df
    
def convert2classify(arr):
    return [i if i == 0 else 1 for i in arr]


def cal_f1score(gth, pred):
    gth_cus = convert2classify(gth)
    pred_cus = convert2classify(pred)
    return f1_score(y_true=gth_cus, y_pred=pred_cus, zero_division =True)

def cal_r2score(gth, pred):
    l2, s2= 0, 0

    new_gth = []
    new_pred = []

    for i in range(len(gth)):
        if gth[i] !=0 and pred[i] != 0:
            new_gth.append(gth[i])
            new_pred.append(pred[i])

    if len(new_gth) == 0:
        return 1
    mean = sum(new_gth) / len(new_gth)

    for i in range(len(new_gth)):
            l2 += (new_gth[i] - new_pred[i]) ** 2
            s2 += (new_gth[i] - mean) ** 2
    return 1 - l2 / (len(new_gth) * 16)

def get_pred_from_gth(gth, api):
    data_hash = {"Review": [],
            "giai_tri": [],
            "luu_tru": [],
            "nha_hang": [],
            "an_uong":[],
            "di_chuyen": [],
            "mua_sam": []}

    start = time.time()
    for idx in tqdm(range(len(gth))):
        raw_text = gth.loc[idx, ["Review"]].values[0]
        url = "{}/review-solver/solve?review_sentence={}".format(api, raw_text)
        # response = requests.get(url)
        # data_out = response.text
        try:
            response = requests.get(url)
            data_out = response.text
            data = json.loads(data_out)
        except Exception as e:
            return "API", 0, e

        data_hash["Review"].append(data["review"])
        data_hash["giai_tri"].append(data["results"]["giai_tri"])
        data_hash["luu_tru"].append(data["results"]["luu_tru"])
        data_hash["nha_hang"].append(data["results"]["nha_hang"])
        data_hash["an_uong"].append(data["results"]["an_uong"])
        data_hash["di_chuyen"].append(data["results"]["di_chuyen"])
        data_hash["mua_sam"].append(data["results"]["mua_sam"])

        run_time = time.time() - start
        if run_time > 1200:
            return "TIMEOUT", run_time, data_out

    pred_df = pd.DataFrame.from_dict(data_hash)
    return pred_df, run_time, data_out


def final_metrics(path_gth, email, api):
    '''
    input: path of 3 files path_gth: file ground_truth; path_pred: file predict; path_res: file result, default='result.json'.
    output: save result into file path_res with json format:
        {'giai_tri': score, 'luu_tru': score, 'nha_hang': score, 'an_uong': score, 'di_chuyen': score, 'mua_sam': score, 'final_score': score}.
    file ground_truth and predict with format csv with 7 ordered columns: Review, giai_tri, luu_tru, nha_hang, an_uong, di_chuyen, mua_sam.

    '''

    gth = load_dataset(path_gth)
    pred, run_time, data_out = get_pred_from_gth(gth, api)
    if (type(pred) == str) and (pred == "TIMEOUT"):
        return 0, run_time, data_out
    elif (type(pred) == str) and (pred == "API"):
        return "API", 0, data_out
    else:
        res = dict()
        score_total = 0
        for aspect in LABELS_COLUMNS:
            gth_aspect = gth[aspect]
            pred_aspect = pred[aspect]
            try:
                f1 = cal_f1score(gth=gth_aspect, pred=pred_aspect)
                r2 = cal_r2score(gth=gth_aspect, pred=pred_aspect)
            except:
                return "FORMAT", 0
            res[aspect] = f1 * r2
            score_total += res[aspect]    
        score = score_total / len(LABELS_COLUMNS)
        
        ### push predition to aws s3
      
        return score, run_time, data_out

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--groundtruth', '-g', type=str, required=True)
    parser.add_argument('--api', '-a', type=str, required=True)
    parser.add_argument('--email', '-e', type=str, required=True)
    args = parser.parse_args()

    url = ''
    current_time = int(round(time.time() * 1000))

    body = {
        "md5": hashlib.md5(f"QAI_{current_time}_{args.email}".encode()).hexdigest(),
        "email": args.email,
        "time": current_time,
        "leader_board": [
            {
                "score": -1,
                "submit_time": str(datetime.now().ctime()),
                "run_time": -1,
                "topic": 2,
                "eval_status": "",
                "eval_error": ""
            },
        ]
    }

    try:
        #time.sleep(30)
        score, run_time, data_out = final_metrics(path_gth=args.groundtruth, email=args.email, api=args.api)
        body["leader_board"][0]["run_time"] = run_time
        #==> TuND update 2022-08-22
        if (type(score) == str) and (score == "API"):
            body["leader_board"][0]["score"] = 0
            body["leader_board"][0]["eval_status"] = "failed"
            body["leader_board"][0]["eval_error"] = f"CAN NOT DATA FROM YOUR API!"
            body["leader_board"][0]["eval_error_detail"] = str(data_out)
        elif (type(score) == str) and (score == "FORMAT"):
            body["leader_board"][0]["score"] = 0
            body["leader_board"][0]["eval_status"] = "failed"
            body["leader_board"][0]["eval_error"] = f"PREDICTION DATA WRONG FORMAT!\n{data_out}"
        elif run_time > 1200:
            body["leader_board"][0]["score"] = score
            body["leader_board"][0]["eval_status"] = "failed"
            body["leader_board"][0]["eval_error"] = "TIMEOUT!"
        else:
            body["leader_board"][0]["score"] = score
            body["leader_board"][0]["eval_status"] = "succeeded"
        #<== TuND update 2022-08-22
    except Exception as e:
        logger.exception(e)
        body["leader_board"][0]["eval_status"] = "failed"
        #==> TuND update 2022-08-22
        # body["leader_board"][0]["eval_error"] = str(e)
        body["leader_board"][0]["eval_error"] = "EVALUATE ERROR!"
        body["leader_board"][0]["eval_error_detail"] = str(e)
        #<== TuND update 2022-08-22

   
    
   
    ## terminal:
    ## python metrics.py -g chall_02_private_test.csv -a http://127.0.0.1:<local-port> -e lampt13@fsoft.com.vn
