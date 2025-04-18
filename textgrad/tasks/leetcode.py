import platformdirs
from .base import Dataset
import os
import json
import textgrad as tg


from typing import *
import collections
from collections import *
import math
from math import *
import heapq
from heapq import *
import bisect
import numpy as np
import itertools
from functools import *


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class LeetCodeHardEval(Dataset):
    def __init__(self, root: str = None):
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        self.root = root
        data_path = f"{self.root}/leetcode-hard.jsonl"
        self._check_or_download_dataset()

        self.dataset = [json.loads(line) for line in open(data_path)]
        
        self._task_description = 'You will solve a hard coding problem from LeetCode. You will be given a prompt describing a problem. You need to write a function that passes all the tests.'

    def get_task_description(self):
        return self._task_description

    def _check_or_download_dataset(self):
        data_path = f"{self.root}/leetcode-hard.jsonl"
        if os.path.exists(data_path):
            return
        
        os.makedirs(f"{self.root}/", exist_ok=True)
        import requests
        url = "https://raw.githubusercontent.com/vinid/data/master/leetcode_with_tests.jsonl"
        r = requests.get(url)
        with open(data_path, 'wb') as f:
            f.write(r.content)

    def __getitem__(self, index):
        row = self.dataset[index]
        task_id = row["task_id"]
        prompt = row["prompt"]
        tests = row["test"]

        return task_id, prompt, tests

    def __len__(self):
        return len(self.dataset)
    

import platformdirs

from .base import Dataset

class GSM8K(Dataset):
    def __init__(self, subset:str, root: str=None, split: str="train", *args, **kwargs):
        """
        GSM8K dataset from HF."""
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        self.subset = subset
        assert split in ["train", "val", "test"]
        if split == "test":
            self.data = load_dataset("gsm8k", subset, cache_dir=root, split="test[:300]")
        elif split == "val":
            # Split the training set into half. Let the second half be the training set.
            # Let the first 100 samples be the validation set.
            self.data = load_dataset("gsm8k", subset, cache_dir=root, split="train[:100]")
        elif split == "train":
            self.data = load_dataset("gsm8k", subset, cache_dir=root, split="train[100:]")
        self.split = split
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row["question"]
        answer = row["answer"]
        question_prompt = f"Question: {question}"
        return question_prompt, answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return "You will answer a mathemetical reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
import threading
import queue
import time
import traceback
import sys
def exec_with_timeout(code_str,test):
    result_queue = queue.Queue()
    code_str = '\n'+code_str

    def execution_thread():
        try:
            exec(code_str)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            
            # 构建详细的错误报告
            error_details = {
                'error_type': exc_type.__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'code': code_str,
                'line_number': exc_traceback.tb_lineno if exc_traceback else None,
                'locals': locals(),
                'globals': {k: v for k, v in globals().items() if not k.startswith('_')}
            }
            # 构建错误信息字符串
            error_str = (
                "\n" + "="*50 + " ERROR REPORT " + "="*50 +
                "\nNon-execution error, original Python code compilation error."+
                f"\nerror_type: {error_details['error_type']}"+
                str(e)
            )
            # print("编译错误！")
            # print(code_str)
            # print(error_str)
            result_queue.put(("error", str(error_str)))
            return
        try:
            exec(test)
            result_queue.put(("success", None))
            return
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_details = {
                'error_type': exc_type.__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'code': code_str,
                'line_number': exc_traceback.tb_lineno if exc_traceback else None,
                'locals': locals(),
                'globals': {k: v for k, v in globals().items() if not k.startswith('_')}
            }
            if isinstance(e, AssertionError):
                # 构建错误信息字符串
                error_str = (
                    "\n" + "="*50 + " ERROR REPORT " + "="*50 +
                    "\nCompilation successful, execution error due to test cases."+
                    f"\nerror_type: {error_details['error_type']}" +
                    str(e)
                )
                result_queue.put(("error", str(error_str)))
                # print("test cases错误")
                # print(error_str)
                # print
                return
            error_str = (
                "\n" + "="*50 + " ERROR REPORT " + "="*50 +
                "\nCompilation successful, execution error."+
                f"\nerror_type: {error_details['error_type']}" +
                str(e)
            )
            result_queue.put(("error", str(error_str)))
            # print("执行错误！")
            # print(error_str)
            return
    thread = threading.Thread(target=execution_thread)
    thread.daemon = True
    thread.start()
    
    # 等待5秒
    thread.join(timeout=30)
    
    if thread.is_alive():
        return False, "Execution timed out after 5 seconds"
    
    # 获取执行结果
    if not result_queue.empty():
        status, result = result_queue.get()
        if status == "error":
            return False, f"Execution failed: {result}"
        return True, "Execution successful"
    
    return False, "Unknown error"
    
def string_based_equality_fn_3(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    import re
    pattern1 = r"```python\s*(.*?)\s*```"
    # 匹配``` 和 ``` 之间的代码块，且内容看起来像Python代码
    # pattern2 = r"```\s*((?:[\w\s\(\)=\{\}\[\]:,\.'\"#+\-*/<>!@&|]+\n?)+)\s*```"
    
    # 先尝试匹配明确标记为Python的代码块
    matches = re.findall(pattern1, prediction.value, re.DOTALL)
    
    # 如果没找到明确的Python代码块，尝试匹配可能的Python代码块
    # if not matches:
    #     matches = re.findall(pattern2, prediction.value, re.DOTALL)
    if matches:
        if len(matches) == 1:
            t_or_f,str_infor = exec_with_timeout(matches[0],ground_truth_answer.value)
            if t_or_f:
                return 1
        elif len(matches)>1:
            sorted_list = sorted(matches, key=len, reverse=True)
            for i in sorted_list:
                t_or_f,str_infor = exec_with_timeout(i,ground_truth_answer.value)
                if t_or_f:
                    return 1
        return 0
    return 0
def construct_return_str(acc,expection):
    if expection == "":
        feedback = f'Test cases accuracy:({acc}). Compiler feedback: All test cases Passed!\n'
    else:
        feedback = f'Test cases accuracy:({acc}). Compiler feedback: {expection}.\n'
    return feedback

def string_based_equality_fn_4(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    import re
    pattern1 = r"```python\s*(.*?)\s*```"
    # 匹配``` 和 ``` 之间的代码块，且内容看起来像Python代码
    # pattern2 = r"```\s*((?:[\w\s\(\)=\{\}\[\]:,\.'\"#+\-*/<>!@&|]+\n?)+)\s*```"
    
    # 先尝试匹配明确标记为Python的代码块
    matches = re.findall(pattern1, prediction.value, re.DOTALL)
    
    # 如果没找到明确的Python代码块，尝试匹配可能的Python代码块
    # if not matches:
    #     matches = re.findall(pattern2, prediction.value, re.DOTALL)
    if matches:
        if len(matches) == 1:
            t_or_f,str_infor = exec_with_timeout(matches[0],ground_truth_answer.value)
            if t_or_f:
                return construct_return_str(1,str_infor)
        elif len(matches)>1:
            sorted_list = sorted(matches, key=len, reverse=True)
            longest_feedback = ""
            for step,i in enumerate(sorted_list):
                t_or_f,str_infor = exec_with_timeout(i,ground_truth_answer.value)
                if t_or_f:
                    return construct_return_str(1,str_infor)
                if step == 0:
                    longest_feedback = str_infor
            return construct_return_str(0,longest_feedback)
        return construct_return_str(0,str_infor)
    return construct_return_str(0,None)
    #     try:
            
    #         exec(res)
    #         exec()
    #         # # print('Test passed!')
    #         return 1
    #     except:
            
    #         if len(matches)>1:
    #             res = str(matches[-1])
    #             try:
    #                 exec(res)
    #                 exec(ground_truth_answer.value)
    #                 # print('Test passed!')
    #                 return 1
    #             except:
    #                 # print('Test failed!')
    #                 return 0
    #         # print('Test failed!')
    #         return 0
    #     # return int(str(res) == str())
    # print("没有匹配")
    # return 0
    
class LEETCODE(GSM8K):
    def __init__(self,rnd,seed, root:str=None, split: str="train"):
        """DSPy splits for the GSM8K dataset."""
        import tqdm
        import random
        from datasets import load_dataset
        import json
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        file_path = 'D:/DLearning/HIT_scir/tg_2/textgrad/data_local/leetcode/20240121-Jul.jsonl'
# data_local\leetcode\20240121-Jul.jsonl data_local\leetcode\20240121-Jul-zh.jsonl
        # 初始化一个空列表来存储所有的记录
        records = []
        # 打开并读取文件
        # with open(file_path, mode='r', encoding='utf-8') as file:
        #     for line in file:
        #         # 每一行都是一个独立的JSON对象，使用json.loads()进行解析
        #         record = json.loads(line)
        #         records.append(record)
        with open(file_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
        rnd.seed(seed)
        rnd.shuffle(records)
        # 计算分割点
        split_index = len(records) // 2

        # 分割数据集
        hf_official_train = records[:split_index]
        hf_official_test= records[split_index:]

        official_train = []
        official_test = []
        for example in tqdm.tqdm(hf_official_train):
            question = example['prompt']
            answer = example['test']
            meta = example['meta']
            difficulty = meta['difficulty']
            gold_reasoning = 'none'
            # if difficulty =='Easy':
            official_train.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            question = example['prompt']
            answer = example['test']
            gold_reasoning = 'none'
            official_test.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))
        rnd.seed(seed)
        rnd.shuffle(official_train)
        half = len(official_train)//2
        trainset = official_train[:half]
        
        devset = official_train[half:]
        testset = official_test[0:100]
        # print("train\n")
        # print(trainset[0]['question'][-13])
        # print("val\n")
        # print(devset[0]['question'][-13])
        # trainset = official_train[:200]
        # devset = official_train[200:500]
        # # 从 official_test 中随机选择 200 个样本
        # rng = random.Random(42)  # 使用相同的种子以确保可重复性
        # testset = rng.sample(official_test, k=200)  # 随机选择 200 个样本
        if split == "train":
            self.data = trainset
        elif split == "val":
            self.data = devset
        elif split == "test":
            self.data = testset
