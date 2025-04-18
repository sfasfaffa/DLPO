import platformdirs
import textgrad as tg
from .base import Dataset
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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

import re
def parse_integer_answer(answer: str, only_first_line: bool=False):
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]
        answer = answer.strip()
        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0
    
    return answer
def extract_first_number(s):
    # 查找所有匹配的数字模式，并返回第一个匹配项
    match = re.search(r'\d+', s)
    if match:
        # 如果找到匹配项，返回该匹配项
        # print(int(match.group()))
        return int(match.group())
    else:
        # 如果没有找到匹配项，可以返回None或根据需要返回其他值
        return 0
def extract_numbers(s):
    # 查找所有匹配的数字模式，并返回所有匹配项
    matches = re.findall(r'\d+', s)
    if matches:
        # 如果找到匹配项，返回最后一个匹配项并转换为整数
        return matches
    else:
        # 如果没有找到匹配项，返回0或根据需要返回其他值
        return None
def string_based_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    if int(parse_integer_answer(str(prediction.value)) == int(parse_integer_answer(str(ground_truth_answer.value))))>0:
        return 1
    else:
        matchs = extract_numbers(str(prediction.value))
        if not matchs:
            return 0
        else:
            if int(int(matchs[-1]) == int(parse_integer_answer(str(ground_truth_answer.value))))>0:
                return 1
            matchs.reverse()
            step  = 0
            for i in matchs:
                if int(i)== int(parse_integer_answer(str(ground_truth_answer.value))):
                    return 1
                step += 1
                if step>3:
                    break
                # else:
                #     print(f"ans:{i},gold:{str(parse_integer_answer(str(ground_truth_answer.value)))}")
        # print(f"\nprediction.value:{prediction.value},###ground_truth_answer.value:{ground_truth_answer.value}")
        # print(prediction.value)
        return 0
    
    
class MGSM(GSM8K):
    def __init__(self,rnd,seed, root:str=None, split: str="train"):
        """DSPy splits for the GSM8K dataset."""
        import tqdm
        import random
        from datasets import load_dataset
        import json
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        
        # 初始化一个空列表来存储所有的记录
        records = []
        from datasets import load_dataset, concatenate_datasets

        def load_all_subsets(dataset_name):
            configs = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
            datasets_train = []
            datasets_test = []
            for config in configs:
                ds = load_dataset(dataset_name, config)
                datasets_train.append(ds['train'])  # 假设你只需要训练集
                datasets_train.append(ds['test'])
            return concatenate_datasets(datasets_train)

        # 使用函数加载所有子集
        dataset_train = load_all_subsets("juletxara/mgsm")

        print(dataset_train)
        dataset_train = list(dataset_train)  # 转换为列表
        rnd.seed(1001)
        rnd.shuffle(dataset_train)
        hf_official_train = dataset_train[0:50]
        hf_official_test = dataset_train[50:]
        official_train = []
        official_test = []
        for example in tqdm.tqdm(hf_official_train):
            # print("pass")
            question = example['question']
            answer = example['answer_number']
            gold_reasoning = example['answer']
            official_train.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            question = example['question']
            answer = example['answer_number']
            gold_reasoning = example['answer']
            official_test.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        # rng = random.Random()
        # rng.shuffle(official_train)
        # rng = random.Random()
        # rng.shuffle(official_test)
        # print(official_train[0])
        # print(official_train[50])
        # print(official_test[0])
        # rnd.seed(seed)
        # rnd.shuffle(official_train)
        # rnd.seed(seed)
        # rnd.shuffle(official_test)

        trainset = official_train[:50]
        devset = official_test[:100]
        # testset = official_test
        testset = official_test[200:]
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
