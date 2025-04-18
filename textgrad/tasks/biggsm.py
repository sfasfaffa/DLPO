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

    
    
    
class BIG_GSM(GSM8K):
    def __init__(self,rnd,seed, root:str=None, split: str="train"):
        """DSPy splits for the GSM8K dataset."""
        import tqdm
        import random
        from datasets import load_dataset
        import json
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        file_path = 'data_local/biggsm/data.jsonl'

        records = []

        # 打开并读取文件
        with open(file_path, mode='r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                records.append(record)
        rnd.seed(1001)
        rnd.shuffle(records)
        # 计算分割点
        # split_index = len(records) // 2

        # 分割数据集
        hf_official_train = records[:200]
        hf_official_test= records[200:]

        official_train = []
        official_test = []
        for example in tqdm.tqdm(hf_official_train):
            # print("pass")
            question = example['question']
            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].rstrip('.').replace(',', '')))
            official_train.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            question = example['question']
            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].rstrip('.').replace(',', '')))
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
        rnd.seed(1002)
        rnd.shuffle(official_train)
        trainset = official_train[:190]
        devset = official_train[190:200]
        testset = official_test
        # testset = official_test + official_train
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
