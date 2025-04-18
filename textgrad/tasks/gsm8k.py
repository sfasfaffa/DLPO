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

    
    
    
class GSM8K_DSPy(GSM8K):
    def __init__(self,rnd,seed, root:str=None, split: str="train"):
        """DSPy splits for the GSM8K dataset."""
        import tqdm
        import random
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        dataset = load_dataset("gsm8k", 'main', cache_dir=root)
        hf_official_train = dataset['train']
        hf_official_test = dataset['test']
        official_train = []
        official_test = []
        for example in tqdm.tqdm(hf_official_train):
            question = example['question']
            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))
            official_train.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            question = example['question']
            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))
            official_test.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))
        
        rnd.seed(seed)
        rnd.shuffle(official_train)
        rnd.seed(seed)
        rnd.shuffle(official_test)
        # print(official_train[0])
        # print(official_train[50])
        # print(official_test[0])
        # """
        trainset = official_train[0:50]
        devset = official_train[50:150]
        # """

        """        
        trainset = official_train[0:100]
        devset = official_train[100:200]
        """
        testset = official_test[0:400]
        if split == "train":
            self.data = trainset
        elif split == "val":
            self.data = devset
        elif split == "test":
            self.data = testset
