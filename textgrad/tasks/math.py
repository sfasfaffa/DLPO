import platformdirs

from .base import Dataset
import textgrad as tg
import re
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
def extract_boxed_content(text):
    # print(text)
    def find_matching_brace(s, start):
        count = 1  #
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    results = []
    pattern = r'\\boxed\{'
    
    for match in re.finditer(pattern, text):
        start = match.end()  # 左括号后的位置
        end = find_matching_brace(text, start)
        
        if end != -1:
            content = text[start:end]
            results.append(content)
    
    return results
# def string_based_equality_fn_2(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    
#     return int(str(prediction.value) == str(ground_truth_answer.value))
def string_based_equality_fn_2(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    import re
    matches = extract_boxed_content(prediction.value)
    if matches:
        if str(matches[-1].replace(" ", "").replace("{", "").replace("}", "")) == str(ground_truth_answer.value):
            return 1
        elif len(matches)>1:
            # print("long!")
            if str(matches[-2].replace(" ", "").replace("{", "").replace("}", "")) == str(ground_truth_answer.value):
                return 1
            if str(matches[0].replace(" ", "").replace("{", "").replace("}", "")) == str(ground_truth_answer.value):
                return 1

    return 0
class MATH(GSM8K):
    def __init__(self, rnd,seed,root:str=None, split: str="train"):
        """DSPy splits for the GSM8K dataset."""
        import tqdm
        import random
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        
        import os
        import random
        import glob
        import json

        base_path = r'D:\DLearning\HIT_scir\tg_2\MATH\train'
        folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        # 
        if len(folders) != 7:
            raise ValueError("The base path does not contain exactly 7 folders.")
        records_train = []
        for i, folder in enumerate(folders):
            folder_path = os.path.join(base_path, folder)
            json_files = glob.glob(os.path.join(folder_path, '*.json'))
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    records_train.append(data)  # 添加解析后的字典
        base_path = r'D:\DLearning\HIT_scir\tg_2\MATH\test'
       
        folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        if len(folders) != 7:
            raise ValueError("The base path does not contain exactly 7 folders.")
        records_test = []
        for i, folder in enumerate(folders):
            folder_path = os.path.join(base_path, folder)
            json_files = glob.glob(os.path.join(folder_path, '*.json'))
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    records_test.append(data) 
        hf_official_train = records_train
        hf_official_test= records_test

        official_train = []
        official_test = []
        for example in tqdm.tqdm(hf_official_train):
            question = example['problem']
            import re
            # match = re.search(r'\\boxed\{(\d+)\}', example['solution'])
            pattern = r'\\boxed\{(.*?)\}'
            matches = extract_boxed_content(example['solution'])
            if not matches:
                continue
            
            answer = str(matches[-1].replace(" ", "").replace("{", "").replace("}", ""))
            solution_without_answer = re.sub(pattern, '', example['solution']).strip()
            gold_reasoning = solution_without_answer.rstrip('. ').strip()
            import time
            # time.sleep(0.2)
            # print(answer)
            # print
            official_train.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))
        print(len(official_train))
        for example in tqdm.tqdm(hf_official_test):
            question = example['problem']
            import re
            # match = re.search(r'\\boxed\{(\d+)\}', example['solution'])
            pattern = r'\\boxed\{(.*?)\}'
            # matches = re.findall(pattern, example['solution'])
            matches = extract_boxed_content(example['solution'])
            if not matches:
                continue
                # raise ValueError("Solution does not contain a boxed answer.")
            answer = str(matches[-1].replace(" ", "").replace("{", "").replace("}", ""))
            solution_without_answer = re.sub(pattern, '', example['solution']).strip()
            gold_reasoning = solution_without_answer.rstrip('. ').strip()
            official_test.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))
        print(len(official_test))
        # for i in official_train:
        #     print("\n")
        #     print(i['answer'])
        #     import time
        #     time.sleep(0.2)
        rnd.seed(seed)
        rnd.shuffle(official_train)
        rnd.seed(seed)
        rnd.shuffle(official_test)
        # print(official_train[0])
        # print(official_train[50])
        # print(official_test[0])
        trainset = official_train[:100]
        devset = official_train[100:300]
        testset = official_test[0:300]

        if split == "train":
            self.data = trainset
        elif split == "val":
            self.data = devset
        elif split == "test":
            self.data = testset
