# from .mmlu import MMLU, MMLUInstanceDataset
from .base import Dataset, DataLoader
from .leetcode import LeetCodeHardEval
# from .mmlu_v2 import MMLU_DSPy
from typing import Tuple, Callable
from textgrad import Variable
from textgrad.engine import EngineLM
import random
# def set_seed(seed_value, rnd):
#     rnd.seed(seed_value)
AVAILABLE_DATASETS = [
    "BBH_object_counting",
    "BBH_word_sorting",
    "GSM8K_DSPy",
]

AVAILABLE_INSTANCE_DATASETS = [
    "MMLU_machine_learning",
    "MMLU_college_physics",
    "GPQA_diamond"
    "LeetCodeHardEval"
]

def load_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs) -> Tuple[Dataset, Dataset, Callable]:
    """
    Args:
        task_name: the name of the task to evaluate
        evaluation_api: the engine to use for evaluation, if needed
    """
    if "object_counting" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard, string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        rnd = random.Random()
        random_integer = random.randint(0, 100)
        train_set = BigBenchHard(rnd, random_integer,task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(rnd, random_integer,task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(rnd, random_integer,task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif "BBH" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard
        task_name = task_name[4:]
        random_integer = random.randint(0, 100)
        rnd = random.Random()
        # set_seed(rnd=rnd,seed_value=random_integer)
        train_set = BigBenchHard(rnd,random_integer,task_name, split="train", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        val_set = BigBenchHard(rnd,random_integer,task_name, split="val", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        test_set = BigBenchHard(rnd,random_integer,task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        
        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )
        
        return train_set, val_set, test_set, eval_fn
    
    elif task_name == "GSM8K_DSPy":
        from textgrad.tasks.gsm8k import GSM8K_DSPy
        from .big_bench_hard import string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        evaluation_instruction = "Below is a prediction we got for a question answering task, and the correct final answer. Is the final answer correct? Say only 1 (yes) or 0 (no). Return 1 if and only if the final answer is correct. Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")
        # Should we do train/test like this?
        random_integer = random.randint(0, 100)
        rnd = random.Random()
        # set_seed(rnd=rnd,seed_value=random_integer)
        train_set = GSM8K_DSPy(rnd=rnd,seed=random_integer,split="train", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        val_set = GSM8K_DSPy(rnd=rnd,seed=random_integer,split="val", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        test_set = GSM8K_DSPy(rnd=rnd,seed=random_integer,split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    elif task_name == "BIG_GSM":
        from .mgsm import string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        from .biggsm import BIG_GSM
        evaluation_instruction = "Below is a prediction we got for a question answering task, and the correct final answer. Is the final answer correct? Say only 1 (yes) or 0 (no). Return 1 if and only if the final answer is correct. Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")
        # Should we do train/test like this?
        random_integer = random.randint(0, 100)
        rnd = random.Random()
        # set_seed(rnd=rnd,seed_value=random_integer)
        train_set = BIG_GSM(rnd=rnd,seed=random_integer,split="train", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        val_set = BIG_GSM(rnd=rnd,seed=random_integer,split="val", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        test_set = BIG_GSM(rnd=rnd,seed=random_integer,split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        
        return train_set, val_set, test_set, eval_fn
    elif task_name == "MATH":
        from .math import string_based_equality_fn_2
        from textgrad.autograd.string_based_ops import StringBasedFunction
        from .math import MATH
        evaluation_instruction = "Below is a prediction we got for a question answering task, and the correct final answer. Is the final answer correct? Say only 1 (yes) or 0 (no). Return 1 if and only if the final answer is correct. Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")
        
        random_integer = random.randint(0, 100)
        rnd = random.Random()
        # set_seed(rnd=rnd,seed_value=random_integer)
        train_set = MATH(rnd=rnd,seed=random_integer,split="train", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        val_set = MATH(rnd=rnd,seed=random_integer,split="val", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        test_set = MATH(rnd=rnd,seed=random_integer,split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn_2, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    elif task_name == "LEETCODE":
        from textgrad.autograd.string_based_ops_v2 import StringBasedFunction_v2
        from .leetcode import LEETCODE,string_based_equality_fn_4
        evaluation_instruction = "Below is a prediction we got for a  python programming question, and the correct test case. Is the final code correct? Say only 1 (yes) or 0 (no). Return 1 if and only if the final answer is correct. Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")
        
        random_integer = random.randint(0, 100)
        rnd = random.Random()
        # set_seed(rnd=rnd,seed_value=random_integer)
        train_set = LEETCODE(rnd=rnd,seed=random_integer,split="train", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        val_set = LEETCODE(rnd=rnd,seed=random_integer,split="val", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        test_set = LEETCODE(rnd=rnd,seed=random_integer,split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the predicted python code is correct for test cases."
        eval_fn = StringBasedFunction_v2(string_based_equality_fn_4, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    elif task_name == "MGSM":
        from .mgsm import string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        from .mgsm import MGSM
        evaluation_instruction = "Below is a prediction we got for a question answering task, and the correct final answer. Is the final answer correct? Say only 1 (yes) or 0 (no). Return 1 if and only if the final answer is correct. Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")
        # Should we do train/test like this?
        random_integer = random.randint(0, 100)
        rnd = random.Random()
        # set_seed(rnd=rnd,seed_value=random_integer)
        train_set = MGSM(rnd=rnd,seed=random_integer,split="train", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        val_set = MGSM(rnd=rnd,seed=random_integer,split="val", *args, **kwargs)
        # set_seed(rnd=rnd,seed_value=random_integer)
        test_set = MGSM(rnd=rnd,seed=random_integer,split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        
        return train_set, val_set, test_set, eval_fn