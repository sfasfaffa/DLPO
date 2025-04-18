from abc import ABC, abstractmethod
from typing import List, Union
from collections import defaultdict
from textgrad.variable import Variable
from textgrad import logger
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .optimizer_prompts import  OPTIMIZER_SYSTEM_PROMPT, GRADIENT_TEMPLATE, GRADIENT_MULTIPART_TEMPLATE,construct_tgd_promptV2,TGD_LR_PROMPT
import copy

def get_gradient_and_context_text(variable) -> Union[str, List[Union[str, bytes]]]:
    """For the variable, aggregates and returns 
    i. the gradients 
    ii. the context for which the gradients are computed.

    This is used by the optimizer.  
    :return: A string containing the aggregated gradients and their corresponding context.
    :rtype: str
    """

    gradient_content = []
    for g in variable.gradients:
        if variable.gradients_context[g] is None:
            gradient_content.append(g.value)
        else:
            # If context is a list, we handle it differently.
            context = variable.gradients_context[g]
            if isinstance(context["context"], str):
                # The context could be all string.
                criticism_and_context = GRADIENT_TEMPLATE.format(
                    feedback=g.value, **context)
                gradient_content.append(criticism_and_context)
            elif isinstance(context["context"], list):
                # The context may have a list of images / strings. In this case, we need to handle it differently.
                context_prompt = GRADIENT_MULTIPART_TEMPLATE.format(**context, feedback=g.value)
                criticism_and_context = context["context"] + [context_prompt]
                gradient_content.extend(criticism_and_context)
            else:
                raise ValueError("Context must be either a string or a list.")
    
    # Check if all instances are string
    if all(isinstance(i, str) for i in gradient_content):
        return "\n".join(gradient_content)
    else:
        return gradient_content


class Optimizer(ABC):
    """
    Base class for all optimizers.

    :param parameters: The list of parameters to optimize.
    :type parameters: List[Variable]

    :Methods:
        - zero_grad(): Clears the gradients of all parameters.
        - step(): Performs a single optimization step.
    """

    def __init__(self, parameters: List[Variable]):
        for parameter in parameters:
            if type(parameter.value) !=  str:
                raise NotImplementedError(f"We cannot yet update multimodal content and this data type: {type(parameter.value)}. We can only evaluate gradients using multimodal models. This may change soon (looking at you, GPT-5).")
        self.parameters = parameters
        
    def zero_grad(self):
        """
        Clears the gradients of all parameters.
        """
        for p in self.parameters:
            p.gradients = set()

    @abstractmethod
    def step(self):
        """
        Performs a single optimization step.
        """
        pass

class TGD_OPT(Optimizer):
    def __init__(self, 
                 parameters: List[Variable], 
                 verbose: int=0, 
                 engine: Union[EngineLM, str]=None, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str]=None,
                 gradient_memory: int=0,
                 learning_rate: int=0,
                 weight_decay: float = 0.0,
                 dropout: float = 0.0,
                 momentum: int = 0,
                 acc_mem: int = 0,
                 dy_lr: int = 0,
                 tsa_tem_decay: int = 2,
                tsa_tem: int = 64,
                 ):
        """TextualGradientDescent optimizer

        :param engine: the engine to use for updating variables
        :type engine: EngineLM
        :param parameters: the parameters to optimize
        :type parameters: List[Variable]
        :param verbose: whether to print iterations, defaults to 0
        :type verbose: int, optional
        :param constraints: a list of natural language constraints, defaults to []
        :type constraints: List[str], optional
        :param optimizer_system_prompt: system prompt to the optimizer, defaults to textgrad.prompts.OPTIMIZER_SYSTEM_PROMPT. Needs to accept new_variable_start_tag and new_variable_end_tag
        :type optimizer_system_prompt: str, optional
        :param in_context_examples: a list of in-context examples, defaults to []
        :type in_context_examples: List[str], optional
        :param gradient_memory: the number of past gradients to store, defaults to 0
        :type gradient_memory: int, optional
        :param learning_rate: the number of changed sentence, defaults to 0
        :type learning_rate: int, optional
        """
        super().__init__(parameters)

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        self.verbose = verbose
        self.constraints = constraints if constraints is not None else []
        """"""
        self.learning_rate = learning_rate
        self.do_learning_rate = (learning_rate>0)
        self.dropout = dropout
        self.weight_decay = weight_decay
        # print(learning_rate)
        # print(dropout)
        # print(weight_decay)
        """"""
        # if self.do_learning_rate:
        #     self.lr_prompt = f
# Limit the number of your updates to be less than the learning rate. Keep other sentences unchanged.
#  by updating the most important ones and discarding the relatively unimportant ones.
        #     self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1])+lr_prompt
        # else:
        self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1]) 
        # print(self.optimizer_system_prompt)
        self.do_constrained = (len(self.constraints) > 0)
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)
        self.gradient_memory = gradient_memory
        self.gradient_memory_dict = defaultdict(list)
        self.do_gradient_memory = (gradient_memory > 0)
        self.train_acc_memory = []
        self.acc_mem = acc_mem
        self.train_acc_memory_2 = []
        self.gradient_memory_tgd = []
        self.momentum = momentum
        self.annealing_rate = 0.3


        self.temprature = tsa_tem
        self.tem_decay = tsa_tem_decay
        self.last_acc = 0
        self.dy_lr = dy_lr
        self.step_index = 0
    @property
    def constraint_text(self):
        """
        Returns a formatted string representation of the constraints.

        :return: A string containing the constraints in the format "Constraint {index}: {constraint}".
        :rtype: str
        """
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)
    
    def get_gradient_memory_text(self, variable: Variable):
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[variable][-self.gradient_memory:]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i+1}> {grad_info['value']}</FEEDBACK-{i+1}>\n"
        return grad_memory
    
    def update_gradient_memory(self, variable: Variable):
        self.gradient_memory_dict[variable].append({"value": variable.get_gradient_text()})
    
    def _update_prompt(self, variable: Variable) -> Union[str, List[Union[str, bytes]]]:
        grad_memory = self.get_gradient_memory_text(variable)
        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": get_gradient_and_context_text(variable),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
            "gradient_memory": grad_memory,
            "learning_rate":self.learning_rate,
            "gradient_memory_2":self.gradient_memory_tgd
        }
        prompt = construct_tgd_promptV2(do_constrained=self.do_constrained, 
                                      do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                                      do_gradient_memory=(self.do_gradient_memory and (grad_memory != "")),
                                      lr=self.learning_rate,
                                      dropout=self.dropout,
                                      weight_decay=self.weight_decay,
                                      origin_var=variable.value,
                                      train_acc_mem=self.train_acc_memory,
                                      train_acc_mem_2=self.train_acc_memory_2,
                                      **optimizer_information)
        # prompt = construct_tgd_promptV2(do_constrained=self.do_constrained, 
        #                               do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
        #                               do_gradient_memory=(self.do_gradient_memory and (grad_memory != "")),
        #                               lr_prompt=self.lr_prompt,
        #                               **optimizer_information)
        
        logger.info(f"TextualGradientDescent prompt for update", extra={"prompt": prompt})
        # print(prompt)
        return prompt

    def step(self):
        """
        Perform a single optimization step.
        This method updates the parameters of the optimizer by generating new text using the engine and updating the parameter values accordingly.
        It also logs the optimizer response and the updated text.
        Returns:
            None
        """
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            
            # print("prompt_update_parameter:")
            # print(prompt_update_parameter)
            # print("self.optimizer_system_prompt:")
            # print(self.optimizer_system_prompt)
            
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            # print("new_text:<")
            # print(new_text)
            # print(">")
            logger.info(f"TextualGradientDescent optimizer response", extra={"optimizer.response": new_text})
            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            # Check if we got a cannot be indexed error
            except IndexError:
                logger.error(f"TextualGradientDescent optimizer response could not be indexed", extra={"optimizer.response": new_text})
                raise IndexError(f"TextualGradientDescent optimizer response could not be indexed. This can happen if the optimizer model cannot follow the instructions. You can try using a stronger model, or somehow reducing the context of the optimization. Response: {new_text}")
            parameter.set_value(new_value)
            logger.info(f"TextualGradientDescent updated text", extra={"parameter.value": parameter.value})
            if self.verbose:
                print("-----------------------TextualGradientDescent------------------------")
                print(parameter.value)
            
            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)
        self.step_index += 1
        self.gradient_memory_tgd.append(parameter.get_gradient_text())
        if self.dy_lr>0:
            self.learning_rate = max(self.dy_lr-self.step_index,1)
    def add_train_acc_mem(self,acc,prompt):
        if self.acc_mem>0:
            prompt_ = f'{prompt}(train_accuracy:{str(acc)})'
            self.train_acc_memory.append([acc,prompt_])
            self.train_acc_memory = sorted(self.train_acc_memory, key=lambda x: x[0],reverse=True)

        if self.momentum>0:
            prompt__ = prompt
            self.train_acc_memory_2.append([acc,prompt__])
            if len(self.train_acc_memory_2)>0:
                if self.train_acc_memory_2[-1][1]!=prompt__:
                    self.train_acc_memory_2.append([acc,prompt__])
            else:
                self.train_acc_memory_2.append([acc,prompt__])
            self.train_acc_memory_2.pop(0)
    def add_train_acc_mem2(self,acc,prompt):
        if self.acc_mem>0:
            prompt_ = f'{prompt}(train_accuracy:{str(acc)})'
            self.train_acc_memory.append([acc,prompt_])
            self.train_acc_memory = sorted(self.train_acc_memory, key=lambda x: x[0],reverse=True)

    def annealing_step(self,eval_dataset,test_set = None,eval_fn = None,model = None,epoch_sum = None):

        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            
            # print("prompt_update_parameter:")
            # print(prompt_update_parameter)
            # print("self.optimizer_system_prompt:")
            # print(self.optimizer_system_prompt)
            
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            # print("new_text:<")
            # print(new_text)
            # print(">")
            logger.info(f"TextualGradientDescent optimizer response", extra={"optimizer.response": new_text})
            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            # Check if we got a cannot be indexed error
            
            except IndexError:
                logger.error(f"TextualGradientDescent optimizer response could not be indexed", extra={"optimizer.response": new_text})
                raise IndexError(f"TextualGradientDescent optimizer response could not be indexed. This can happen if the optimizer model cannot follow the instructions. You can try using a stronger model, or somehow reducing the context of the optimization. Response: {new_text}")
            import copy
            old_value = copy.copy(parameter.get_value())
            print(f"old_value:{old_value}")
            print(f"new_value:{new_value}")
            parameter.set_value(new_value)
            test_acc = eval_dataset(test_set, eval_fn, model,summary = epoch_sum)
            # print(f"now_value:{model.system_prompt.get_value()}")
            correct_count = sum(test_acc)  # 计算答对的数量
            total_count = len(test_acc)     # 总问题数量
            accuracy = correct_count / total_count  # 计算正确率
            # print(self.annealing_rate)
            if accuracy<self.last_acc:
                # print(self.annealing_rate)
                if self.last_acc-accuracy>self.annealing_rate:
                    parameter.set_value(old_value)
                    print(f"now_value:{model.system_prompt.get_value()}")
                    self.add_train_acc_mem2(accuracy,new_value)
                    # print("return;")
                    return
                else:
                    import math
                    import random
                    p = math.exp((accuracy-self.last_acc)/self.temprature)
                    random_number = random.random()
                    # print(f'p:{p},1-p:{1-p}')
                    if random_number>p:
                        parameter.set_value(old_value)
                        print(f"now_value:{model.system_prompt.get_value()}")
                        self.add_train_acc_mem2(accuracy,new_value)
                        # print("return;")
                        return
            # parameter.set_value(old_value)
            # print(f"now_value:{model.system_prompt.get_value()}")
            # # print("contnue;")
            self.last_acc = accuracy
            logger.info(f"TextualGradientDescent updated text", extra={"parameter.value": parameter.value})
            if self.verbose:
                print("-----------------------TextualGradientDescent------------------------")
                print(parameter.value)
            
            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)
        """"""
        # biggsm超参:
        # self.temprature = self.temprature/1.75
        # MGSM超参:
        # self.temprature = self.temprature/2
        # BBH超参:
        # self.temprature = self.temprature/4
        # MMLU超参:
        # self.temprature = self.temprature/16
        """"""
        self.temprature = self.temprature/self.tem_decay
        self.step_index += 1
        if self.dy_lr>0:
            if self.step_index<4:
                self.learning_rate = 1
            self.learning_rate = max(self.dy_lr-self.step_index,1)
        self.gradient_memory_tgd.append(parameter.get_gradient_text())
        