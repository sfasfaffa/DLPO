GLOSSARY_TEXT = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""
import random
### Optimize Prompts

# System prompt to TGD
OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
    "This is very important: You MUST give your response by sending the improved variable between {new_variable_start_tag} {{improved variable}} {new_variable_end_tag} tags. "
    "The text you send between the tags will directly replace the variable.\n\n"
    f"{GLOSSARY_TEXT}"
)

# TGD update instruction
TGD_PROMPT_PREFIX = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
    "<CONTEXT><FEEDBACK>{variable_grad}</FEEDBACK></CONTEXT>\n\n"
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)


# If the gradients are in a multi-part container
TGD_MULTIPART_PROMPT_INIT = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
)

TGD_MULTIPART_PROMPT_PREFIX = (
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)

TGD_PROMPT_SUFFIX  = (
    "\n\nThe original variable is the text within the following span: <VARIABLE> {variable_value} </VARIABLE>\n\n"
    "Send the improved variable "
    "in the following format:\n\n{new_variable_start_tag}{{the improved variable}}{new_variable_end_tag}\n\n"
    "Improve the variable based on all the above requirements and feedback , then send the final version between <IMPROVED_VARIABLE> tags."
)

MOMENTUM_PROMPT_ADDITION = (
    "Here are the past iterations of this variable:\n\n"
    "<PAST_ITERATIONS>{past_values}</PAST_ITERATIONS>\n\n"
    "Similar feedbacks across different steps suggests that the modifications to the variable are insufficient." 
    "If this is the case, please make more significant changes to the variable.\n\n"
)

CONSTRAINT_PROMPT_ADDITION = (
    "You must follow the following constraints:\n\n"
    "<CONSTRAINTS>{constraint_text}</CONSTRAINTS>\n\n"
)

IN_CONTEXT_EXAMPLE_PROMPT_ADDITION = (
    "You must base on the following examples when modifying the {variable_desc}:\n\n"
    "<EXAMPLES>{in_context_examples}</EXAMPLES>\n\n"
)

"""
DLPO prompts:
"""

TGD_LR_PROMPT_V0 = """
\nYou need to update the original variable on a sentence level, and the number of updates (including adding sentence, deleting sentence, and modifying sentence) should be limited to a specific quantity (which we call the 'learning rate').
If the learning rate is: 4, here's an example:
Initial:
<VARIABLE>
As a Math Calculator, please solve:

Required Steps:
1. Identify problem type
2. Show calculation steps

Output Format:
- Process:
- Final Result:
- Verification:
</VARIABLE>

Modified Version with exactly 4 changes:
<IMPROVED_VARIABLE>
As a reasoning Engine, please solve:  [modifying sentence]

Required Steps:
1. Identify problem type
2. Show calculation steps
3. Analyze complexity [adding sentence]
4. Assess stability [adding sentence]

Output Format:
- Process:
- Final Result:   [deleting sentence 'Verification:']
</IMPROVED_VARIABLE>
Conclusion:
1(modify) + 2(add) + 1(delete) = 4.

Your learning rate is: {learning_rate}. For each optimize step, please make {learning_rate} update(s) to the original sentences and keep the other unchanged.
"""
TGD_LR_PROMPT_SAMPLE = (
"\nYou need to update the original variable on a sentence level, and the number of updates (including adding sentence, deleting sentence, and modifying sentence) should be limited to a specific quantity (which we call the 'learning rate')."
"Your learning rate is: {learning_rate}. For this optimize step, please make {learning_rate} update(s) to the original sentences you think need changing and keep the rest of the prompt you think is already good unchanged."
)
TGD_DROPOUT_PROMPT = (
    "\nWe have introduced a dropout mechanism. The <DROPOUT>'{sentences}'</DROPOUT> in the original variable need to remain unchanged for this optimize step."
)

TGD_REGU_PROMPT = (
    "\nPlease simplify the overly complex and lengthy sentences in the variable. Ensure the output is concise, easy to understand, and suitable for a general audience. "
    "\nIf you are certain that a particular sentence in the variable has no impact on the overall meaning or purpose or has a negative effect, please delete that sentence."
)
TGD_TRAIN_ACC_MEM = (
    "\nYou can learn valuable insights by comparing the good and bad variables from past data. On the training set, the better-performing variables are <BETTER_VAR>{best_var}</BETTER_VAR>, while the poorer-performing variables are <POOR_VAR>{worst_var}</POOR_VAR>. To improve your model, focus on adopting the unique features that contribute to the success of the better variables and eliminate the unique features associated with the poorer variables. This approach will help enhance performance and avoid repeating past mistakes."
)
TGD_TRAIN_BEST_ACC_MEM = (
    "\nYou can gain valuable insights by analyzing high-performing variables from historical data. On the training set, the top-performing variables are <BETTER_VAR>{best_var}</BETTER_VAR>."
)

TGD_MOMENTUM_ = (
    "\nHere is the historical feedback on this variable:\n\n"
    "<PAST_FEEDBACK>{history}</PAST_FEEDBACK>\n\n"
    "Please analyze the main trends and patterns in the feedback across different iterations. If the feedback consistently points to similar issues or suggests insufficient modifications, it indicates that the changes made to the variable are not substantial enough."
    "In such cases, please propose more significant and impactful adjustments to the variable to better address the feedback and improve its performance.\n\n"
)
TGD_LARGER_G = (
    "In order to break away from convention, discover more creative solutions, and explore limitless possibilities, please boldly unleash your imagination based on feedback and make transformative modifications to the previous variables. Do not be confined by existing forms; courageously break the mold and experiment with entirely new combinations and ideas, as this may spark unexpected and groundbreaking outcomes."
)
def count_words(s: str) -> int:

    words = s.split()

    return len(words)
def count_sentences(text):
    import re
    sentences = re.split(r'[.!?]+', text)

    sentences = [s for s in sentences if s.strip()]
    
    return sentences
def dropout_select(ori_var:str = "",
                   dropout:float = 0,
                   ):
    sentences = count_sentences(ori_var)
    import math
    import random
    select_num = int(math.ceil(len(sentences) * dropout))
    selected_sentences = random.sample(sentences, select_num)
    result = ';'.join(selected_sentences)
    return result

def construct_tgd_promptV2(do_momentum: bool = False,
                         do_constrained: bool = False,
                         do_in_context_examples: bool = False,
                         lr: int = 0,
                         dropout: float = 0,
                         weight_decay = 0,
                         origin_var: str = "",
                         train_acc_mem = None,
                         train_acc_mem_2 = None,
                         gradient_memory_2 = None,
                         **optimizer_kwargs):

    if isinstance(optimizer_kwargs["variable_grad"], str):
        multipart=False
        prompt = TGD_PROMPT_PREFIX.format(**optimizer_kwargs)
        
    else:
        gradient_context = optimizer_kwargs["variable_grad"]
        gradient_context = [TGD_MULTIPART_PROMPT_INIT.format(**optimizer_kwargs)] + gradient_context
        multipart=True
        prompt = TGD_MULTIPART_PROMPT_PREFIX.format(**optimizer_kwargs)
           
    if do_momentum:
        prompt += MOMENTUM_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_constrained:
        prompt += CONSTRAINT_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_in_context_examples:
        prompt += IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(**optimizer_kwargs)
    if train_acc_mem_2!=None:
        if len(train_acc_mem_2)>1:
            k = min(len(train_acc_mem_2),4)
            historical_steps = ''
            for i in range(k):
                steps_index = i-k+1
                step_index_true = len(train_acc_mem_2)+steps_index -1
                _,prompt_ = train_acc_mem_2[step_index_true]
                if(i == k - 1):
                    historical_steps += f'->; step t(Current step):{prompt_}'
                else:
                    historical_steps += f'->; step t({steps_index}):{prompt_}'
            print(f"\nhistory:{historical_steps}")
            var = {
                'history':str(historical_steps),
            }

    if train_acc_mem != None:
        worst_v = ""
        best_v = ""
        if len(train_acc_mem)>=2:
            w_acc,worst_v = train_acc_mem[-1]
            b_acc,best_v = train_acc_mem[0]
            if worst_v!="" and b_acc-w_acc>=0.15:
                format_b_w = {
                'best_var':best_v,
                'worst_var':worst_v,
                }
                prompt += TGD_TRAIN_ACC_MEM.format(**format_b_w)
            else:
                format_b_w = {
                'best_var':best_v,
                }
                prompt += TGD_TRAIN_BEST_ACC_MEM.format(**format_b_w)
        elif len(train_acc_mem)>=4:

            mid_index = len(train_acc_mem)//2
            bests = train_acc_mem[:mid_index]
            n = len(bests)
            decreasing_weights = [n - i for i in range(n)]
            print(f"\ndecreasing_weights:{decreasing_weights}")
            sampled_elements = random.choices(bests, weights=decreasing_weights, k=2)
            best_v = ""
            step = 0
            min_best = 1
            for i in sampled_elements:
                step+=1
                b_acc,best = i
                best_v = best_v +f"\n{str(step)}." + best
                if b_acc<min_best:
                    min_best = b_acc
                break
            print(f"\nbest_v:{best_v}")
            worsts = train_acc_mem[mid_index:]
            n = len(worsts) 
            increasing_weights = [i + 1 for i in range(n)]
            print(f"\nincreasing_weights:{increasing_weights}")
            sampled_elements = random.choices(worsts, weights=increasing_weights, k=2)
            worst_v = ""
            step = 0

            for i in sampled_elements:
                
                w_acc,worst = i
                if min_best-w_acc>=0.05:
                    step+=1
                    worst_v = worst_v +f"\n{str(step)}." + worst
                if worst_v!="":
                    break
            print(f"\nworst_v:{worst_v}")
            if worst_v!="":
                format_b_w = {
                'best_var':best_v,
                'worst_var':worst_v,
                }
                prompt += TGD_TRAIN_ACC_MEM.format(**format_b_w)
            else:
                format_b_w = {
                'best_var':best_v,
                }
                prompt += TGD_TRAIN_BEST_ACC_MEM.format(**format_b_w)

    if gradient_memory_2!=None:
        if len(gradient_memory_2)>0:
            k = min(len(gradient_memory_2),3)
            historical_steps = ''
            for i in range(k):
                steps_index = i-k+1
                step_index_true = len(gradient_memory_2)+steps_index -1
                prompt_ = gradient_memory_2[step_index_true]
                if(i == k - 1):
                    historical_steps += f'\n<Histroy Begin>{prompt_}<End>'
                else:
                    historical_steps += f'\n<History Begin>{prompt_}<End>'
            var = {
                'history':str(historical_steps),
            }
            prompt += TGD_MOMENTUM_.format(**var)


        
    if dropout > 0:
        select_var = dropout_select(ori_var=origin_var,dropout=dropout)
        dropout_information = {
            "sentences" : select_var
        }
        prompt += TGD_DROPOUT_PROMPT.format(**dropout_information)
    if weight_decay > 0 and len(train_acc_mem)>=16:
        prompt += TGD_REGU_PROMPT
    if lr > 0:
        if lr<=4:
            prompt += TGD_LR_PROMPT_V0.format(**optimizer_kwargs)
        elif train_acc_mem!=None:
            #BIGGSM :
            # if len(train_acc_mem)>=4  and lr>20:
            # MGSM :
            if len(train_acc_mem)>=4  and lr>10:
                prompt += TGD_LARGER_G
    prompt += TGD_PROMPT_SUFFIX.format(**optimizer_kwargs)

    if not multipart:
        return prompt
    
    else:
        return gradient_context + [prompt]


# This is how we save gradients to the variable.
GRADIENT_TEMPLATE = (
    "Here is a conversation:\n\n<CONVERSATION>{context}</CONVERSATION>\n\n"
    "This conversation is potentially part of a larger system. The output is used as {response_desc}\n\n"
    "Here is the feedback we got for {variable_desc} in the conversation:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)
GRADIENT_MULTIPART_TEMPLATE = (
    "Above is a conversation with a language model.\n"
    "This conversation is potentially part of a larger system. The output is used as {response_desc}\n\n"
    "Here is the feedback we got for {variable_desc} in the conversation:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)
