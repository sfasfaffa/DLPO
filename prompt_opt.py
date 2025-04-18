import argparse
import concurrent
from dotenv import load_dotenv
load_dotenv(override=True)
import copy
from tqdm import tqdm


import numpy as np
import random
import os
import wandb
from textgrad.autograd import FormattedLLMCall
import textgrad as tg
from textgrad.tasks import load_task


os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_BASE_URL'] = ''
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
summary = None
epoch_summary = None
def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    # parser.add_argument("--task", type=str, default="BBH_object_counting", help="The task to evaluate the model on.")
    # parser.add_argument("--task", type=str, default="MGSM", help="The task to evaluate the model on.")
    # parser.add_argument("--task", type=str, default="GSM8K_DSPy", help="The task to evaluate the model on.")
    # parser.add_argument("--task", type=str, default="MATH", help="The task to evaluate the model on.")
    parser.add_argument("--task", type=str, default="BIG_GSM", help="The task to evaluate the model on.")

    parser.add_argument("--evaluation_engine", type=str, default="gpt-4o", help="The API to use for evaluation.")
    # parser.add_argument("--evaluation_engine", type=str, default="HiSpeed/DeepSeek-R1", help="The API to use for evaluation.")

    parser.add_argument("--test_engine", type=str, default="gpt-4o-mini", help="The API to use for evaluation.")
    parser.add_argument("--batch_size", type=int, default=5, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=1, help="The maximum number of epochs to train for.")
    # parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1001)
    # parser.add_argument("--seed", type=int, default=13125)
    parser.add_argument("--run_validation", action="store_true", help="Whether to run validation or not.")
    parser.add_argument("--do_not_run_larger_model", action="store_true", help="Whether to run the larger model or not.")
    parser.add_argument("--num_threads", type=int, default=60, help="The number of threads to use for evaluation.")
    
    parser.add_argument("--lr", type=int, default=0, help="learning rate.")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout.")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay.")
    parser.add_argument("--wandb",  type=int, default=0)
    parser.add_argument("--train_acc_contras", type=int, default=0, help="contractive learning")
    parser.add_argument("--annealing", type=int, default=1, help="annealing")
    parser.add_argument("--momentum", type=int, default=0, help="text momentum")
    parser.add_argument("--dy_lr", type=int, default=0, help="learning rate decay")
    parser.add_argument("--tsa_tem", type=int, default=64, help="in context learning")
    parser.add_argument("--tsa_tem_decay", type=int, default=2, help="in context learning")
    return parser.parse_args()


args = config()

def eval_sample(item, eval_fn, model,summary=None):
    x, y = item
    if summary!= None:
        x = summary+str(x)+"</question>"
    if args.task == "MATH":
        format_confirm = " Please enclose the final answer in \\boxed{} at the end. This is a required response format."
    else:
        format_confirm = "Your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
    x = tg.Variable(str(x)+format_confirm, requires_grad=False, role_description="query to the language model")
    if args.task == "MATH":
        y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    else:
        y = tg.Variable(int(y), requires_grad=False, role_description="correct answer for the query")

    
    try:
        response = model(x)
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:

        return 0
    

def eval_dataset(test_set, eval_fn, model, max_samples: int=None,summary:str=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            
            future = executor.submit(eval_sample, sample, eval_fn, model,summary)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list

def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)

def main():
    import copy

    use_wandb = bool(args.wandb)
    if use_wandb:
        wandb.init(
            project="text_grad_v2",
            config=vars(args),
            name=f"{args.task}_{args.test_engine}_{str(args.seed)}_v2"
        )
    
    set_seed(args.seed)
    llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
    llm_api_test = tg.get_engine(engine_name=args.test_engine)
    tg.set_backward_engine(llm_api_eval, override=True)

    # Load the data and the evaluation function
    train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=llm_api_eval)
    STARTING_SYSTEM_PROMPT_GSM = "You will provide a clear and concise answer to the reasoning question, directly addressing the key points without unnecessary elaboration."
    STARTING_SYSTEM_PROMPT_BBH = "You will answer a reasoning question"
    if "GSM" in args.task:
        STARTING_SYSTEM_PROMPT = STARTING_SYSTEM_PROMPT_GSM
    elif args.task == "MATH":
        STARTING_SYSTEM_PROMPT = ""
    else:
        STARTING_SYSTEM_PROMPT = STARTING_SYSTEM_PROMPT_BBH
    train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                                requires_grad=True, 
                                role_description="system prompt to the language model")
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                                requires_grad=True,
                                role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
    model = tg.BlackboxLLM(llm_api_test, system_prompt)

    optimizer = tg.TGD_OPT(engine=llm_api_eval, parameters=[system_prompt],learning_rate=args.lr,
                           weight_decay=args.weight_decay, dropout=args.dropout,momentum=args.momentum,acc_mem=args.train_acc_contras,dy_lr=args.dy_lr,tsa_tem = args.tsa_tem,tsa_tem_decay = args.tsa_tem_decay)

    print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
    results = {"val_acc": [], "prompt": [], "validation_acc": []}
    print()
    print("step 0:")
    print("system prompt:")
    print(STARTING_SYSTEM_PROMPT)
    print("val_set:")
    val_acc = eval_dataset(val_set, eval_fn, model)
    correct_count = sum(val_acc)
    total_count = len(val_acc)    
    val_accuracy = correct_count / total_count

    

    train_acc = eval_dataset(train_set, eval_fn, model)
    correct_count = sum(train_acc)  
    total_count = len(train_acc)   
    train_accuracy = correct_count / total_count 
    print(f"\nval_acc:{val_accuracy}, train_acc:{train_accuracy}")
    results["val_acc"].append([val_accuracy,system_prompt.get_value()])
    results["prompt"].append([system_prompt.get_value()])

    if use_wandb:
        wandb.log({
            "val_accuracy": val_accuracy,
            "train_accuracy":train_accuracy,
            "loss":(1-train_accuracy),
        })

    system_prompt_past = copy.copy(system_prompt)
    for epoch in range(args.max_epochs):
        
        for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            optimizer.zero_grad()
            losses = []
            for (x, y) in zip(batch_x, batch_y):
                new_x = x
                # if args.batch_norm == 1:
                #     new_x = summary+x +"</question>"
                new_x = tg.Variable(str(new_x), requires_grad=False, role_description="query to the language model")
                y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
                response = model(new_x)

                try:
                    eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
                except:
                    eval_output_variable = eval_fn([new_x, y, response])
                losses.append(eval_output_variable)

            
            total_loss = tg.aggregate(losses)
            total_loss.backward()

            if args.annealing == 1:
                optimizer.annealing_step(eval_dataset,train_set, eval_fn, model,epoch_summary)
            else:
                optimizer.step()

            print(f"\nlearning rate:{optimizer.learning_rate}")
            if args.run_validation:
                run_validation_revert(system_prompt, results, model, eval_fn, val_set)
            
            if (args.batch_size*(steps+1)%1 ==0) or (args.batch_size*(steps+1)>=len(train_set)):
                print("\nsys prompt: ", system_prompt)
                print("\nsummary:",epoch_summary)
                """计算val acc"""
                # val_acc = eval_dataset(val_set, eval_fn, model,summary=epoch_summary)
                val_acc = eval_dataset(val_set, eval_fn, model,summary=None)
                correct_count = sum(val_acc)  
                total_count = len(val_acc)     
                val_accuracy = correct_count / total_count

                """计算train acc"""

                train_acc = eval_dataset(train_set, eval_fn, model,summary=None)
                correct_count = sum(train_acc)
                total_count = len(train_acc)     
                train_accuracy = correct_count / total_count

                results["val_acc"].append([val_accuracy,system_prompt.get_value()])
                results["prompt"].append([system_prompt.get_value()])
                print(f"\nval_acc:{val_accuracy}, train_acc:{train_accuracy}")
                
                if system_prompt_past != system_prompt:
                    print(f"past:{system_prompt_past}")
                    print(f"new:{system_prompt}")
                    optimizer.add_train_acc_mem(train_accuracy,copy.copy(system_prompt.get_value()))
                    system_prompt_past = copy.copy(system_prompt)
                if use_wandb:
                    wandb.log({
                        "val_accuracy": val_accuracy,
                        "train_accuracy":train_accuracy,
                        "loss":(1-train_accuracy),
                    })
    import json
    import os
    from datetime import datetime

    current_date_time = datetime.now()

    current_date_str = current_date_time.strftime("%m-%d") 

    # Create the directory if it doesn't exist
    os.makedirs("./figures", exist_ok=True)
    # Create and write to the file
    with open(f"./figures/{current_date_str}_results_{args.task}_{args.test_engine}_{str(args.batch_size)}_{str(args.lr)}_{str(args.dropout)}_{str(args.weight_decay)}_sd={str(args.seed)}.json", "w") as f:
        json.dump(results, f)

    return f"./figures/{current_date_str}_results_{args.task}_{args.test_engine}_{str(args.batch_size)}_{str(args.lr)}_{str(args.dropout)}_{str(args.weight_decay)}_sd={str(args.seed)}.json",test_set,eval_fn
def test(prompt,test_set = None,eval_fn=None):
    set_seed(args.seed)
    llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
    llm_api_test = tg.get_engine(engine_name=args.test_engine)
    tg.set_backward_engine(llm_api_eval, override=True)
    # Load the data and the evaluation function
    if test_set == None or eval_fn == None:
        _, _, test_set, eval_fn = load_task(args.task, evaluation_api=llm_api_eval)
    # STARTING_SYSTEM_PROMPT = "You will answer a reasoning question. Think step by step."
    STARTING_SYSTEM_PROMPT = prompt
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                                requires_grad=True,
                                role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
    print(f'final sys prompt:{system_prompt.value}')
    model = tg.BlackboxLLM(llm_api_test, system_prompt)
    test_acc = eval_dataset(test_set, eval_fn, model,summary=epoch_summary)
    correct_count = sum(test_acc) 
    total_count = len(test_acc)   
    accuracy = correct_count / total_count  
    print(f"\nacc:{accuracy}")
    return accuracy
def test_all(path = None,test_set = None,eval_fn=None):
    print("\n\n\n\n************test start**********:\n\n\n\n")
    import json
    file_path = 'D:/DLearning/HIT_scir/tg_2/textgrad/figures/11-26_results_BBH_object_counting_gpt-3.5_6_0_0.0_0.0_sd=42.json'  
    if path!= None:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

    val_acc = data.get("val_acc", [])
    acc_max = 0
    acc_last = 0
    best_p = ""
    last_p = ""

    for val_accuracy, description in val_acc:
        accuracy = float(val_accuracy)
        if accuracy>acc_max:
            acc_max = accuracy
            best_p = description

        last_p = description
        acc_last =accuracy
    # print(f"Validation Accuracy: {acc_max}, Description: {best_p}")
    print(f"\nbest prompt:{best_p},\nacc:{acc_max}")
    best_acc = test(best_p,test_set,eval_fn)
    
    wandb.log({
        "best_acc":best_acc
    })
    print(f"\nlast prompt:{last_p},\nacc{acc_last}")
    last_acc = test(last_p,test_set,eval_fn)
    wandb.log({
        "last_acc":last_acc
    })

def test_2(prompt,name=None):
    set_seed(args.seed)
    llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
    llm_api_test = tg.get_engine(engine_name=args.test_engine)
    tg.set_backward_engine(llm_api_eval, override=True)
    # Load the data and the evaluation function
    _, _, test_set, eval_fn = load_task(name, evaluation_api=llm_api_eval)
    # STARTING_SYSTEM_PROMPT = "You will answer a reasoning question. Think step by step."
    STARTING_SYSTEM_PROMPT = prompt
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                                requires_grad=True,
                                role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
    print(f'final sys prompt:{system_prompt.value}')
    model = tg.BlackboxLLM(llm_api_test, system_prompt)
    test_acc = eval_dataset(test_set, eval_fn, model,summary=epoch_summary)
    correct_count = sum(test_acc) 
    total_count = len(test_acc)     
    accuracy = correct_count / total_count
    print(f"\nacc:{accuracy}")
    return accuracy
def test_all_2(path = None,name = None):
    print("\n\n\n\n************test start2**********:\n\n\n\n")
    import json
    if path!= None:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        file_path = 'D:/DLearning/HIT_scir/tg_2/textgrad/figures/11-26_results_BBH_object_counting_gpt-3.5_6_0_0.0_0.0_sd=42.json'  

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

    val_acc = data.get("val_acc", [])
    acc_max = 0
    acc_last = 0
    best_p = ""
    last_p = ""
    for val_accuracy, description in val_acc:
        accuracy = float(val_accuracy)
        if accuracy>acc_max:
            acc_max = accuracy
            best_p = description

        last_p = description
        acc_last =accuracy
    # print(f"Validation Accuracy: {acc_max}, Description: {best_p}")
    print(f"\nbest prompt:{best_p},\nacc:{acc_max}")
    best_acc = test_2(best_p,name=name)
    str_b_acc = f"best_acc_{name}"
    wandb.log({
        str_b_acc:best_acc
    })
    print(f"\nlast prompt:{last_p},\nacc{acc_last}")
    last_acc = test_2(last_p,name=name)
    str_l_acc = f"last_acc_{name}"
    wandb.log({
        str_l_acc:last_acc
    })
def get_path():
    from datetime import datetime
    current_date_time = datetime.now()

    current_date_str = current_date_time.strftime("%m-%d")
    return f"./figures/{current_date_str}_results_{args.task}_{args.test_engine}_{str(args.batch_size)}_{str(args.lr)}_{str(args.dropout)}_{str(args.weight_decay)}_sd={str(args.seed)}.json"

if __name__ == '__main__':


    path,test_set,eval_fn = main()

    test_all(path,test_set,eval_fn)
    """test your prompt on other dataset ?"""
    # test_all_2(path,"GSM8K_DSPy")
    # test_all_2(path,"BIG_GSM")
    # test_all_2(path,"BBH_object_counting")
    # args.task = "MATH"
    # test_all_2(path,"MATH")
