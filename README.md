# DLPO: Towards a Robust, Efficient, and Generalizable Prompt Optimization Framework from a Deep-Learning Perspective

Existing prompt optimization frameworks still face critical challenges in robustness, efficiency, and generalization.
To systematically address these challenges, we first conduct an empirical analysis to identify the limitations of current reflection-based prompt optimization paradigm.
Building on these insights, we propose 7 innovative approaches inspired by traditional deep learning paradigms for prompt optimization **DLPO**, seamlessly integrating these concepts into text-based gradient optimization. 
Through these advancements, we progressively tackle the aforementioned challenges and validate our methods through extensive experimentation.
We hope our study not only provides valuable guidance for future research but also offers a comprehensive understanding of the challenges and potential solutions in prompt optimization.

(https://arxiv.org/abs/2503.13413)

## Setup and Configuration

### API Configuration

Before running the project, you need to configure your API credentials. Follow these steps:

1. Open the file `prompt_opt.py`.
2. Set your API key and base URL in the following section:

```python
import os

os.environ['OPENAI_API_KEY'] = 'your_api_key_here'
os.environ['OPENAI_BASE_URL'] = 'your_base_url_here'
```

Replace `your_api_key_here` and `your_base_url_here` with your actual API credentials.

### Running the Script

Once the API configuration is complete, execute the script using the following command:

```bash
python prompt_opt.py
```

## Hyperparameter Settings: 

Under the current hyperparameters, the model only uses TSA and is trained on the BigGSM dataset with 200 training samples. If you wish to use TSA, TCL, TMnt, and TRegu, you simply need to set the hyperparameters --annealing, --train_acc_contras, --momentum, and --dy_lr to 1. The parameter --tsa_tem sets the initial temperature for TSA, while --tsa_tem_decay controls the decay rate of the temperature. To encourage more exploration, you should increase the value of --tsa_tem and decrease the value of --tsa_tem_decay.

## Acknowledgments

We are looking forward to someone who can further improve our work!

This project is built upon the work of **TextGrad** ([GitHub Repository](https://github.com/zou-group/textgrad)). Thanks for their excellent work.

If you find our work inspiring, please cite:

```bibtex
@article{peng2025dlpo,
  title={DLPO: Towards a Robust, Efficient, and Generalizable Prompt Optimization Framework from a Deep-Learning Perspective},
  author={Peng, Dengyun and Zhou, Yuhang and Chen, Qiguang and Liu, Jinhao and Chen, Jingjing and Qin, Libo},
  journal={arXiv preprint arXiv:2503.13413},
  year={2025}
}
