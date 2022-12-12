Code repository for model weights robustifying research

Base on the official code from AWP
Paper (NeurIPS 2020 "[Adversarial Weight Perturbation Helps Robust Generalization](https://arxiv.org/pdf/2004.05884.pdf)").
Github link https://github.com/csdongxian/AWP.

# Baseline Experiments
Some notes:
- Train:
  - Specify the output path before each realization with arg `--fname` to avoid confounding.
  - If possible, run multiple different realizations (arg `--seed`) for certain configuration, to measure performance in terms of mean and std. Results would be output into the folder `f'{fname}/{seed}'`.
  - Default backbone architecture is **PreActResNet18**.
  - Model with **best train robust loss** would be newly saved as candidate.
- Test:
  - Default test process would be conducted after training via *attack pgd-20* with target metrics **natural accuracy** and **robust accuracy**.
  - Autoattack could be launched via ```python AT-AWP/eval_autoattack.py``` for both training and test sets, with standard attacks **apgd-ce**, **apgd-t**, **fab-t**, and **square**. Note that args `--norm` and `--epsilon` shall be kept in consistency with that in training. For instance, in setting 1 of cifar-10, run ```python AT-AWP/eval_autoattack.py --data CIFAR10 --norm L2 --epsilon 128/255```.

## Cifar-10 Dataset

### Setting 1
Training attack pgd-10 with L2 norm, epsilon 128/255, and step-size (pgd-alpha) 15/255. 
Example command as follows
```python AT-AWP/train_cifar10.py --norm l_2 --pgd-alpha 15 --epsilon 128```

### Setting 2
Training attack pgd-10 with Linf norm, epsilon 8/255, and step-size (pgd-alpha) 2/255. 
Example command as follows
```python AT-AWP/train_cifar10.py --norm l_inf --pgd-alpha 2 --epsilon 8```

## SVHN Dataset

