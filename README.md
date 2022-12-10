Code repository for model weights robustifying research

Base on the official code from AWP
Paper (NeurIPS 2020 "[Adversarial Weight Perturbation Helps Robust Generalization](https://arxiv.org/pdf/2004.05884.pdf)").
Github link https://github.com/csdongxian/AWP.

## Cifar-10 Dataset
#### Setting 1
Training attack pgd-10 with L2 norm, epsilon 128/255, and step-size (pgd-alpha) 15/255. 
Example command as follows
`python AT-AWP/train_cifar10.py --norm l_2 --pgd-alpha 15 --epsilon 128`

#### Setting 2
Training attack pgd-10 with Linf norm, epsilon 8/255, and step-size (pgd-alpha) 2/255. 
Example command as follows
`python AT-AWP/train_cifar10.py --norm l_inf --pgd-alpha 2 --epsilon 8`
