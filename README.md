
# MER
Nonlinear Monte Carlo Method for Imbalanced Data Learning

Nonlinear Monte Carlo Method is the application of nonlinear expectation (Peng, 2005). In machine learning, we substitute the mean value of loss function with the maximum value of subgroup mean loss, which follows the assumption of sublinear expectation. We call it Maximum Empirical Risk (MER).

****

Dev requirement:

```
tensorflow-gpu-1.13.1
cvxopt-1.2.5
numpy-1.15.0
pandas-0.25.2
```

#### Toydata:
```
python MER_for_toydata.py
```

#### CelebA:  
1, DNN+MER:


```
python 10-nonlinearized-dual.py --num_of_group 4 --num_minor 2 --train_subgroup_batch 100
``` 
Here,  
--num_of_group : total subgroup number $N$ ;    
--num_minor    : number of minor-class subgroup $c$ ;      
--train_subgroup_batch : number of data in each subgroup $s_j$ ;   



2, ResNet+MER:

```
python nonlinearized-dual-ResNet.py --num_of_group 4 --num_minor 2 --train_subgroup_batch 100
```

The label now is "eyeglasses" in CelebA dataset, which can be changed in data_process.py. 
