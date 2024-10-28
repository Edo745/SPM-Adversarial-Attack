## Overview
This repository contains the implementation of adversarial attacks on the MNIST dataset using the Sequential Penalty Method (SPM). The approach focuses on generating adversarial examples that force a classifier to misclassify an image by introducing minimal perturbations, ensuring the changes are imperceptible to humans while still deceiving the machine learning model.

<p align="center">
<img src="https://github.com/user-attachments/assets/674840d8-3491-450b-a975-b5ffb7049123" width="800"/> 
</p> 

## Introduction

Adversarial attacks are techniques used to deceive machine learning models by introducing small, carefully crafted perturbations to the input data. These perturbations are generally imperceptible to humans but can cause a model to make incorrect predictions. In this project, we implement a **Sequential Penalty Method** to carry out adversarial attacks on images from the MNIST dataset.

The **Sequential Penalty Method (SPM)** transforms a constrained optimization problem into a series of unconstrained subproblems by introducing a penalty function. At each iteration, the penalty parameter is increased to enforce constraint satisfaction while minimizing the objective function. This method ensures that the adversarial perturbation remains small while still achieving misclassification.

## Model and Dataset

The adversarial attacks are performed on images from the MNIST dataset. The model used is a Small Convolutional Neural Network (SmallCNN) trained for 10 epochs, achieving 99% accuracy on non-perturbed images. The architecture and training details can be found in the models/smallcnn.py file.
<p align="center">
  <img src="https://github.com/user-attachments/assets/203e432f-21fe-4cb5-9728-374c5338b3cf" width="300" alt="SmallCNN" title="SmallCNN"/> 
</p> 

The MNIST dataset is a collection of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels, split into 60,000 training images and 10,000 test images.

The SmallCNN model consists of two convolutional layers followed by ReLU activations, then a max pooling layer, followed by two more convolutional layers with ReLU activations, another max pooling layer, and finally three fully connected layers for classification.

The model was trained using the Adam optimizer (learning rate of 0.001) with cross-entropy loss, achieving high accuracy on non-perturbed images.
<p align="center">
  <img src="https://github.com/user-attachments/assets/c468d977-696a-49e7-9293-c64fdbbf16df" width="400" alt="Train Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/d940f9bf-499a-4d4d-b083-743d27239aaf" width="400" alt="Train Accuracy" title="Train Accuracy"/>
</p> 

## Problem Formulation
The constrained optimization problem is defined as follows:

$$\min f(x) \quad \text{s.t.} \quad g(x) \leq 0$$

Where:

- $f(x) = \frac{1}{2} \||x - x_k\||^2$ : This objective function minimizes the difference between the perturbed image $x_k$ and the original image $x$.

- $g(x) = (I_k - 1^T_k \cdot e_j) \cdot C(x_k) $: This inequality constraint ensures that the classifier misclassifies the perturbed image into the target class $j$. Here, $C(x_k)$ represents the classifier's prediction output, and $e_j$ is the target class vector. Inserire le matrici e il vettore tipo esplicitare il prodotto.


## Sequential Penalty Methods
The Sequential Penalty Method consists of starting from the constrained problem:

$$\min f(x) \quad \text{s.t.} \quad g(x) \leq 0,$$

and solving a sequence of unconstrained subproblems of the form:

$$F(x) = f(x) + \tau \cdot P(x)$$
where $\tau$ is a penalty parameter that is increased at each iteration and $P(x) = \max \\{0, g(x)\\}^2$.

As shown in the figure below, the higher the value of $\tau$, the greater the tendency of the solution of $F(x)$ to coincide with the solution of the original problem $f(x)$.
<p align="center">
  <img src="https://github.com/user-attachments/assets/4fe9ed38-2760-46b2-b1d9-d077e4e5a766" width="400" alt="From 2 to 6, tau0=1, rho=1.2" title="From 2 to 6, tau0=1, rho=1.5"/> 
</p> 


In our specific implementation, we have the following key components:

1. **Perturbation**: Is what we're trying to optimize to create the adversarial example. It is inizialized as a tensor of zeros with the same shape as the input image. We want to compute gradients with respect to this variable during optimization that's why `requires_grad=True`.
   ```python
   perturbation = torch.zeros_like(input_image, requires_grad=True)
   ```

2. **Optimizer**: We use PyTorch's Adam optimizer to update the perturbation.
   ```python
   optimizer = torch.optim.Adam([perturbation], lr=0.01)
   ```
 

3. **Perturbed Input Image**: This line creates the perturbed image by adding the perturbation to the original image.
   ```python
   input_image_perturbed = input_image + perturbation
   ```
   

4. **Objective Function**: it measures the magnitude of the perturbation.
   ```python
   f = 0.5 * torch.norm(perturbation) ** 2
   ```
 

5. **Constraint Function**:
   ```python
   output = model(input_image_perturbed) # Gets the model's output for the perturbed image.
   IK = torch.eye(10, device=input_image.device) # Creates a 10x10 identity matrix (for the 10 MNIST classes).
   one_K = torch.ones(10, device=input_image.device) # Creates a vector of 10 ones.
   ej = torch.zeros(10, device=input_image.device) 
   ej[target_label] = 1 # Creates a one-hot vector for the target class.
   g = (IK - torch.outer(one_K, ej)) @ output.squeeze() # constraint that ensures that the output for the target class is greater than the outputs for all other classes.
   ```
    

6. **Unconstrained Subproblem**: The complete loss function that combines the objective function (`f`) and the penalty term.
   ```python
   loss = f + tau * torch.sum(torch.max(g) ** 2)
   ```

7. **Penalty Parameter Update**: After each iteration, the penalty parameter `tau` is increased by multiplying it by `rho`
   ```python
   tau = tau * rho
   ```

These components work together to iteratively refine the perturbation, balancing the goals of minimizing the perturbation size and achieving misclassification to the target label. The algorithm continues until either the misclassification is achieved or the maximum number of iterations is reached.

Using this SPM algorithm we are able to solve the original constrained problem:

$$\min \frac{1}{2} ||x-x_k||^2 \quad s.t. \quad (I_k - 1_K^T\cdot ej )\cdot C(x_k) \leq 0$$

by solving unconstrained subproblems:

$$\min \frac{1}{2} ||x-x_k||^2 + \tau \cdot \max\\{0, (I_k - 1_K^T\cdot ej )\cdot C(x_k)\\}^2$$

### Deep Dive Into SMP
Let's have a look at the second iteration (target_label=8).

1. Objective function:

$$f=0.8900532722473145$$

3. Model output:
   
$$
\left (
\begin{matrix}
-4.0147 & 13.6417 & -0.8090 & -4.4129 & 0.6421 & -1.7495 & 0.3227 & -0.3685 & -0.1891 & -3.0631
\end{matrix}
\right )
$$

4. Identical Matrix IK
   
$$I_K=\left ( \begin{matrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{matrix} \right )$$

6. All ones vector:
 
$$one_K = \left (
\begin{matrix}
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1
\end{matrix}
\right )$$

7. Canonical vector:
   
$$e_j=\left (
\begin{matrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
\end{matrix}
\right )$$

8.

$$
(IK-1_k^T*e_j)=\left (
\begin{matrix}
1  & 0  & 0  & 0  & 0  & 0  & 0  & 0  & -1 & 0 \\
0  & 1  & 0  & 0  & 0  & 0  & 0  & 0  & -1 & 0 \\
0  & 0  & 1  & 0  & 0  & 0  & 0  & 0  & -1 & 0 \\
0  & 0  & 0  & 1  & 0  & 0  & 0  & 0  & -1 & 0 \\
0  & 0  & 0  & 0  & 1  & 0  & 0  & 0  & -1 & 0 \\
0  & 0  & 0  & 0  & 0  & 1  & 0  & 0  & -1 & 0 \\
0  & 0  & 0  & 0  & 0  & 0  & 1  & 0  & -1 & 0 \\
0  & 0  & 0  & 0  & 0  & 0  & 0  & 1  & -1 & 0 \\
0  & 0  & 0  & 0  & 0  & 0  & 0  & 0  & 0  & 0 \\
0  & 0  & 0  & 0  & 0  & 0  & 0  & 0  & -1 & 1
\end{matrix}
\right )
$$


10. Constraint function g:
   
$$g = \left (
\begin{matrix}
-3.8256 & 13.8309 & -0.6199 & -4.2238 & 0.8312 & -1.5604 & 0.5119 & -0.1794 & 0.0000 & -2.8740
\end{matrix}
\right )$$


## Experiments
Several experiments were conducted to assess the effectiveness of the Sequential Penalty Method under different conditions:

1. **Different Images**: We experimented with attacking different images from the MNIST dataset, varying the true label and target label combinations. The attack was consistently effective across different samples, although certain digit pairs required more iterations due to visual similarities between digits.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb343c79-1863-4e75-94b5-cc97d27046e01" width="800" alt="From 4 to 9" title="From 4 to 9"/> 
</p> 
<p align="center">
  <img src="https://github.com/user-attachments/assets/351b730e-c42c-4fc2-a91e-201fc00a5d2f" width="800" alt="From 3 to 0" title="From 3 to 0"/> 
</p> 

| input  | target  | iterations |
|------|------|------------|
| 4  | 9  | 41       |
| 3  | 0 | 61        |


2. **Different tau Values**: We tested different values for the hyperparameter tau to analyze its impact on the perturbation magnitude and misclassification rate. 
<p align="center">
  <img src="https://github.com/user-attachments/assets/c8987a5e-6e5e-4347-8aa3-6e88abe2db76" width="800" alt="From 8 to 1, tau0=1, rho=1.1" title="From 8 to 1, tau0=1, rho=1.2"/> 
</p> 
<p align="center">
  <img src="https://github.com/user-attachments/assets/c8987a5e-6e5e-4347-8aa3-6e88abe2db76" width="800" alt="From 8 to 1, tau0=1, rho=1.5" title="From 8 to 1, tau0=1, rho=1.2"/> 
</p> 

| tau  | rho  | iterations |
|------|------|------------|
| 1.0  | 1.1  | 51       |
| 1.0  | 1.5  | 50        |

By increasing rho (the tau increment term), we expected a significant reduction in the number of iterations required to corrupt the model. In reality, we noticed that even a slight increase in the value of rho leads to numerical instabilities that prevent the algorithm from converging. In this experiment we can observe how a rho larger than 0.4 only saves one iteration. The result is not very significant.


3. **Different Learning Rates**:  We tested different values for the learning rate of the Adam optimizer.
 * **lr=0.1**
  <p align="center">
  <img src="https://github.com/user-attachments/assets/f402117f-3441-4310-a5bf-1c1b94d5875f" width="800" alt="From 8 to 1, tau0=1, rho=1.1" title="From 8 to 1, tau0=1, rho=1.5, lr=0.1"/> 
</p> 

 * **lr=0.01**
<p align="center">
<img src="https://github.com/user-attachments/assets/ecf9fcbc-a102-4ce9-8c75-7e539cefb623" width="800" alt="From 8 to 1, tau0=1, rho=1.5" title="From 8 to 1, tau0=1, rho=1.5, lr=0.01"/> 
</p> 

| tau  | rho  | lr | iterations |
|------|------|----|------------|
| 1.0  | 1.5  | 0.1 |      9      |
| 1.0  | 1.5  | 0.01 |      52      |

The use of a larger step drastically reduces the number of iterations needed to corrupt the model.

4. **Different Optimizer**: we compared Adam and SGD:
* **ADAM**
<p align="center">
<img src="https://github.com/user-attachments/assets/ecf9fcbc-a102-4ce9-8c75-7e539cefb623" width="800" alt="From 8 to 1, tau0=1, rho=1.5" title="From 8 to 1, tau0=1, rho=1.5, lr=0.01"/> 
</p> 

  
* **SGD**
<p align="center">
<img src="https://github.com/user-attachments/assets/86fe880d-21de-4d9a-863b-d5ab8644f06c" width="800" alt="From 8 to 1, SGD, lr=0.01" title="From 8 to 1, SGD, lr=0.01"/> 
</p> 

4. **Different Dataset**: We tested the algorithm on ImageNet as well to observe its behavior with higher resolution images (224,224,3).
*   
<p align="center">
<img src="https://github.com/user-attachments/assets/d79fceaf-169d-4f96-97bd-a97979bc65fa" width="800"/> 
</p> 

## Usage
### Requirements

Requirements

To run this project, you will need to install all the requirements with following command.

```bash
pip install -r requirements.txt
```

### Training the Model

To train the SmallCNN model on the MNIST dataset, use the following command:

```bash
python train-cnn.py --epochs 10
```

This will train the model for 10 epochs and save the weights in the `checkpoints/` directory.

### Running the Adversarial Attacks

To run the adversarial attacks using the Sequential Penalty Method on MNIST/ImageNet, execute the following command:

```bash
python spm-attack-mnist.py --true-label 3 --target-label 7 --tau 1.0 --rho 1.5 --Niter 100
```
```bash
python spm-attack-imagenet.py --target-label 134 --image-path data/img/imagenet-sample-images-master/n01774750_tarantula.JPEG
```

The results are saved in the `results/mnist` or `results/imagenet` directory.



## References

- Beuzeville et al., "Adversarial Attacks via Sequential Quadratic Programming", 2024.  
- Rony et al., "Sequential Penalty Method for Adversarial Attacks", 2021. 
- Grippo et al., "Sequential Penalty Methods: Theory and Applications", 2023. 


