# Understanding Pytorch

In this repository, I experiment with pytorch modules and functions to understand them better. I found the official Pytorch documentation quite poor and difficult to understand what any function does. So, I have experimented with several of the pytorch functionalities on examples to understand them better.

I will keep updating this repository as I experiment with more functions.

## Table Of Content

- [Repository Structure](#repository-structure)
- [Useful Resources](#useful-resources)
- [Usage](#usage)

## Repository Structure

#### `tensor_manipulations/`

Contains experiments with different tensor manipulations functions in Pytorch.

- [`understanding_tensors_part_1.ipynb`](tensor_manipulations/understanding_tensors_part_1.ipynb)
    - What is a tensor?
    - How to create a tensor?
    - What is the concept of dimension for a tensor?
    - How to understand tensors of higher dimensions?

- [`understanding_tensors_part_2.ipynb`](tensor_manipulations/understanding_tensors_part_2.ipynb)
    - What is stride of a tensor?
    - How to access the underlying storage of a tensor?
    - What is the difference between `tensor.data_ptr()` and `tensor.storage().data_ptr()`?

- [`understanding_tensor_manipulations_part_1.ipynb`](tensor_manipulations/understanding_tensor_manipulations_part_1.ipynb)
    - How [`torch.unsqueeze`](https://www.google.com/url?q=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Fgenerated%2Ftorch.unsqueeze.html) works?
        - Adds an additional dimension to the existing tensor.
    - How [`torch.nn.functional.pad`](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad) works?
        - Adds padding to tensors.
    - How to perform slicing on tensors by regular indexing?
    - How [`torch.unbind`](https://pytorch.org/docs/stable/generated/torch.unbind.html#torch-unbind) works?

- [`understanding_tensor_manipulations_part_2.ipynb`](tensor_manipulations/understanding_tensor_manipulations_part_2.ipynb)
    - How [`torch.cat`](https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) works?
        - Concatenates tensors.
    - How [`torch.stack`](https://pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) works?
        - Stacks tensors -- I didn't understand this operation at higher dimensions.

- [`understanding_tensor_manipulations_part_3.ipynb`](tensor_manipulations/understanding_tensor_manipulations_part_3.ipynb)
    - How [`torch.mean`](https://pytorch.org/docs/stable/generated/torch.mean.html#torch.mean) works?
        - Computes the mean along the specified dimension.
    - How [`torch.topk`](https://pytorch.org/docs/stable/generated/torch.topk.html#torch.topk) works?
        - Finds the top k values and the corresponding indices.

- [`understanding_tensor_manipulations_part_4.ipynb`](tensor_manipulations/understanding_tensor_manipulations_part_4.ipynb)
    - How [`torch.matmul`](https://pytorch.org/docs/stable/generated/torch.matmul.html#torch-matmul) works?
        - Computes matrix multiplication.
    - How [`broadcasting`](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics) works in pytorch?
        - Explains how broadcasting works in pytorch.
        - This is one of the most confusing and prone to error concept. Spend time to understand this effectively.
    - How [`torch.scatter`](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch-tensor-scatter) works?
        - Scatter the values from source tensor to a target tensor.

- [`understanding_tensor_manipulations_part_5.ipynb`](tensor_manipulations/understanding_tensor_manipulations_part_5.ipynb)
    - How [`torch.zeros`](https://pytorch.org/docs/stable/generated/torch.zeros.html#torch-zeros) works?
        - Creates a tensor of all zeros.
    - How [`torch.arange`](https://pytorch.org/docs/stable/generated/torch.arange.html#torch-arange) works?
        - Creates a tensor from a range.
    - How [`torch.exp`](https://pytorch.org/docs/stable/generated/torch.exp.html#torch-exp) works?
        - Performs exponentiation on the elements of the tensor.
    - How [`torch.ones`](https://pytorch.org/docs/stable/generated/torch.ones.html#torch-ones) works?
        - Creates a tensor of all ones.

- [`understanding_tensor_manipulation_part_6.ipynb`](tensor_manipulations/understanding_tensor_manipulations_part_6.ipynb)
    - How [`torch.reshape`](https://pytorch.org/docs/stable/generated/torch.reshape.html#torch-reshape) works?
        - Updates the shape of a tensor along with underlying storage.
    - How [`torch.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch-tensor-view) works?
        - Updates the shape of a tensor while keeping the storage same.
    - How [`torch.transpose`](https://pytorch.org/docs/stable/generated/torch.transpose.html#torch-transpose) works?
        - Transposes the tensor.
    - How [`torch.tensor.repeat`](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html#torch-tensor-repeat) works?
        - Repeats the tensors along specified dimensions, given number of times.

- [`understanding_tensor_manipulations_part_7.ipynb`](tensor_manipulations/understanding_tensor_manipulations_part_7.ipynb)
    - How [`torch.triu`] works?
        - Create triangular matrices by zeroing out relevant elements.
    - How [`torch.tensor.masked_fill`](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch-tensor-masked-fill) works?
        - Fills the masked elements with given value.
    - How [`torch.index_select`](https://pytorch.org/docs/stable/generated/torch.index_select.html) works?
        - Retains the elements for a tensor selectively based on the provided indices.

- [`understanding_tensor_manipulations_part_8.ipynb`](tensor_manipulations/understanding_tensor_manipulations_part_8.ipynb)
    - How [`torch.unique`](https://pytorch.org/docs/stable/generated/torch.unique.html) works?
        - Selects unique elements from the provided dimension.

#### `autograd/`

Contains experiments with `autograd` functionality in Pytorch. 

- [`understanding_autograd_pytorch_part_1.ipynb`](autograd/understanding_autograd_pytorch_part_1.ipynb)
    - How to use autograd on general tensor operations without using neural networks.

#### `modules/`

Contains experiments with Pytorch `modules`.

- [`understanding_nn_linear.ipynb`](modules/understanding_nn_linear.ipynb)
    - What is [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#linear) and how to use the linear module.
- [`using_modules.ipynb`](modules/using_modules.ipynb)
    - What is a `nn.module` in Pytorch?
    - How to use `Sequential module`?
    - What are `submodules` of a module?
    - What is the `state` of a module?
    - What is `Buffer` in a module?
    - What is a `ModuleList`?
- [`building_simple_neural_network_using_modules.ipynb`](modules/building_simple_neural_network_using_modules.ipynb)
    - How to build a simple custom neural network using modules?

#### `miscellaneous/`

Contains all other notebooks that do not fall into any specific category.

- [`common_functions.ipynb`](miscellaneous/common_functions.ipynb)
    - What is `Softmax` and how to use it?
    - What is `LogSoftmax` and how to use it?
- [`optimizers.ipynb`](miscellaneous/optimizers.ipynb)
    - What is `AdamOptimizer` and how to use it?
- [`learning_rates.ipynb`](miscellaneous/learning_rates.ipynb)
    - What is learning rate and why is it used?
    - What is learning rate decay and why is it used?
    - How to use different learning rate schedulers in Pytorch?
- [`loss_functions.ipynb`](miscellaneous/loss_functions.ipynb)
    - What is KL Divergence loss and how to use it in Pytorch?
- [`using_gpu_in_pytorch.ipynb`](miscellaneous/using_gpu_in_pytorch.ipynb)
    - How to use Pytorch to train models on GPUs?

#### `Data/`

Contains any artifacts generated while running the notebooks in this repository.

- [`my_neural_network_1.pt`](data/my_neural_network_1.pt)
    - The pickle file containing the model created in [`building_simple_neural_network_using_modules.ipynb`](modules/building_simple_neural_network_using_modules.ipynb)

## Useful Resources

This section contains some of the resources I referred to to understand tensors and tensor manipulations.

#### Tensor Manipulations

- [Official Documentation](https://pytorch.org/docs/stable/tensors.html#torch-tensor) explains the basics of `tensors`.
- [Blog](https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be) explains the concept of `dimensions` in tensor and visualizing them.
- [Blog](https://martinlwx.github.io/en/how-to-reprensent-a-tensor-or-ndarray/) explains how a tensor is represented in memory and how are the values in the tensor accessed.
- [Post](https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2) explains what is a `contiguous` tensor and how to create contiguous tensors.
- [Blog](https://dzone.com/articles/reshaping-pytorch-tensors) explains how `reshape` and `view` operations work using `stride` and `contiguous` properties.
- [Blog](https://kamilelukosiute.com/pytorch/When+can+a+tensor+be+view()ed%3F) explains the necessary conditions for a `view` operation to be valid.
- [Video](https://www.youtube.com/watch?v=kF2AlpykJGY) explains what `torch.stack` operation does.
- [Video](https://www.youtube.com/watch?v=tKcLaGdvabM) explains the concept of `broadcasting` and how it works in pytorch.
- [Video](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics) explains how the shape of the input tensors and resultant tensors change because of broadcasting.
- [Video](https://yuyangyy.medium.com/understand-torch-scatter-b0fd6275331c) explains how `scatter_` function works in detail with examples.

#### Autograd

- [Video](https://www.youtube.com/watch?v=wG_nF1awSSY) explains the concept of `Automatic Differentiation`.
- [Video](https://videolectures.net/deeplearning2017_johnson_automatic_differentiation/) gives a deep-dive on Automatic Differentiation including the mathematics behind it.
- [Blog](https://deeplearning.neuromatch.io/tutorials/W1D2_LinearDeepLearning/student/W1D2_Tutorial1.html) explains automatic differentiation at high level and how to apply it in Pytorch.
- [GitHub Repository](https://github.com/MB1151/mimic_micro_autograd) implements a mini-version of autograd from scratch.

#### Pytorch Modules

- [Blog](https://docs.kanaries.net/topics/Python/nn-linear) explains what is Linear layer in Pytorch and how to use it.
- [Official Documentation](https://pytorch.org/docs/stable/notes/modules.html) explains the concepts of modules and everything associated with it.
- [Blog](https://discuss.pytorch.org/t/what-does-register-buffer-do/121091) discusses the usage of `Buffers` vs `Parameters`.
- [Blog](https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723) again discusses the differences between `Buffers` and `Parameters`. 

#### Optimizers

- [Video](https://www.youtube.com/watch?v=lAq96T8FkTw&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=18) explains the concept of exponentially weighted average.
- [Video](https://www.youtube.com/watch?v=NxTFlzBjS-4&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=19) explains why exponentially weighted average works.
- [Video](https://www.youtube.com/watch?v=lWzo8CajF5s&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=19) explains bias correction in exponentially weighted average.
- [Video](https://www.youtube.com/watch?v=k8fTYJPd3_I&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=20) explains gradient descent with momentum.
- [Video](https://www.youtube.com/watch?v=_e-LFe_igno&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=22) explains RMS Prop.
- [Video](https://www.youtube.com/watch?v=JXQT_vxqwIs&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=22) explains the Adam Optimization algorithm.
- [Blod](https://www.linkedin.com/pulse/getting-know-adam-optimization-comprehensive-guide-kiran-kumar/) reiterates the concept of Adam Optimization.

#### Learning Rate

- [Blog](https://www.jeremyjordan.me/nn-learning-rate/) explains what learning rate is and why it is used.
- [Video](https://www.youtube.com/watch?v=QzulmoOg2JE&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=23) explains what learning rate decay is and why is it used.
- [Video](https://www.youtube.com/watch?v=81NJgoR5RfY) explains how to use different learning rate schedulers in Pytorch.

#### Loss Functions

- [Blog](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained) gives an intuitive explanation of KL Divergence.
- [Blog](https://encord.com/blog/kl-divergence-in-machine-learning/) explains KL Divergence in the context of Machine Learning.
- [Blog](https://dibyaghosh.com/blog/probability/kldivergence.html) explains the mathematics behind KL Divergence.

#### Using GPU for training

- [Video](https://youtu.be/6stDhEA0wFQ?si=rkc0iKKRxWnaYbYo) gives a very high level overview of why we use GPUs.
- [Video](https://youtu.be/Bs1mdHZiAS8?si=0SpkfO3POIuffsv3) shows how to create tensors and modules on GPU using Pytorch.
- [Blog](https://wandb.ai/ayush-thakur/dl-question-bank/reports/How-To-Check-If-PyTorch-Is-Using-The-GPU--VmlldzoyMDQ0NTU) shows some basic commands to retrieve GPU details on laptops.
- [Blog](https://www.run.ai/guides/gpu-deep-learning/pytorch-gpu) explains a few more details about how GPUs are used in Pytorch.


## Usage

### Set Up

Create a Virtual Environment for this project that will contain all the dependencies.

```python -m venv .pytorch_venv```

Run the following command to install the necessary packages in the virtual environment.

```pip install -r requirements.txt```

Continue to run the Jupyter notebooks in the `pytorch_venv` virtual environment.