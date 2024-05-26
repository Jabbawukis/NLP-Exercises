[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/VBjsdHXJ)
# Exercise 4

This exercise contains two tasks: (1) Check for well-formed parentheses using RNNs and (2) build a RNN part-of-speech tagger.


## Instructions

Accept the assignment in GitHub classrooms (link on exercise sheet and slides).
Clone the repository to your local machine.
All code completion tasks are required with a TODO-tag inside the code.

## Task 1

In this task you need to check for well-formed parentheses using RNNs. The RNN is simple - it has 2 hidden states and the input vectors will be [1,0] for "(" and [0,1] for ")". After processing the sequence, the 2-dimensional will be passed through a linear layer (weights + bias term) to obtain a final scalar which will be used in a sigmoid activation to predict whether the given string consists out of well-formed parentheses.

Your task is to set the weights (in task_1/weights.txt):
- apply linear transformation to input vectors (rnn.weight_ih_l0)
- apply linear transformation to last hidden state (rnn.weight_hh_l0)
- both outputs will be added, followed by a ReLU activation
- after last parantheses, the hidden state will be passed through a classification layer (classifier.weight and classifier.bias)

## Task 2

Complete the class RecurrentModel in task_2/model.py. You will need to complete the constructor and rnn_forward method. In the second part of this task, you will extend your implementation to be a bi-directional forward.
