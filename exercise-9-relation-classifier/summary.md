# Task 1

In this task, I modified the test/val split of the training data by adding the special tokens to the tokenizer and adding the tokens to the sentences. The encased sentences are passed to each batch with the labels (see process() method).

I tried different hyperparameters by simply testing different batch sizes and epochs (grid search-like). The results of the models are saved in the scores.txt. Here are the hyperparameters:

Test_1

* per_device_train_batch_size=16
* per_device_eval_batch_size=16
* num_train_epochs=10

Test_2

* per_device_train_batch_size=16
* per_device_eval_batch_size=16
* num_train_epochs=5

Test_3

* per_device_train_batch_size=64
* per_device_eval_batch_size=64
* num_train_epochs=10

Test_4

* per_device_train_batch_size=16
* per_device_eval_batch_size=16
* num_train_epochs=20

Test_5

* per_device_train_batch_size=16
* per_device_eval_batch_size=16
* num_train_epochs=15

Test_6

* per_device_train_batch_size=64
* per_device_eval_batch_size=64
* num_train_epochs=20

I uploaded the best model to HuggingFace with the following [link](https://huggingface.co/J4bb4wukis/exercise-9-relation-classifier)
since the model was too large.

# Task 2


