# Setup

Clone this repository via Github Classrooms: https://classroom.github.com/a/IpwD52NO

Navigate to the cloned folder:
```bash
cd nlp-exercise-1
```

We will be using Python 3.9 throughout all exercises.
If you use conda (recommended), create a central conda environment with Python 3.9. You can easily re-use this environment through all exercises:
```bash
conda create -n nlpcourse python=3.9
conda activate nlpcourse
```

Alternatively, create a virtual environment with virtualenv:
```bash
pip install virtualenv
python -m venv env
source env/bin/activate
```

# Install dependencies

```bash
pip install -r requirements.txt
```

Verify dependencies are installed:
```bash
conda list flair
```

This should list:
```bash
# packages in environment at /vol/tmp/goldejon/.conda/envs/nlpcourse:
#
# Name                    Version                   Build  Channel
flair                     0.13.1                   pypi_0    pypi
```

# Assignments

You will need to complete functions and class in this exercise. We marked all places with the `TODO` label where you need to implement something. Places without the `TODO` labels do not need to be adjusted. You need to complete the tasks such it is possible to run each task like:

```bash
python run_task_1.py
```

## Task 1

In this task, you will crawl 10 headlines from 2 publishers of your choice with fundus. In a second step, you will obtain the sentiment for each headline with a pre-trained flair model. You need to implement three functions:

1. `crawl_headlines()` in `task_1/fundus_crawler.py`: Crawl 10 headlines for two publishers of your choice. See slides and the [fundus](https://github.com/flairNLP/fundus) documentation for reference.
2. `predict_labels()`in `task_1/flair_model.py`: Load a sentiment model with flair and predict the sentiment for each headline. See slides and the [flair](https://github.com/flairNLP/flair) documentation for reference.
3. `print_statistics()` in `run_task_1.py`: This function should print out information about the sentiments, i.e. how many headlines are positive or negative per publisher.

## Task 2:

We provide you with some basic implementation such as the training procedure. You need to implement three things:

1. Complete the class `BoWClassifier` in `task_2/bow_model.py`: The constructor and forward pass are missing. Fill these two parts as shown on the slides.

2. Implement the `get_large_training_data()` method in `task_2/data.py`: This function should read-in data from a given path (the `data` folder contains two .txt files - one for training and one for validation) and return a list of tuples in format (label, sentence) where label is a string and sentence a list of words.

3. Implement the `evaluate_final_model` in `task_2/evaluation.py`: This function should return the accuracy metric on your test dataset. Start by setting the `dataset_size` variable to `small` in `run_task_2.py`. This will give you a small example dataset to count correct and false predictions for debugging.

    Note: You may need to consider out-of-vocabulary words (words that are not observed in the training dataset). To handle these words, you may want to adjust `make_word_dictionary()` and `make_bow_vector()` in `task_2/bow_model.py`.

## Task 3 (Optional):

Add a new publisher to fundus. See [this](https://github.com/flairNLP/fundus/blob/master/docs/how_to_add_a_publisher.md) tutorial for reference.
