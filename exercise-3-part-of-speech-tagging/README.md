[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/9ZkftzNQ)
# Exercise 3


## Setup

Clone this repository via Github Classrooms.

Navigate to the cloned folder:

```bash
cd nlp-exercise-3-solution
```

Activate your environment (refer to exercise 1 for setting up your environment):
```bash
conda activate nlpcourse
```


## POS Tags used in this Assignment

- `.`: end of sentence
- `NN`: noun (singular)
- `NNS`: noun (plural)
- `NNP`: proper noun
- `DT`: determiner
- `PRP`: personal pronoun
- `IN`: preposition/subordinating conjunction
- `JJ`: adjective or numeral, ordinal
- `JJR`: adjective, comparative
- `JJS`: adjective, superlative
- `RB`: adverb
- `VB`: verb, base form
- `VBP`: verb (singular, present)
- `VBZ`: verb (singular, 3rd person, present)
- `VBD`: verb (singular, past tense)
- `CC`: coordinating conjunction
- `MD`: modal
- `EX`: existential there
- `TO`: "to" as preposition or infinitive marker
- `WDT`: WH-determiner
- `WP`: WH-pronoun
- `WRB`: WH-adverb


## Testing your code before submitting

We provided some test cases to give guidance while solving the tasks.
In your environment, run `python -m pytest` to execute the test cases.
Your submission should pass all of these tests.

You may need to update your environment using `pip install -r requirements.txt` to enable the use of pytest.


## Running the the training script

Use `python run_task_2.py --help` to get information on the cli options which can be used to set the hyperparameters for the training loop.
