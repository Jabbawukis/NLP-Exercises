# Test if your model performs as expected

from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained(
    "J4bb4wukis/exercise-9-relation-classifier", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("J4bb4wukis/exercise-9-relation-classifier")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

example_pos = "Kakizawa suggested that Hata 's remarks expressed anticipation of the truth of claims by North Korean President [E1]Kim Il-song[/E1] that [E2]North Korea[/E2] has neither the will nor the ability to develop nuclear weapons ."
example_neg = "In [E2]California[/E2] 's [E1]Sierra Nevada[/E1] foothills , firefighters on Tuesday contained a blaze that burned 11 , 700 acres of brush and grassland and destroyed seven homes ."

out = pipe([example_pos, example_neg], padding=True)
print(out) # should return LABEL_1 for the first example and LABEL_0 for the second