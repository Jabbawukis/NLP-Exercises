import torch

from sequence_to_sequence_model_old import Seq2Seq

# oad models with and without attention
translator: Seq2Seq = torch.load(f"models/best-translation-model.pt", map_location='mps')
print(translator)

text = "Mein Laptop ist neu and schnell ."

print("---- NO ATTENTION ----")
for i in range(4):
    translation = translator.translate(f"{text}\t", i+1)
    print(translation)