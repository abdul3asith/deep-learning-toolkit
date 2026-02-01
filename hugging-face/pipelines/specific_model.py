# a specific model is provided for different types of tasks 
from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
result = generator("Here in this chapter, I am gonna be talking about", max_length=30, num_return_sequence=3
)
print(result)