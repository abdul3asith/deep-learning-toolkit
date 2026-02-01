from transformers import pipeline

# define a classifier function
classifier = pipeline("sentiment-analysis")
result = classifier(["I am really excited to learn python", "I am gonna beat the shit out of that guy", "I might go the party today", "I am having second thoughts about today's event"])

print(result)

# zero-shot classification

classifier2 = pipeline("zero-shot-classification")
result2 = classifier2("This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],)
print(result2)
