from transformers import pipeline

classifier = pipeline("fill-mask")
res = classifier(
    "Sobhan works in Montral Canada. Yesterday he went to work and <mask> paper"
)
print(res)
