from transformers import pipeline

classifier =  pipeline('fill-mask')
res = classifier("Paris is the <mask> of France")
print(res)