# Trensformers

This is the implemetation of transformers based on what is explained here;
https://peterbloem.nl/blog/transformers

Reference for ext classifier dataset loading and preprocessing; 
https://torchtutorialstaging.z5.web.core.windows.net/beginner/text_sentiment_ngrams_tutorial.html

Notes: 
- even with one epoch above 85% test-accuracy is achieved for text classification task !!!
- The model can be run on CPU / GPU
- `Rate Scheduling` (sch.step()) is crutial for high level of accuracy - or fast training 
- I used `padding` to make the sequence have the same length in a batch
