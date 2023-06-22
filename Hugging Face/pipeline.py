from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F
import torch.nn as nn

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

sentences = [
'we are very happy to show you the transformers library',
'we are not that happy to show you the transformers library',
'you are not what you think you are'

]
res = classifier(sentences)

for r in res:
    print(r)

