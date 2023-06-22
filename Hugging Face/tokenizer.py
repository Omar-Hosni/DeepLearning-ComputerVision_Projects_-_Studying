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

tokens = tokenizer.tokenize('we are very happy to show you the transformers library')
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer('we are very happy to show you the transformers library')

print(f'tokens: {tokens}')
print(f'token ids: {token_ids}')
print(f'input ids: {input_ids}')

X_train = [
'we are very happy to show you the transformers library',
'we are not that happy to show you the transformers library',
'you are not what you think you are'
]

batch = tokenizer(X_train, padding=True, truncation=True, max_length=100, return_tensors='pt')
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)

save_dir = 'saved'
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForSequenceClassification.from_pretrained((save_dir))


