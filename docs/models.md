# Other NN Architecture

## Classification Head
Our current MLP implementation (`src\networks\classifier`) can be used as a classification head for different backbone nets. All our methods such as balanced training and adversarial training will be supported for the new model.

## Customized model architecture

Take a look at the following example, we use the BERT model as the feature extracting network, i.e., extracting sentence representations from the BERT, and then use the extracted features as the input to the MLP classifier to make predictions. 

We only need to define three functions: `__init__`, which is used to init the model with pretrained BERT parameters, MLP classifier, and optimizers; `forward`, which is the same to before where we extract sentence representations then use MLP to make predictions; and `hidden`, which is used to get hidden representations for adversarial training.

```python
from transformers import BertModel

class BERTClassifier(BaseModel):
    model_name = 'bert-base-cased'

    def __init__(self, args):
        super(BERTClassifier, self).__init__()
        self.args = args

        # Load pretrained model parameters.
        self.bert = BertModel.from_pretrained(self.model_name)

        # Init the classification head 
        self.classifier = MLP(args)

        # Init optimizers, criterions, etc.
        self.init_for_training()

    def forward(self, input_data, group_label = None):
        # Extract sentence representations from bert
        bert_output = self.bert(input_data)[1]

        # Make predictions
        return self.classifier(bert_output, group_label)
    
    def hidden(self, input_data, group_label = None):
        # Extract sentence representations from bert
        bert_output = self.bert(input_data)[1]

        # Make predictions
        return self.classifier.hidden(bert_output, group_label)
```