# Other NN Architecture

## Classification Head
Our current MLP implementation (`fairlib/src/networks/classifier`) can be used as a classification head for different backbone nets. All our methods such as balanced training and adversarial training will be supported for the new model.

## Customized model architecture

Take a look at the following example, we use the BERT model as the feature extracting network, i.e., extracting sentence representations from the BERT, and then use the extracted features as the input to the MLP classifier to make predictions. 

We only need to define three functions: `__init__`, which is used to init the model with pretrained BERT parameters, MLP classifier, and optimizers; `forward`, which is the same to before where we extract sentence representations then use MLP to make predictions; and `hidden`, which is used to get hidden representations for adversarial training.

```python
from transformers import BertModel
from fairlib.networks.classifier import MLP

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

## Register Model
the model architecture is indicated by `--encoder_architecture`, so we will need to handle different values of this argument. 
Specifically, we need to modify the `get_main_model` function in `fairlib/src/networks/__init__.py` to support new models.

```python
def get_main_model(args):
    # Add the new model name here.
    assert args.encoder_architecture in ["Fixed", "BERT", "DeepMoji", "NEW_MODEL_NAME"], "Not implemented"

    if args.encoder_architecture == "Fixed":
        model = MLP(args)
    elif args.encoder_architecture == "BERT":
        model = BERTClassifier(args)
    # Init the model 
    elif args.encoder_architecture == "NEW_MODEL_NAME":
        model = MODEL(args)
    else:
        raise "not implemented yet"
```

## Register the Dataloader
Since different models have their own mapping from tokens to numerical ids. We need to handle this in the dataloader.

Firstly, we need to init the tokenizer in `fairlib/src/dataloaders/encoder.py`, for example,
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
```

Next, we need to modify the corresponding dataloader to return the idx of input texts. Please take a look at the Bios loader in `fairlib/src/dataloaders/loaders.py` for detailed examples.

Noticing that, to avoid encoding text to idx repeatedly, we could pre-calculate the mapped idx for the desired model, and load from file to save time.

## Extensions

```python
class BiLSTMPOSTagger(nn.Module):
    def __init__(self, 
                input_dim, 
                embedding_dim, 
                hidden_dim, 
                output_dim, 
                n_layers, 
                bidirectional, 
                dropout, 
                pad_idx):
        
        super().__init__()
        
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        # args.input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        # args.output_dim = hidden_dim * 2 if bidirectional else output_dim
        # args.n_hidden = 0
        # self.fc = MLP(args)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        #text = [sent len, batch size]
        
        #pass text through embedding layer
        embedded = self.embedding(text)
        
        #embedded = [sent len, batch size, emb dim]
        
        #pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        #outputs holds the backward and forward hidden states in the final layer
        #hidden and cell are the backward and forward hidden and cell states at the final time-step
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden/cell = [n layers * n directions, batch size, hid dim]
        
        #we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions

```