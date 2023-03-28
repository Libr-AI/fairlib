import torch
import itertools

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.pow(x - y, p).sum(2)
    
    return dist ** (1/p)

def KNN(x, y, p, k, include_self):
    dist = distance_matrix(x, y, p)
        
    if include_self:
        k_indices = dist.topk(k, largest=False).indices
    else:
        k_indices = dist.topk(k+1, largest=False).indices[:,1:]
    return k_indices

def KNN_labels(criterion, tags, text, model, predictions, loss, p = 2, k = 5, average_first = False, include_self = True):
    """Derive proxy labels with NN correction

    Args:
        criterion (function): loss function
        tags (torch.tensor): target labels
        text (inputs): inputs
        model (torch.module): target model
        predictions (torch.tensor): model predictions
        loss (torch.tensor): average loss
        p (int, optional): norm. Defaults to 2.
        k (int, optional): number of NN. Defaults to 5.
        average_first (bool, optional): voting after average aggregation. Defaults to False.
        include_self (bool, optional): if the query instance itself is considered as a NN. Defaults to True.

    Returns:
        proxy label (torch.tensor): proxy label assignment with correction
    """
    y_item = list(set(tags.tolist()))
    hs = model.hidden(text)

    if not average_first:
        if model.args.regression:
            criterion = torch.nn.MSELoss(reduction = "none")
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction = "none")

    y_mask = {}
    for tmp_y in y_item:
        y_mask[tmp_y] = (tags == tmp_y)

    knn_labels = torch.zeros_like(tags)

    for temp_y in y_item:
        temp_y_masks = y_mask[temp_y]
        class_hs = hs[temp_y_masks]

        k_indices = KNN(class_hs, class_hs, p, k, include_self)

        temp_y_knn_labels = []
        for tmp_tags, tmp_preds in zip(tags[temp_y_masks][k_indices],predictions[temp_y_masks][k_indices]):
            tmp_loss = criterion(tmp_preds, tmp_tags)
            temp_y_knn_labels.append(tmp_loss > loss)

        if average_first:
            temp_y_knn_labels = torch.tensor(temp_y_knn_labels).squeeze().long().to(knn_labels.device)
        else:
            temp_y_knn_labels = torch.stack(temp_y_knn_labels, dim=0).mode(1).values.long().to(knn_labels.device)
        knn_labels[temp_y_masks.nonzero().squeeze()] = temp_y_knn_labels
    
    return knn_labels.long()

class KNN_Loss(torch.nn.Module):

    def __init__(self, args):
        super(KNN_Loss, self).__init__()

        self.args = args
        if self.args.regression:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.UKNN_lambda = args.UKNN_lambda
        
        # The proxy labels of KNN outputs
        self.g_item = [0,1]
    
    def calculate_loss(self, predictions, tags, g_mask, regression_tags = None):
        knn_loss = 0

        for tmp_group in self.g_item:
            tmp_group_index = g_mask.get(tmp_group, [])
            if len(tmp_group_index) > 0:
                tmp_group_loss = self.criterion(
                    predictions[tmp_group_index], 
                    tags[tmp_group_index] if regression_tags is None else regression_tags[tmp_group_index]
                    )
                if tmp_group == 1:
                    knn_loss += tmp_group_loss
                else:
                    knn_loss -= tmp_group_loss 

        return knn_loss * self.UKNN_lambda
    
    def forward(self, predictions, tags, knn_tags, regression_tags=None):

        # Makes masks
        g_mask = {}

        for tmp_g in self.g_item:
            g_mask[tmp_g] = list(torch.where(knn_tags == tmp_g)[0].cpu().numpy())
            
        return self.calculate_loss(predictions, tags, g_mask, regression_tags)