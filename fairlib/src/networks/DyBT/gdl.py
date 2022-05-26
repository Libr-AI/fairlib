import itertools
import torch

class Group_Difference_Loss(torch.nn.Module):

    def __init__(self, args):
        super(Group_Difference_Loss, self).__init__()

        self.args = args
        if self.args.regression:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.DyBTObj = args.DyBTObj
        self.DyBTalpha = args.DyBTalpha
        
        # Takes the unique values of the target and group labels
        self.g_item = [tmp_g for tmp_g in range(args.num_groups)]
        self.y_item = [tmp_y for tmp_y in range(args.num_classes)]
        self.yg_tuple = list(itertools.product(self.y_item, self.g_item))
    
    def calculate_difference_loss(self, predictions, tags, g_mask, y_mask, yg_mask, regression_tags = None):
        gd_loss = 0

        if self.DyBTObj in ["joint", "y", "g"]:
            overall_loss = self.criterion(
                predictions, 
                tags if regression_tags is None else regression_tags
                )
            if self.DyBTObj == "joint":
                distinct_groups = self.yg_tuple
                group_mask = yg_mask
            elif self.DyBTObj == "y":
                distinct_groups = self.y_tuple
                group_mask = y_mask
            elif self.DyBTObj == "g":
                distinct_groups = self.g_tuple
                group_mask = g_mask
            else:
                raise NotImplementedError

            for tmp_group in distinct_groups:
                tmp_group_index = group_mask.get(tmp_group, [])
                if len(tmp_group_index) > 0:
                    tmp_group_loss = self.criterion(
                        predictions[tmp_group_index], 
                        tags[tmp_group_index] if regression_tags is None else regression_tags[tmp_group_index]
                        )
                    if tmp_group_loss>overall_loss:
                        gd_loss += tmp_group_loss
                    elif tmp_group_loss<overall_loss:
                        gd_loss -= tmp_group_loss 
        
        elif self.DyBTObj in ["EO", "stratified_y"]:
            for tmp_y in self.y_item:
                tmp_y_index = y_mask.get(tmp_y, [])
                tmp_y_loss = self.criterion(
                    predictions[tmp_y_index], 
                    tags[tmp_y_index] if regression_tags is None else regression_tags[tmp_y_index]
                    )
                for tmp_g in self.g_item:
                    tmp_yg_index = yg_mask.get((tmp_y, tmp_g), [])
                    tmp_yg_loss = self.criterion(
                        predictions[tmp_yg_index], 
                        tags[tmp_yg_index] if regression_tags is None else regression_tags[tmp_yg_index],
                        )
                    if tmp_yg_loss > tmp_y_loss:
                        gd_loss += tmp_yg_loss
                    elif tmp_yg_loss < tmp_y_loss:
                        gd_loss -= tmp_yg_loss
        
        else:
            raise NotImplementedError

        return gd_loss * self.DyBTalpha
    
    def forward(self, predictions, tags, p_tags, regression_tags=None):

        # Makes masks
        g_mask = {}
        y_mask = {}
        yg_mask = {}
        
        for tmp_g in self.g_item:
            g_mask[tmp_g] = list(torch.where(p_tags == tmp_g)[0].cpu().numpy())
            
        for tmp_y in self.y_item:
            y_mask[tmp_y] = list(torch.where(tags == tmp_y)[0].cpu().numpy())
            
        for (tmp_y, tmp_g) in self.yg_tuple:
            yg_mask[(tmp_y, tmp_g)] = list(set(torch.where(tags == tmp_y)[0].cpu().numpy()).intersection(set(torch.where(p_tags == tmp_g)[0].cpu().numpy())))
        
        return self.calculate_difference_loss(predictions, tags, g_mask, y_mask, yg_mask, regression_tags)