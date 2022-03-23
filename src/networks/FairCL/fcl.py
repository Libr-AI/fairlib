from .utils import Contrastive_Loss
import torch


class Fair_Contrastive_Loss(torch.nn.Module):

    def __init__(self, args):
        super(Fair_Contrastive_Loss, self).__init__()

        self.device = args.device
        self.FCLObj = args.FCLObj
        self.temperature_y = args.fcl_temperature_y
        self.temperature_g = args.fcl_temperature_g
        self.base_temperature_y = args.fcl_base_temperature_y
        self.base_temperature_g = args.fcl_base_temperature_g
        self.fcl_lambda_y = args.fcl_lambda_y
        self.fcl_lambda_g = args.fcl_lambda_g


        self.contrastive_loss_y = Contrastive_Loss(device=self.device, temperature=self.temperature_y, base_temperature= self.base_temperature_y)
        self.contrastive_loss_g = Contrastive_Loss(device=self.device, temperature=self.temperature_g, base_temperature= self.base_temperature_g)

    def forward(self, hs, tags, p_tags):
        
        fcl_loss_y = self.fcl_lambda_y * self.contrastive_loss_y(hs, tags)
        if self.FCLObj == "g":
            fcl_loss_g = self.fcl_lambda_g * self.contrastive_loss_g(hs, p_tags)
        elif self.FCLObj == "EO":
            # Get fcl for each groups
            distinct_y_labels = list(set(tags.cpu().numpy()))

            y_mask = {}
            for tmp_y in distinct_y_labels:
                y_mask[tmp_y] = list(torch.where(tags == tmp_y)[0].cpu().numpy())

            fcl_loss_g = 0
            for tmp_group in distinct_y_labels:
                tmp_group_index = y_mask.get(tmp_group, [])
                fcl_loss_g += self.contrastive_loss_g(hs[tmp_group_index], p_tags[tmp_group_index])
            
            fcl_loss_g = self.fcl_lambda_g * fcl_loss_g/len(distinct_y_labels)

        return fcl_loss_y - fcl_loss_g