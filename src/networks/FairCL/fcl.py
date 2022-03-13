from .utils import Contrastive_Loss
import torch


class Fair_Contrastive_Loss(torch.nn.Module):

    def __init__(self, args):
        super(Fair_Contrastive_Loss, self).__init__()

        self.device = args.device
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
        fcl_loss_g = self.fcl_lambda_g * self.contrastive_loss_g(hs, p_tags)

        return fcl_loss_y - fcl_loss_g