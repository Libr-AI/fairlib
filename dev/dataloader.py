import sys
sys.path.append('../')

from src import dataloaders

class Args:
    task = "Moji"
    data_dir = "D:\\Project\\User_gender_removal\\data\\deepmoji\\split2\\"
    full_label = True
    BT = None
    BTObj = None

if __name__ == "__main__":

    args = Args()
    args.task = "Bios"
    args.data_dir = r"D:\Project\Minding_Imbalance_in_Discriminator_Training\data\bios"
    args.protected_task = "economy"

    for BT in ["Reweighting", "Resampling"]:
        for BTObj in ["joint", "y", "g", "stratified_y", "stratified_g"]:
            args.BT = BT
            args.BTObj = BTObj
            print(args.BT, args.BTObj)
            _train, _dev, _test = dataloaders.get_dataloaders(args)

            print(_train.X[:3])
            print(_train.y[:3])
            print(_train.protected_label[:3])
            print(_train.instance_weights[:3])