
from model import *
from dataloader import *

from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os
import pickle





TRAIN_DIR = "/home/rabink1/D1/vtf_images/tuh/train"
VAL_DIR = "/home/rabink1/D1/vtf_images/tuh/eval"
CHECKPOINT_PATH = "./saved_models/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def normalise(x):
    norm=(2*x - x.max()- x.min())/(x.max() - x.min())
    return norm 

def norm_unit8(image):
    image /= image.max()/255.0 
    return image

train_transforms = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Lambda(normalise)
        #transforms.Lambda(norm_unit8)
    ])
val_transforms = transforms.Compose([
       #transforms.ToTensor(),
        transforms.Lambda(normalise)
        #transforms.Lambda(norm_unit8)

    ])

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=False,
                              num_workers=4,
                              # pin_memory=True if device else False
                              )
val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=4,
                            # pin_memory=True if device else False
                            )

def get_train_images(num):
    return torch.stack([train_dataset[i][0] for  i in range(num)], dim=0)





def train_eeg(latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"EEG_{latent_dim}"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=500,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                   # GenerateCallback(get_train_images(8), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"EEG_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=64, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    #test_result = trainer.test(model, test_loader, verbose=False)
    #result = {"test": test_result, "val": val_result}
    result = {"val": val_result}
    return model, result






if __name__=="__main__":

    model_dict = {}
    for latent_dim in [64, 128, 256]:
        model_ld, result_ld = train_eeg(latent_dim)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
    f=open("latent_dim.pkl","wb")
    pickle.dump(model_dict,f)
    f.close()
