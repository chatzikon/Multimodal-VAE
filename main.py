import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from src.config import config
from src.data import get_celeba_subset, CelebADataset, custom_collate_fn, custom_collate_fn_flickr, visualize_dataset_samples
from src.models import MultimodalVAE, PatchDiscriminator
from src.training import run_phased_training, TrainingManager
from src.visualization import visualize_results
from clip_dl import Flickr30kDataset
from transformers import AutoModel, AutoTokenizer, BertTokenizer


import gc

class Tokenizer:
    def __init__(self, max_length, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer
        self.max_length=max_length

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

def main(kl_coef):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    print(f"Using device: {device}")

    kl_coef_init=np.ones(3)

    for i in range(len(config['phase_configs'])):

        kl_coef_init[i] = config['phase_configs'][i + 1]['kl_weight']
        config['phase_configs'][i+1]['kl_weight']*=kl_coef

    if config['dataset']=='CelebAMask-HQ':

        model = MultimodalVAE(dataset=config['dataset'], latent_dim=config['latent_dim'], temperature=1.0).to(device)

    elif config['dataset'] == 'Flickr30k':

        model = MultimodalVAE(dataset=config['dataset'], latent_dim=config['latent_dim'], num_attributes=config['max_length'], temperature=1.0).to(device)


    discriminator = PatchDiscriminator(in_channels=3, ndf=64).to(device)

    optimizer_G = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate_G'],
        betas=(0.5,0.999),
        weight_decay=0.01
    )

    optimizer_D = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config['learning_rate_D'],
        betas=(0.5,0.999),
        weight_decay=0.02
    )

    if config['dataset']=='CelebAMask-HQ':

        # Load dataset
        train_dataset = get_celeba_subset(
            config['data_path'],
            subset_size=5000,
            random_subset=True,
            cache_path=config['cache_path']
        )
        full_dataset = CelebADataset(train_dataset)
        visualize_dataset_samples(full_dataset, num_samples=5, save_path='dataset_samples.png')

        dataset_size = len(full_dataset)
        val_size = int(config['val_split'] * dataset_size)
        train_size = dataset_size - val_size

        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config['seed'])
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        clip_tokenizer=None


    elif config['dataset']=='Flickr30k':

        train_subset = Flickr30kDataset('train')
        val_subset = Flickr30kDataset('validation')
        #
        #
        # Create the DataLoader
        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                                  num_workers=config['num_workers'], collate_fn=custom_collate_fn_flickr)
        val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, pin_memory=True,
                                num_workers=config['num_workers'], collate_fn=custom_collate_fn_flickr)

        visualize_dataset_samples(train_subset, num_samples=5, save_path='dataset_samples.png')

        clip_tokenizer = Tokenizer(config['max_length'], AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"))

    trainer = TrainingManager(kl_coef)

    phase1_epochs = config['phase1_epochs']
    phase2_epochs = config['phase2_epochs']
    phase3_epochs = config['phase3_epochs']

    phase1_start = 0
    phase2_start = 0
    phase3_start = 0

    if config['resume_phase'] in [1,2,3] and config['resume_epoch']:
        checkpoint_path = f"training/checkpoints/phase{config['resume_phase']}/checkpoint_epoch_{config['resume_epoch']}.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming phase {config['resume_phase']} from epoch {start_epoch+1}")
            if config['resume_phase']==1:
                phase1_start=start_epoch
            elif config['resume_phase']==2:
                phase1_epochs=0
                phase2_start=start_epoch
            elif config['resume_phase']==3:
                phase1_epochs=0
                phase2_epochs=0
                phase3_start=start_epoch
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")

    best_loss = run_phased_training(
        model=model,
        discriminator=discriminator,
        clip_tokenizer=clip_tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        device=device,
        config=config,
        trainer=trainer,
        val_subset=val_subset,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        phase3_epochs=phase3_epochs,
        phase1_start=phase1_start,
        phase2_start=phase2_start,
        phase3_start=phase3_start,
        kl_coef=kl_coef
    )

    print(f"\nTraining completed! Best validation loss: {best_loss:.4f}")
    final_checkpoint_path = 'checkpoints/final_model_kl_coef_'+str(kl_coef)+'.pt'
    torch.save({
        'model_state_dict':model.state_dict(),
        'discriminator_state_dict':discriminator.state_dict(),
        'optimizer_G_state_dict':optimizer_G.state_dict(),
        'optimizer_D_state_dict':optimizer_D.state_dict(),
        'config':config,
        'final_loss':best_loss
    }, final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")

    del model
    del discriminator
    del optimizer_G
    del optimizer_D

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    for i in range(len(config['phase_configs'])):
        config['phase_configs'][i+1]['kl_weight']=kl_coef_init[i]
        print(config['phase_configs'][i + 1]['kl_weight'])



if __name__ == "__main__":

    kl_coef=[1000,500,400,300,200,100,10,1,1e-1,1e-2,1e-3]

    for i in range(len(kl_coef)):
        main(kl_coef[i])


