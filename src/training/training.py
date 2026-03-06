import os
import json
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

from torchmetrics.text import ROUGEScore, BLEUScore


from .losses import (
    discriminator_loss, generator_loss, generator_loss_eval,
    discriminator_loss_eval, compute_distribution_matching_loss,
    identity_consistency_loss
)
from ..visualization.viz import visualize_results

def create_masks(batch, pad_token):
    """
    Create padding and square subsequent masks
    S - max sequence length
    B - batch size
    :param batch: batch of sentences to calculate masks for tensor(S,B)
    :param pad_token: value of pad_token in the vocabulary definition
    :return: pad_mask: padding mask, tensor(S,B)
             mask: square subsequent mask, tensor(S,S)
    """
    pad_mask = (batch == pad_token).transpose(0, 1)

    seq_len = batch.shape[0]
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=batch.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return pad_mask, mask

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
         flat_list.extend(row)
    return flat_list

def evaluate_all_metrics(gts, res):
    scores = {}

    # BLEU (1–4)
    bleu_scorer = Bleu(4)
    bleu, bleu_ind = bleu_scorer.compute_score(gts, res)
    scores['BLEU'] = bleu  # list of 4 BLEU scores



    print(gts)
    print(res)
    # METEOR
    try:
        meteor, meteor_ind = Meteor().compute_score(gts, res)
    except ValueError as e:
        import json
        import datetime

        # 1. Create a unique filename for the crash dump
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_file = f"meteor_crash_epoch_10_{timestamp}.json"

        # 2. Save the state so you can debug later
        with open(dump_file, "w") as f:
            json.dump({"gts": gts, "res": res}, f)

        print(f"\n[!] METEOR Error at Epoch 10: {e}")
        print(f"[!] Data dumped to {dump_file}. Searching for empty strings...")

        # 3. Quick internal check to tell you WHY it failed in the console
        for img_id, captions in res.items():
            if not captions or not captions[0].strip():
                print(f"    -> Found empty prediction at ID: {img_id}")

        # 4. Provide fallback values so the training script doesn't stop
        meteor = 0.0
        meteor_ind = [0.0] * len(gts)
        print("[!] Continuing training with METEOR = 0.0 for this epoch.\n")



    scores['METEOR'] = meteor

    # ROUGE-L
    rouge, rouge_ind = Rouge().compute_score(gts, res)
    scores['ROUGE_L'] = rouge

    # CIDEr
    cider, cider_ind = Cider().compute_score(gts, res)
    scores['CIDEr'] = cider

    # SPICE (requires Java 8!)
    spice, spice_ind = Spice().compute_score(gts, res)
    scores['SPICE'] = spice

    return scores


class TrainingManager:
    def __init__(self, kl_coef, lr, save_dir='training'):
        self.save_dir = save_dir
        os.makedirs(save_dir,exist_ok=True)
        self.history_path = os.path.join(save_dir,'history_'+str(kl_coef)+'_lr_'+str(lr)+'.csv')
        self.history=[]
        self.info_path = os.path.join(save_dir,'run_info_'+str(kl_coef)+'_lr_'+str(lr)+'.json')
        self.kl_coef=kl_coef
        self.lr=lr


    def save_checkpoint(self, epoch, model, discriminator, optimizer_G, optimizer_D, loss, phase):
        phase_checkpoint_dir = os.path.join(self.save_dir, 'checkpoints', str(self.kl_coef)+'_lr_'+str(self.lr)+'_', f'phase{phase}')
        os.makedirs(phase_checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(phase_checkpoint_dir,f'checkpoint_epoch_{epoch}_kl_coef_{self.kl_coef}_lr_{self.lr}.pt')
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'discriminator_state_dict':discriminator.state_dict(),
            'optimizer_G_state_dict':optimizer_G.state_dict(),
            'optimizer_D_state_dict':optimizer_D.state_dict(),
            'loss':loss
        }, checkpoint_path)
        self._save_run_info(epoch+1,phase)

    def _save_run_info(self, last_epoch, phase):
        info = {
            'last_epoch':last_epoch,
            'phase':phase,
            'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.info_path,'w') as f:
            json.dump(info,f)

    def log_metrics(self,epoch,metrics):
        metrics['epoch'] = epoch+1
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.history.append(metrics)
        df = pd.DataFrame(self.history)
        df.to_csv(self.history_path,index=False)

def train_step(model, discriminator, batch, optimizer_G, optimizer_D, device, epoch, phase_config, clip_tokenizer):
    images = batch['image'].to(device)
    if phase_config['dataset']=='CelebAMask-HQ':
        target_attributes = batch['attributes'].to(device) if 'attributes' in batch else None
        pad_mask=0
        tgt_mask=0
    elif phase_config['dataset']=='Flickr30k':
        text = batch["caption"]
        input_tokens = clip_tokenizer(text).to(device)
        target_attributes = input_tokens['input_ids']
        attention_mask = input_tokens['attention_mask']
        target_attributes=target_attributes.transpose(0,1)


        pad_mask, tgt_mask = create_masks(target_attributes, pad_token=clip_tokenizer.tokenizer.pad_token_id)

        pad_idx = clip_tokenizer.tokenizer.pad_token_id

    outputs = model(True, images=images, target_attributes=target_attributes, pad_mask=pad_mask, tgt_mask=tgt_mask, pad_id=pad_idx)

    # if phase_config['dataset'] == 'Flickr30k':
    #     pred_ids = torch.argmax(outputs['recon_text_probs'], dim=-1).float()
    #
    #     pred_ids_text2img = torch.argmax(outputs['text_from_image_probs'], dim=-1).float()


    losses = {}

    if phase_config['adversarial_weight'] > 0:
        optimizer_D.zero_grad()
        d_loss = phase_config['adversarial_weight'] * discriminator_loss(
            discriminator, images, outputs['recon_images'], epoch, phase_config
        )
        d_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_D.step()
        losses['d_loss'] = d_loss.item()
    else:
        losses['d_loss'] = 0.0

    if phase_config['adversarial_weight'] > 0:
        losses['adversarial'] = phase_config['adversarial_weight'] * generator_loss(
            discriminator, outputs['recon_images']
        )
    else:
        losses['adversarial'] = torch.tensor(0.0, device=device)

    optimizer_G.zero_grad()

    # Use model's vgg for perceptual loss
    with torch.no_grad():
        target_feats = model.vgg(images)
    recon_feats = model.vgg(outputs['recon_images'])
    mse_loss = F.mse_loss(outputs['recon_images'], images)
    perceptual_loss_val = F.mse_loss(recon_feats, target_feats)
    identity_loss = identity_consistency_loss(
        model.image_encoder(images)[0],
        model.image_encoder(outputs['recon_images'])[0]
    )
    losses['image_recon'] = phase_config['reconstruction_weight']*(mse_loss + 0.05*perceptual_loss_val + 0.1*identity_loss)

    if target_attributes is not None:
        # bce_loss = F.binary_cross_entropy(
        #     outputs['recon_text_probs'],
        #     target_attributes,
        #     reduction='none'
        # )
        #
        # penalized_loss = torch.where(
        #     bce_loss>0.15,
        #     bce_loss*8.0,
        #     bce_loss
        # )




        if phase_config['dataset']=='CelebAMask-HQ':
            text_attr_loss = F.binary_cross_entropy(
                outputs['recon_text_probs'],
                target_attributes
            )

            img2text_attr_loss = F.binary_cross_entropy(
                outputs['text_from_image_probs'],
                target_attributes
            )

        elif phase_config['dataset']=='Flickr30k':
            text_attr_loss = F.cross_entropy(
                outputs['recon_text_probs'][:-1,:,:].view(-1, outputs['recon_text_probs'].size(-1)),
                target_attributes[1:,:].flatten(), ignore_index=pad_idx
            )


            img2text_attr_loss = F.cross_entropy(
                outputs['text_from_image_probs'][:-1,:,:].view(-1, outputs['text_from_image_probs'].size(-1)),
                target_attributes[1:,:].flatten(), ignore_index=pad_idx
            )


        losses['text_recon_loss'] = phase_config['text_reconstruction_weight'] * text_attr_loss



        losses['text_from_image'] = phase_config['cross_modal_weight'] * img2text_attr_loss

        attr_consistency_loss = F.mse_loss(
            outputs['recon_text_probs'],
            outputs['text_from_image_probs']
        )




    losses['attr_consistency'] = phase_config['consistency_weight']*attr_consistency_loss

    if 'image_from_text' in outputs:
        with torch.no_grad():
            target_feats_txt = model.vgg(images)
        recon_feats_txt = model.vgg(outputs['image_from_text'])
        losses['image_from_text'] = phase_config['cross_modal_weight']*(
            F.mse_loss(outputs['image_from_text'],images)+
            0.1*F.mse_loss(recon_feats_txt, target_feats_txt)+
            0.1*identity_consistency_loss(
                model.image_encoder(images)[0],
                model.image_encoder(outputs['image_from_text'])[0]
            )
        )

    if all(k in outputs for k in ['image_mu','image_log_var']):
        losses['image_kl'] = phase_config['kl_weight'] * (-0.5 * torch.mean(
            1+outputs['image_log_var'] - outputs['image_mu'].pow(2)-outputs['image_log_var'].exp()
        ))

    if all(k in outputs for k in ['text_mu','text_log_var']):
        losses['text_kl'] = phase_config['kl_weight'] * (-0.5*torch.mean(
            1+outputs['text_log_var']-outputs['text_mu'].pow(2)-outputs['text_log_var'].exp()
        ))

    if all(k in outputs for k in ['image_mu','image_log_var','text_mu','text_log_var']):
        losses['distribution_matching'] = phase_config['consistency_weight'] * compute_distribution_matching_loss(
            outputs['image_mu'], outputs['image_log_var'],
            outputs['text_mu'], outputs['text_log_var']
        )

    if 'consistency_score' in outputs:
        losses['consistency'] = phase_config['consistency_weight']*(1-outputs['consistency_score'].mean())

    # if target_attributes is not None and phase_config['attribute_weight']>0:
    #     if 'image_attributes' in outputs:
    #         losses['image_attribute_loss'] = phase_config['attribute_weight'] * F.binary_cross_entropy_with_logits(
    #             outputs['image_attributes'],
    #             target_attributes
    #         )
    #
    # if target_attributes is not None and phase_config['attribute_weight']>0:
    #     if 'text_attributes' in outputs:
    #         losses['text_attribute_loss'] = phase_config['attribute_weight']*F.binary_cross_entropy_with_logits(
    #             outputs['text_attributes'],
    #             target_attributes
    #         )

    total_loss = sum(losses.values())
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer_G.step()


    losses['total_loss'] = total_loss.item()
    return {k:(v.item() if isinstance(v,torch.Tensor) else v) for k,v in losses.items()}

def validate(model, discriminator, val_loader, device, epoch, phase_config, clip_tokenizer):
    model.eval()
    discriminator.eval()
    metrics = {
        'val_image_recon_loss':0,
        'val_text_recon_loss':0,
        'val_text_from_image_loss':0,
        'val_image_from_text_loss':0,
        'val_image_kl_loss':0,
        'val_text_kl_loss':0,
        'val_distribution_matching_loss':0,
        'val_consistency_loss':0,
        'val_adversarial_loss':0,
        'val_d_loss':0,
        'val_attribute_accuracy':0,
        'val_key_attribute_accuracy':0,
        'val_total_loss':0
    }

    pad_idx = clip_tokenizer.tokenizer.pad_token_id

    num_batches=0
    caption_pred_list=[]
    caption_gt_list_tot=[]


    with torch.no_grad():
        for batch in val_loader:
            num_batches+=1

            images = batch['image'].to(device)
            if phase_config['dataset'] == 'CelebAMask-HQ':
                target_attributes = batch['attributes'].to(device)
                attention_mask=0
            elif phase_config['dataset'] == 'Flickr30k':
                text = batch["caption"]
                input_tokens = clip_tokenizer(text).to(device)
                target_attributes = input_tokens['input_ids']
                attention_mask = input_tokens['attention_mask']
                target_attributes = target_attributes.transpose(0, 1)

                pad_mask, tgt_mask = create_masks(target_attributes, pad_token=clip_tokenizer.tokenizer.pad_token_id)

                pad_idx = clip_tokenizer.tokenizer.pad_token_id




            outputs = model(False, images=images, target_attributes=target_attributes, pad_mask=pad_mask,
                                tgt_mask=tgt_mask, pad_id=pad_idx)

            if phase_config['dataset'] == 'Flickr30k':

                pred_ids = torch.argmax(outputs['recon_text_probs'], dim=-1)
                #pred_ids_text2img = torch.argmax(outputs['recon_text_probs'], dim=-1)

                pred_ids_list = pred_ids.detach().cpu().tolist()

                pred_ids_list=[list(col) for col in zip(*pred_ids_list)]

                caption_pred = [
                    clip_tokenizer.decode(ids, skip_special_tokens=True)
                    #clip_tokenizer.decode(ids)
                    for ids in pred_ids_list
                ]

            with torch.no_grad():
                target_feats = model.vgg(images)
            recon_feats = model.vgg(outputs['recon_images'])

            mse_loss = F.mse_loss(outputs['recon_images'], images)
            perceptual_loss_val = F.mse_loss(recon_feats, target_feats)
            identity_loss = identity_consistency_loss(
                model.image_encoder(images)[0],
                model.image_encoder(outputs['recon_images'])[0]
            )
            metrics['val_image_recon_loss'] += phase_config['reconstruction_weight']*(mse_loss+0.05*perceptual_loss_val+0.1*identity_loss).item()


            if phase_config['dataset'] == 'CelebAMask-HQ':

                text_attr_loss = F.binary_cross_entropy(
                    outputs['recon_text_probs'],
                    target_attributes
                )

            elif phase_config['dataset'] == 'Flickr30k':

                # print(outputs['embedded_tokens'].size())
                # print(outputs['recon_text_probs'].size())

                text_attr_loss = F.cross_entropy(
                    outputs['recon_text_probs'][:-1,:,:].view(-1, outputs['recon_text_probs'].size(-1)),
                    target_attributes[1:,:].flatten(), ignore_index=pad_idx
                )


            metrics['val_text_recon_loss'] = phase_config['text_reconstruction_weight'] * text_attr_loss

            if phase_config['dataset'] == 'CelebAMask-HQ':
                pred_attributes = (outputs['recon_text_probs']>0.5).float()
                accuracy = (pred_attributes == target_attributes).float().mean()
                metrics['val_attribute_accuracy'] += accuracy.item()

            if 'text_from_image_probs' in outputs:

                if phase_config['dataset'] == 'CelebAMask-HQ':
                    img2text_attr_loss = F.binary_cross_entropy(
                        outputs['text_from_image_probs'],
                        target_attributes
                    )

                elif phase_config['dataset'] == 'Flickr30k':
                    img2text_attr_loss = F.cross_entropy(
                        outputs['text_from_image_probs'][:-1,:,:].view(-1, outputs['text_from_image_probs'].size(-1)),
                        target_attributes[1:,:].flatten(),  ignore_index=pad_idx
                    )


                metrics['val_text_from_image_loss'] += (phase_config['cross_modal_weight']*img2text_attr_loss).item()

            if 'image_from_text' in outputs:
                txt_target_feats = model.vgg(images)
                txt_recon_feats = model.vgg(outputs['image_from_text'])
                image_from_text_loss = phase_config['cross_modal_weight'] * (
                    F.mse_loss(outputs['image_from_text'],images) +
                    0.1 * F.mse_loss(txt_recon_feats, txt_target_feats) +
                    0.1 * identity_consistency_loss(
                        model.image_encoder(images)[0],
                        model.image_encoder(outputs['image_from_text'])[0]
                    )
                )
                metrics['val_image_from_text_loss'] += image_from_text_loss.item()

            if all(k in outputs for k in ['image_mu','image_log_var']):
                image_kl = -0.5*torch.mean(
                    1+outputs['image_log_var']-outputs['image_mu'].pow(2)-outputs['image_log_var'].exp()
                )
                metrics['val_image_kl_loss'] += (phase_config['kl_weight']*image_kl).item()

            if all(k in outputs for k in ['text_mu','text_log_var']):
                text_kl = -0.5*torch.mean(
                    1+outputs['text_log_var']-outputs['text_mu'].pow(2)-outputs['text_log_var'].exp()
                )
                metrics['val_text_kl_loss'] += (phase_config['kl_weight']*text_kl).item()

            if all(k in outputs for k in ['image_mu','image_log_var','text_mu','text_log_var']):
                matching_loss = compute_distribution_matching_loss(
                    outputs['image_mu'], outputs['image_log_var'],
                    outputs['text_mu'], outputs['text_log_var']
                )
                metrics['val_distribution_matching_loss'] += (phase_config['consistency_weight']*matching_loss).item()

            if 'consistency_score' in outputs:
                consistency_loss = 1-outputs['consistency_score'].mean()
                metrics['val_consistency_loss'] += (phase_config['consistency_weight']*consistency_loss).item()

            if phase_config['adversarial_weight']>0:
                d_loss = discriminator_loss_eval(discriminator, images, outputs['recon_images'], phase_config)
                metrics['val_d_loss'] += d_loss.item()

                adv_loss = phase_config['adversarial_weight']*generator_loss_eval(discriminator, outputs['recon_images'])
                metrics['val_adversarial_loss'] += adv_loss.item()

            if phase_config['dataset'] == 'Flickr30k':

                # embedding_weights = F.normalize(model.text_encoder.base.text_model.embeddings.token_embedding.weight, dim=0)
                # #embedding_weights = F.normalize(model.text_encoder.base.embeddings.word_embeddings.weight,dim=0)
                # logits = torch.matmul(F.normalize(outputs['recon_text_probs'], dim=-1),embedding_weights.T)  # (batch, seq_len, vocab_size)
                # pred_ids = torch.argmax(outputs['recon_text_probs'], dim=-1)
                #
                # pred_ids_list = pred_ids.detach().cpu().tolist()
                #
                # caption_pred = [
                #     clip_tokenizer.decode(ids, skip_special_tokens=True)
                #     #clip_tokenizer.decode(ids)
                #     for ids in pred_ids_list
                # ]

                ####coco metrics
                caption_pred_list.extend([[x] for x in caption_pred])

                caption_gt_list = [[] for i in range(len(images))]

                caption_gt = tuple(text)
                for j in range(len(caption_gt)):
                    caption_gt_list[j].append(caption_gt[j])


                caption_gt_list_tot.extend(caption_gt_list)

    evaluation_metrics_dict=metrics_evaluation(num_batches, caption_pred_list, caption_gt_list_tot, caption_pred, images.size()[0])

    metrics = {k:v/num_batches for k,v in metrics.items()}
    metrics['val_total_loss'] = sum(v for k,v in metrics.items() if k not in ['val_attribute_accuracy','val_key_attribute_accuracy'])
    metrics.update(evaluation_metrics_dict)

    return metrics

def metrics_evaluation(batch_idx, caption_pred_list, caption_gt_list_tot, caption_pred, batch_size):

    rouge = ROUGEScore()
    bleu = BLEUScore()

    temp = list(range((batch_idx) * batch_size + (len(caption_pred))))

    caption_zip = zip(temp, caption_pred_list)
    caption_pred_dict = dict(caption_zip)

    caption_gt_zip = zip(temp, caption_gt_list_tot)
    caption_gt_dict = dict(caption_gt_zip)

    scores = evaluate_all_metrics(caption_gt_dict, caption_pred_dict)
    for k, v in scores.items():
        print(k, v)

    scores_list = list(scores.values())
    bleu1_score = scores_list[0][0]
    bleu2_score = scores_list[0][1]
    bleu3_score = scores_list[0][2]
    bleu4_score = scores_list[0][3]
    meteor_score = scores_list[1]
    rouge_L_score = scores_list[2]
    cider_score = scores_list[3]
    spice_score = scores_list[4]

    caption_pred_list_ext = flatten_extend(caption_pred_list)
    caption_gt_list_ext = flatten_extend(caption_gt_list_tot)
    bleu_score = bleu(caption_pred_list_ext, caption_gt_list_ext)
    rouge_score = rouge(caption_pred_list_ext, caption_gt_list_ext)

    rouge_l = float(rouge_score['rougeL_fmeasure'])
    bleu = float(bleu_score)

    return {'bleu1_score':bleu1_score,'bleu2_score':bleu2_score, 'bleu3_score':bleu3_score, 'bleu4_score':bleu4_score,
            'meteor_score':meteor_score, 'rouge_L_score':rouge_L_score, 'cider_score':cider_score, 'spice_score':spice_score,
             'rouge_l_torchmetrics':rouge_l, 'bleu_torchmetrics':bleu}

def evaluate_model(model, discriminator, test_loader, device, epoch, phase_config, clip_tokenizer):
    model.eval()
    metrics = validate(model, discriminator, test_loader, device, epoch, phase_config, clip_tokenizer)

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)

            if phase_config['dataset'] == 'CelebAMask-HQ':
                target_attributes = batch['attributes'].to(device)
            elif phase_config['dataset'] == 'Flickr30k':
                text = batch["caption"]
                input_tokens = clip_tokenizer(text).to(device)
                target_attributes = input_tokens['input_ids']
                target_attributes=target_attributes.transpose(0, 1)

                pad_mask, tgt_mask = create_masks(target_attributes, pad_token=clip_tokenizer.tokenizer.pad_token_id)

                pad_idx = clip_tokenizer.tokenizer.pad_token_id

            outputs = model(False, images=images, target_attributes=target_attributes, pad_mask=pad_mask,
                            tgt_mask=tgt_mask, pad_id=pad_idx)


            if phase_config['dataset'] == 'CelebAMask-HQ':
                pred_attributes = (outputs['text_from_image_probs']>0.5).float()
                per_attr_acc = (pred_attributes == target_attributes).float().mean(dim=0)
                print("\nPer-Attribute Accuracy:")
                for idx, acc in enumerate(per_attr_acc):
                    attr_name = model.idx_to_attribute[idx].replace('_',' ')
                    print(f"{attr_name}: {acc.item():.4f}")
                break
    return metrics

def train_phase_1(model, discriminator, clip_tokenizer, train_loader, val_loader, optimizer_G, optimizer_D, device, num_epochs, trainer,
                  config, val_subset, start_epoch, kl_coef):
    print("\nStarting Phase 1: Early Training")
    phase_config = config['phase_configs'][1]
    phase_config['dataset']=config['dataset']
    best_val_loss = float('inf')
    best_epoch=0

    if num_epochs>0:
        for epoch in range(start_epoch,num_epochs):
            model.train()
            discriminator.train()
            train_metrics=defaultdict(float)
            for batch in tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch+1}/{num_epochs}"):
                metrics = train_step(model,discriminator,batch,optimizer_G,optimizer_D,device,epoch,phase_config, clip_tokenizer)
                for k,v in metrics.items():
                    train_metrics[k]+=v
            train_metrics={k:v/len(train_loader) for k,v in train_metrics.items()}

            val_metrics=validate(model,discriminator,val_loader,device,epoch,phase_config, clip_tokenizer)
            if val_metrics['val_total_loss']<best_val_loss:
                best_val_loss=val_metrics['val_total_loss']
                best_epoch=epoch

            if (epoch+1)%config['eval_freq']==0:
                if phase_config['dataset'] == 'CelebAMask-HQ':
                    eval_metrics = evaluate_model(model, discriminator, val_loader, device, epoch + 1, phase_config,clip_tokenizer)
                from src.visualization import visualize_results
                results_dir = visualize_results(phase_config['dataset'], clip_tokenizer, kl_coef, optimizer_G.param_groups[0]['lr'],
                                                model,val_subset,epoch+1,"phase1",num_samples=config['num_vis_samples'],device=device)
                print(f"Phase 1 - Visualization results saved to {results_dir}")

            combined_metrics = {
                'epoch':epoch+1,
                'phase':1,
                'phase_epoch':epoch+1,
                'total_epochs':num_epochs,
                **train_metrics,
                **val_metrics
            }
            trainer.log_metrics(epoch,combined_metrics)

            print(f"\nPhase 1 - Epoch {epoch+1} Metrics:")
            print("\nTraining Metrics:")
            for k,v in train_metrics.items():
                print(f"  {k}: {v:.4f}")

            print("\nValidation Metrics:")
            for k,v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

        trainer.save_checkpoint(epoch=num_epochs,model=model,discriminator=discriminator,optimizer_G=optimizer_G,optimizer_D=optimizer_D,loss=val_metrics['val_total_loss'],phase=1)
    return best_val_loss,best_epoch

def train_phase_2(model, discriminator, clip_tokenizer, train_loader, val_loader, optimizer_G, optimizer_D, device, num_epochs, trainer, config, val_subset, start_epoch, kl_coef):
    print("\nStarting Phase 2: Middle Training")
    phase_config = config['phase_configs'][2]
    phase_config['dataset'] = config['dataset']

    best_val_loss=float('inf')
    best_epoch=0

    if num_epochs>0:
        for epoch in range(start_epoch,num_epochs):
            model.train()
            discriminator.train()

            train_metrics=defaultdict(float)
            for batch in tqdm(train_loader, desc=f"Phase 2 - Epoch {epoch+1}/{num_epochs}"):
                metrics = train_step(model,discriminator,batch,optimizer_G,optimizer_D,device,epoch,phase_config,clip_tokenizer)
                for k,v in metrics.items():
                    train_metrics[k]+=v
            train_metrics={k:v/len(train_loader) for k,v in train_metrics.items()}

            val_metrics=validate(model,discriminator,val_loader,device,epoch,phase_config, clip_tokenizer)
            if val_metrics['val_total_loss']<best_val_loss:
                best_val_loss=val_metrics['val_total_loss']
                best_epoch=epoch

            if (epoch+1)%config['eval_freq']==0:
                if phase_config['dataset']=='CelebAMask-HQ':
                    eval_metrics=evaluate_model(model,discriminator,val_loader,device,epoch+1,phase_config, clip_tokenizer)
                from src.visualization import visualize_results
                results_dir = visualize_results(phase_config['dataset'], clip_tokenizer, kl_coef, optimizer_G.param_groups[0]['lr'],
                                                model,val_subset,epoch+1,"phase2",num_samples=config['num_vis_samples'],device=device)
                print(f"Phase 2 - Visualization results saved to {results_dir}")

            combined_metrics={
                'epoch':epoch+1,
                'phase':2,
                'phase_epoch':epoch+1,
                'total_epochs':num_epochs,
                **train_metrics,
                **val_metrics
            }
            trainer.log_metrics(epoch,combined_metrics)

            print(f"\nPhase 2 - Epoch {epoch+1} Metrics:")
            print("\nTraining Metrics:")
            for k,v in train_metrics.items():
                print(f"  {k}: {v:.4f}")

            print("\nValidation Metrics:")
            for k,v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

        trainer.save_checkpoint(epoch=num_epochs,model=model,discriminator=discriminator,optimizer_G=optimizer_G,optimizer_D=optimizer_D,loss=val_metrics['val_total_loss'],phase=2)
    return best_val_loss,best_epoch

def train_phase_3(model, discriminator, clip_tokenizer, train_loader,  val_loader, optimizer_G, optimizer_D, device, num_epochs, trainer, config, val_subset, start_epoch, kl_coef):
    print("\nStarting Phase 3: Late Training")
    phase_config = config['phase_configs'][3]
    phase_config['dataset'] = config['dataset']

    best_val_loss=float('inf')
    best_epoch=0

    if num_epochs>0:
        for epoch in range(start_epoch,num_epochs):
            model.train()
            discriminator.train()

            train_metrics=defaultdict(float)
            for batch in tqdm(train_loader, desc=f"Phase 3 - Epoch {epoch+1}/{num_epochs}"):
                metrics = train_step(model,discriminator,batch,optimizer_G,optimizer_D,device,epoch,phase_config, clip_tokenizer)
                for k,v in metrics.items():
                    train_metrics[k]+=v
            train_metrics={k:v/len(train_loader) for k,v in train_metrics.items()}

            val_metrics=validate(model,discriminator,val_loader,device,epoch,phase_config, clip_tokenizer)
            if val_metrics['val_total_loss']<best_val_loss:
                best_val_loss=val_metrics['val_total_loss']
                best_epoch=epoch

            if (epoch+1)%config['eval_freq']==0:
                if phase_config['dataset'] == 'CelebAMask-HQ':
                    eval_metrics = evaluate_model(model, discriminator, val_loader, device, epoch + 1, phase_config, clip_tokenizer)
                from src.visualization import visualize_results
                results_dir=visualize_results(phase_config['dataset'], clip_tokenizer, kl_coef, optimizer_G.param_groups[0]['lr'],
                                              model,val_subset,epoch+1,"phase3",num_samples=config['num_vis_samples'],device=device)
                print(f"Phase 3 - Visualization results saved to {results_dir}")

            combined_metrics={
                'epoch':epoch+1,
                'phase':3,
                'phase_epoch':epoch+1,
                'total_epochs':num_epochs,
                **train_metrics,
                **val_metrics
            }
            trainer.log_metrics(epoch,combined_metrics)

            print(f"\nPhase 3 - Epoch {epoch+1} Metrics:")
            print("\nTraining Metrics:")
            for k,v in train_metrics.items():
                print(f"  {k}: {v:.4f}")

            print("\nValidation Metrics:")
            for k,v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

        trainer.save_checkpoint(epoch=num_epochs,model=model,discriminator=discriminator,optimizer_G=optimizer_G,optimizer_D=optimizer_D,loss=val_metrics['val_total_loss'],phase=3)
    return best_val_loss,best_epoch

def run_phased_training(model, discriminator, clip_tokenizer, train_loader, val_loader, optimizer_G, optimizer_D,
                       device, config, trainer, val_subset, phase1_epochs, phase2_epochs, phase3_epochs,
                       phase1_start=0, phase2_start=0, phase3_start=0, kl_coef=1):

    total_epochs=phase1_epochs+phase2_epochs+phase3_epochs
    print(f"\nStarting phased training:")
    print(f"Phase 1: {phase1_epochs} epochs")
    print(f"Phase 2: {phase2_epochs} epochs")
    print(f"Phase 3: {phase3_epochs} epochs")
    print(f"Total: {total_epochs} epochs")

    best_losses=[]

    if phase1_epochs>0:
        print(f"\nPhase 1: Early Training (Reconstruction Focus)")
        phase1_loss,phase1_best = train_phase_1(
            model,discriminator, clip_tokenizer, train_loader,val_loader,optimizer_G,optimizer_D,
            device,phase1_epochs,trainer,config,val_subset,phase1_start, kl_coef
        )
        print(f"\nPhase 1 completed. Best loss: {phase1_loss:.4f} at epoch {phase1_best+1}")
        best_losses.append(phase1_loss)

    if phase2_epochs>0:
        print(f"\nPhase 2: Middle Training (Alignment Focus)")
        phase2_loss,phase2_best = train_phase_2(
            model,discriminator, clip_tokenizer, train_loader,val_loader,optimizer_G,optimizer_D,
            device,phase2_epochs,trainer,config,val_subset,phase2_start, kl_coef
        )
        print(f"\nPhase 2 completed. Best loss: {phase2_loss:.4f} at epoch {phase2_best+1}")
        best_losses.append(phase2_loss)

    if phase3_epochs>0:
        print(f"\nPhase 3: Late Training (Refinement Focus)")
        phase3_loss,phase3_best = train_phase_3(
            model,discriminator, clip_tokenizer, train_loader,val_loader,optimizer_G,optimizer_D,
            device,phase3_epochs,trainer,config,val_subset,phase3_start, kl_coef
        )
        print(f"\nPhase 3 completed. Best loss: {phase3_loss:.4f} at epoch {phase3_best+1}")
        best_losses.append(phase3_loss)

    return min(best_losses) if best_losses else float('inf')
