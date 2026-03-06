from pyexpat import features

import torch
import torch.nn as nn
import torch.nn.functional as F
#from numpy.setup import configuration
from torchvision.models import vgg16, VGG16_Weights

from .components import ResidualBlock, ResidualLinear
from ..training.losses import identity_consistency_loss, compute_distribution_matching_loss
from ..data.utils import clean_and_validate_attributes, generate_natural_description

from transformers import AutoModel

from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPVisionModelWithProjection

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torchvision.models as models
import math

class Word2SentenceEmbedding(nn.Module):
    def __init__(self, hdim):
        super(Word2SentenceEmbedding, self).__init__()
        self.dense = nn.Linear(hdim, hdim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # take the hidden state corresponding to <sos> token
        first_token_tensor = hidden_states[0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def shift_right(labels, pad_token_id):
    shifted = labels.new_zeros(labels.shape)

    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = pad_token_id

    return shifted


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class ImageEncoder(nn.Module):
    def __init__(self, latent_dim, dataset):
        super().__init__()

        self.dataset = dataset

        if dataset == 'CelebAMask-HQ':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Flatten(),
            )

            self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
            self.fc_var = nn.Linear(256 * 8 * 8, latent_dim)

        elif dataset == 'Flickr30k':

            #configuration = CLIPVisionConfig()
            #self.base = CLIPVisionModel(configuration).from_pretrained("openai/clip-vit-base-patch32")
            self.base = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            #self.encoder = models.resnet34(pretrained=True)
            self.encoder=nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * latent_dim)
        )
            #d_in = base.fc.in_features
            #base.fc = nn.Identity()
            #self.base = base
            #self.projection = Projection(768, 2 * latent_dim)
            for p in self.base.parameters():
                p.requires_grad = False

            self.fc_mu = nn.Linear(2*latent_dim, latent_dim)
            self.fc_var = nn.Linear(2*latent_dim, latent_dim)






    def forward(self, x):

        if self.dataset == 'CelebAMask-HQ':
            features = self.encoder(x)
        elif self.dataset == 'Flickr30k':
            x=self.base(x)
            clip_output=x.last_hidden_state
            features=self.encoder(x.image_embeds)
            # projected_vec = self.projection(x.pooler_output)
            # projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
            # features=projected_vec / projection_len
        #mu, logvar = x.chunk(2, dim=1)
        #return self.fc_mu(features), self.fc_var(features), clip_output[:,:-1,:]
        return features.chunk(2,dim=1)[0], features.chunk(2,dim=1)[1], clip_output[:,:-1,:]


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, latent_dim,e_dim, num_attributes, nheads, nlayers, pad_idx):
        super().__init__()
        hidden_dim = 512

        self.num_attributes = num_attributes

        if num_attributes==10:
            self.fc = nn.Sequential(
                nn.Linear(num_attributes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim*2)
            )
        elif num_attributes==49:
            # self.base = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
            # self.projection = Projection(768, 2*latent_dim)
            # configuration = CLIPTextConfig()
            # self.base = CLIPTextModel(configuration).from_pretrained("openai/clip-vit-base-patch32")
            # self.projection = Projection(512, 2 * latent_dim)
            # for p in self.base.parameters():
            #     p.requires_grad = False

            #self.base = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

            self.e_dim = e_dim

            self.embedding = nn.Embedding(vocab_size, e_dim, padding_idx=pad_idx)
            self.pos_encoding = PositionalEncoding(e_dim)
            encoder_layers = TransformerEncoderLayer(d_model=e_dim, nhead=nheads, dim_feedforward=4*latent_dim, dropout=0.2)
            self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
            self.word2sen_hidden = Word2SentenceEmbedding(hdim=e_dim)
            self.hid2latparams = nn.Linear(e_dim, 2 * latent_dim)
            # for p in self.base.parameters():
            #     p.requires_grad = False


        self.mu_head = nn.Linear(latent_dim*2, latent_dim)
        self.logvar_head = nn.Linear(latent_dim*2, latent_dim)

    def forward(self, attributes, pad_mask=None):

        if self.num_attributes==10:
            x = self.fc(attributes)
        elif self.num_attributes==49:
            #out = self.base(attributes, attention_mask=attention_mask)[1]
            # out = self.base(attributes, attention_mask=attention_mask)[0]
            # #out = out[:, 0, :]  # get CLS token output
            # projected_vec = self.projection(out)
            # projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
            # x=projected_vec/projection_len
            # x=x.mean(dim=1)



            embedded = self.embedding(attributes) * math.sqrt(self.e_dim)
            embedded = self.pos_encoding(embedded)
            hidden = self.transformer_encoder(embedded, src_key_padding_mask=pad_mask)
            hidden = self.word2sen_hidden(hidden)
            x = self.hid2latparams(hidden)

        # mu = self.mu_head(x)
        # logvar = self.logvar_head(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

class TextDecoder(nn.Module):
    def __init__(self, latent_dim, e_dim, num_attributes, vocab_size, device, nheads, nlayers, pad_idx):
        super().__init__()
        self.hidden_dim = 512
        self.num_attributes = num_attributes
        self.device=device
        self.num_layers=2
        self.nhead=8

        if num_attributes==10:
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, num_attributes),
                nn.Sigmoid()
            )
        elif num_attributes==49:
        #     self.fc = nn.Sequential(
        #     nn.Linear(latent_dim, 2048),
        #     nn.LeakyReLU(0.2),
        #     ResidualLinear(2048),
        #     #nn.Linear(2048, 512 * 8 * 4),
        #     nn.Linear(2048, 768 * 8 * 4),
        #     nn.LeakyReLU(0.2)
        # )
        #
            # self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
            # self.decoder_rnn = nn.GRU(self.hidden_dim + latent_dim, self.hidden_dim, batch_first=True)
            # self.output_fc=nn.Linear(self.hidden_dim, vocab_size)
            # self.pos_embedding = nn.Parameter(torch.randn(num_attributes, self.hidden_dim))
            # decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.nhead)
            # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
            # self.output_fc = nn.Linear(self.hidden_dim, vocab_size)
            #
            # self.latent_proj = nn.Linear(self.hidden_dim//2, self.hidden_dim)
            self.e_dim = e_dim

            self.embedding = nn.Embedding(vocab_size, e_dim, padding_idx=pad_idx)
            self.pos_encoding = PositionalEncoding(e_dim)
            self.lat2hid = nn.Linear(latent_dim, e_dim)
            decoder_layers = TransformerDecoderLayer(d_model=e_dim, nhead=nheads, dim_feedforward=4*latent_dim, dropout=0.2)
            self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layers, num_layers=nlayers)
            self.hid2logits = nn.Linear(e_dim, vocab_size)

            self.img2input=nn.Linear(768, e_dim)




    def forward(self, z, sentences=None, tgt_mask=None, tgt_pad_mask=None, threshold=0.5):

        if self.num_attributes == 10:
            attribute_probs = self.fc(z)
        elif self.num_attributes==49:
            # #x = attribute_probs.view(-1,512,8,4)
            #batch_size = z.size(0)
            # #x_flat = x.view(batch_size,512,-1).permute(2,0,1)
            # x_flat = x.view(batch_size, 768, -1).permute(2, 0, 1)
            # attn_out, _ = self.attention(x_flat,x_flat,x_flat)
            # x_flat = x_flat + attn_out
            # #attribute_probs = x_flat.permute(1, 2, 0).view(batch_size, 32, 512)
            # attribute_probs = x_flat.permute(1, 2, 0).view(batch_size, 32, 768)

            # if x is not None:
            #     # Use CLIP's embedding layer for input tokens
            #     x_embed = x  # [batch, seq_len, clip_hidden_dim]
            #     # Create padding mask for variable-length sequences
            #
            # else:
            #     x_embed = torch.zeros(batch_size, self.num_attributes, self.hidden_dim).to(self.device)
            #
            #
            # # Each position has a learnable vector to give the model token order information
            # x_embed = x_embed + self.pos_embedding.unsqueeze(0)  # [B, T, hidden_dim]
            #
            # # Repeat latent z for every time step
            # z_repeat = z.unsqueeze(1).repeat(1, self.num_attributes, 1)
            # z_repeat = self.latent_proj(z_repeat)
            # decoder_input = x_embed + z_repeat
            # #decoder_input = torch.cat([x_embed, z_repeat], dim=-1)
            # #out, _ = self.decoder_rnn(decoder_input)
            # out = self.transformer_decoder(
            #     decoder_input.transpose(0, 1),  # [T, B, hidden_dim]
            #     decoder_input.transpose(0, 1),  # memory same as input for non-autoregressive
            #     tgt_key_padding_mask=padding_mask  # True=ignore
            # )
            # attribute_probs = self.output_fc(out)

            if z is not None:
                memories = self.lat2hid(z)
                if memories.ndim==2:
                    memories = memories.unsqueeze(0)



            if sentences.dtype==torch.int64:
                embedded_targets = self.embedding(sentences) * math.sqrt(self.e_dim)
                embedded_targets = self.pos_encoding(embedded_targets)
            else:
                embedded_targets = self.img2input(sentences)
                embedded_targets=embedded_targets.transpose(0,1)


            hidden = self.transformer_decoder(embedded_targets, memories, tgt_mask=tgt_mask,
                                              tgt_key_padding_mask=tgt_pad_mask)
            attribute_probs = self.hid2logits(hidden)

        predicted_attributes = (attribute_probs > threshold).float()
        return attribute_probs, predicted_attributes

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim, dataset):
        super().__init__()

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LeakyReLU(0.2),
            ResidualLinear(2048),
            nn.Linear(2048, 512 * 4 * 4),
            nn.LeakyReLU(0.2)
        )

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        if dataset=='CelebAMask-HQ':

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                ResidualBlock(512),

                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                ResidualBlock(256),

                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                ResidualBlock(128),

                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                ResidualBlock(64),

                nn.Conv2d(64, 3, 3, 1, 1),
                nn.Tanh()
            )


        elif dataset== 'Flickr30k':

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                ResidualBlock(512),

                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                ResidualBlock(256),

                # nn.ConvTranspose2d(256, 256, 4, 2, 1),
                # nn.BatchNorm2d(256),
                # nn.LeakyReLU(0.2),
                # ResidualBlock(256),

                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                ResidualBlock(128),

                # nn.ConvTranspose2d(128, 128, 4, 2, 1),
                # nn.BatchNorm2d(128),
                # nn.LeakyReLU(0.2),
                # ResidualBlock(128),

                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                ResidualBlock(64),

                nn.Conv2d(64, 3, 3, 1, 1),
                nn.AdaptiveAvgPool2d((224, 224)),
                nn.Tanh()
            )


    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1,512,4,4)

        batch_size = x.size(0)
        x_flat = x.view(batch_size,512,-1).permute(2,0,1)
        attn_out, _ = self.attention(x_flat,x_flat,x_flat)
        x_flat = x_flat + attn_out
        x = x_flat.permute(1,2,0).view(batch_size,512,4,4)

        return self.decoder(x)

class MultimodalVAE(nn.Module):
    def __init__(self, device,tokenizer, vocab_size, dataset, latent_dim=32, e_dim=512, nheads=8, nlayers=4, pad_token_id=0, num_attributes=10,  temperature=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_attributes = num_attributes
        self.temperature = temperature

        self.dataset=dataset
        self.device = device
        self.vocab_size=tokenizer.tokenizer.vocab_size
        self.tokenizer = tokenizer

        self.image_encoder = ImageEncoder(latent_dim,dataset)
        self.text_encoder = TextEncoder(vocab_size, latent_dim, e_dim, num_attributes, nheads, nlayers, pad_token_id)
        self.image_decoder = ImageDecoder(latent_dim, dataset)
        self.text_decoder = TextDecoder(latent_dim, e_dim, num_attributes, self.vocab_size, self.device, nheads, nlayers, pad_token_id)

        self.norm_layer = nn.LayerNorm(latent_dim)



        self.idx_to_attribute = {
            0: 'young',
            1: 'male',
            2: 'female',
            3: 'smiling',
            4: 'eyeglasses',
            5: 'black_hair',
            6: 'blond_hair',
            7: 'bald',
            8: 'mustache',
            9: 'wearing_lipstick'
        }

        # Load VGG features for perceptual loss if needed
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False

        # if dataset=='Flickr30k':
        #     self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=latent_dim)



    def _attributes_to_text(self, predicted_attributes):
        batch_descriptions = []
        for attr_vector in predicted_attributes:
            active_attributes = [
                self.idx_to_attribute[idx].replace('_',' ')
                for idx, is_active in enumerate(attr_vector)
                if is_active == 1
            ]
            description = generate_natural_description(active_attributes)
            batch_descriptions.append(description)
        return batch_descriptions

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std * self.temperature
        return mu

    def encode_image(self, images):
        mu, log_var, clip_output = self.image_encoder(images)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var, clip_output

    def encode_text(self, attributes, pad_mask):
        mu, log_var = self.text_encoder(attributes,  pad_mask)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def decode_image(self, z):
        return self.image_decoder(z)

    # def decode_text(self, z, generation_params=None):
    #     if generation_params is None:
    #         generation_params = {'threshold':0.5}
    #     attr_probs, pred_attributes = self.text_decoder(z, **generation_params)
    #     descriptions = self._attributes_to_text(pred_attributes)
    #     return attr_probs, pred_attributes, descriptions

    def forward(self, train_phase, images=None, target_attributes=None, pad_mask=None, tgt_mask=None, pad_id=0):
        outputs = {}


        # target_attributes_emb = self.text_encoder.base.text_model.embeddings.token_embedding(target_attributes)
        # outputs['embedded_tokens'] = target_attributes_emb
        # target_attributes_s = shift_right(
        #     target_attributes,
        #     pad_token_id=pad_id
        # )

        if images is not None:
            z_image, image_mu, image_log_var, clip_output = self.encode_image(images)
            outputs['image_mu'] = image_mu
            outputs['image_log_var'] = image_log_var
            outputs['z_image'] = z_image
            outputs['recon_images'] = self.decode_image(z_image)



            attr_probs_img, pred_attrs_img = self.text_decoder(z_image, clip_output)
            # if train_phase == True:
            #     #padding_mask = (target_attributes == self.tokenizer.tokenizer.pad_token_id)  # [B, T]
            #     attr_probs_img, pred_attrs_img = self.text_decoder(z_image,  target_attributes_s, tgt_mask, pad_mask)
            # else:
            #     attr_probs_img, pred_attrs_img = self.text_decoder(z_image, target_attributes_s, tgt_mask, pad_mask)



            outputs['text_from_image_probs'] = attr_probs_img

        if target_attributes is not None:
            if  self.dataset=='Flickr30k':
                #target_attributes_emb=self.embedding(target_attributes)
                #target_attributes_emb = self.text_encoder.base.embeddings.word_embeddings(target_attributes)




                z_text, text_mu, text_log_var = self.encode_text(target_attributes, pad_mask)

                outputs['text_mu'] = text_mu
                outputs['text_log_var'] = text_log_var


            elif self.dataset=='CelebAMask-HQ':
                z_text, text_mu, text_log_var = self.encode_text(target_attributes,  pad_mask)

                outputs['text_mu'] = text_mu
                outputs['text_log_var'] = text_log_var


            outputs['z_text'] = z_text.transpose(0,1)

            attr_probs, pred_attributes = self.text_decoder(z_text, target_attributes, tgt_mask, pad_mask.type(torch.float))
            # if train_phase == True:
            #     #padding_mask = (target_attributes == self.tokenizer.tokenizer.pad_token_id)  # [B, T]
            #     attr_probs, pred_attributes = self.text_decoder(z_text,  target_attributes_s, tgt_mask, pad_mask)
            # else:
            #     attr_probs, pred_attributes = self.text_decoder(z_text, target_attributes_s, tgt_mask, pad_mask)



            outputs['recon_text_probs'] = attr_probs
            #outputs['recon_text_attributes'] = pred_attributes
            #outputs['recon_text'] = self._attributes_to_text(pred_attributes)

            #z_t2i=z_text.mean(dim=0).squeeze(0)

            image_from_text = self.decode_image(z_text)


            outputs['image_from_text'] = image_from_text


        if images is not None and target_attributes is not None:
            kl_match = torch.mean(0.5 * (
                torch.exp(outputs['text_log_var']) / torch.exp(outputs['image_log_var']) +
                (outputs['image_mu'] - outputs['text_mu'])**2 / torch.exp(outputs['image_log_var']) -
                1 + outputs['image_log_var'] - outputs['text_log_var']
            ))
            outputs['consistency_score'] = 1 / (1 + kl_match)

        return outputs

    def check_consistency(self, z1_mu, z1_logvar, z2_mu, z2_logvar):
        kl_match = torch.mean(0.5 * (
            torch.exp(z2_logvar) / torch.exp(z1_logvar) +
            (z1_mu - z2_mu)**2 / torch.exp(z1_logvar) -
            1 + z1_logvar - z2_logvar
        ))
        return 1 / (1 + kl_match)

    # @torch.no_grad()
    def generate_from_text(self, attributes, pad_mask):
        device = next(self.parameters()).device
        attributes_t = attributes.to(device)
        z_text, _, _ = self.encode_text(attributes_t, pad_mask)
        #z_t2i = z_text.mean(dim=0).squeeze(0)
        return self.decode_image(z_text)

    @torch.no_grad()
    def generate_from_image(self, image, dataset, tokenizer):
        device = next(self.parameters()).device
        image = image.to(device)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        z_image, _, _, clip_output = self.encode_image(image)
        attribute_probs, pred_attributes = self.text_decoder(z_image, clip_output)

        if dataset == 'Flickr30k':
            #embedding_weights = F.normalize(self.text_encoder.base.text_model.embeddings.token_embedding.weight, dim=0)
            #embedding_weights = F.normalize(self.model.text_encoder.base.embeddings.word_embeddings.weight, dim=0)
            #logits = torch.matmul(F.normalize(attribute_probs, dim=-1), embedding_weights.T)  # (batch, seq_len, vocab_size)
            pred_ids = torch.argmax(attribute_probs, dim=-1)

            pred_ids_list = pred_ids.detach().cpu().tolist()

            descriptions = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in pred_ids_list
            ]
        elif dataset=='CelebAMask-HQ':
            descriptions = self._attributes_to_text(pred_attributes)
            if len(descriptions) == 1:
                return descriptions[0]

        return descriptions

    def sample_latent(self, batch_size=1):
        device = next(self.parameters()).device
        return torch.randn(batch_size, self.latent_dim).to(device)

    def interpolate_latent(self, z1, z2, steps=10):
        alphas = torch.linspace(0,1,steps,device=z1.device)
        z_interp = torch.zeros(steps,self.latent_dim,device=z1.device)
        for i,alpha in enumerate(alphas):
            z_interp[i] = (1-alpha)*z1 + alpha*z2
        return z_interp

    # def fuse_representations(self, image, target_attributes, fusion_weight=0.5):
    #     z_image, image_mu, _ = self.encode_image(image)
    #     z_text, text_mu, _ = self.encode_text(target_attributes)
    #
    #     z_image_norm = self.norm_layer(z_image)
    #     z_text_norm = self.norm_layer(z_text)
    #
    #     attention = torch.sigmoid(torch.sum(z_image_norm * z_text_norm, dim=-1, keepdim=True))
    #
    #     z_fused = (1 - fusion_weight) * (z_image_norm + attention * z_image_norm) + \
    #               fusion_weight * (z_text_norm + (1 - attention) * z_text_norm)
    #
    #     z_fused = z_fused + 0.1 * z_image_norm
    #
    #     return self.decode_image(z_fused)