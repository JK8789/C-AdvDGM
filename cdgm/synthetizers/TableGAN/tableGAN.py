#!/usr/bin/env python
# coding: utf-8

### Commit 37a217f
#https://github.com/sdv-dev/SDGym/commit/37a217f9bbd5ec7cd09b33b9c067566019caceb9
import numpy as np
import torch
import wandb
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.nn import functional as F
from cdgm.synthetizers.TableGAN.base_tableGAN import LegacySingleTableBaseline
from cdgm.data_processors.wgan.tab_scaler import TabScaler
from cdgm.constraints_code.correct_predictions import correct_preds
from cdgm.synthetizers.TableGAN.blocks import Generator, Discriminator
from cdgm.synthetizers.TableGAN.blocks import determine_layers, get_optimizer, weights_init
from cdgm.synthetizers.pcgrad import PCGrad
from cdgm.synthetizers.utils import get_sets_constraints, adversarial_loss, round_func_BPDA


def _apply_constrained(self, fake):
    
    if self.args.version == "constrained":
        fake_re = fake.reshape(-1, self.random_dim)
        fake_re = fake_re[:, :self.col_len]
        inverse = self.transformer.inverse_transform(fake_re)
        # if not self.generator.training:
        #     for i in range(inverse.shape[1]):
        #         inverse[:,i] = round_func_BPDA(inverse[:,i], self.args.round_decs[i])
        fake_cons = correct_preds(inverse, self.ordering, self.sets_of_constr)
        if self.generator.training:
            fake_cons = self.transformer.transform(fake_cons)
            fake_cons = self.add_padding(fake_cons)
            fake_cons = fake_cons.reshape(-1, 1, self.side, self.side)
    else:
        fake_cons = fake.clone()
    return fake_cons

def _apply_constrained_sample(self, fake):
    fake_re = fake.reshape(-1, self.random_dim)
    fake_re = fake_re[:, :self.col_len]
    inverse = self.transformer.inverse_transform(fake_re)
    if self.args.version == "constrained" or self.args.version== "postprocessing":
        inverse = correct_preds(inverse, self.ordering, self.sets_of_constr)
    for i in range(inverse.shape[1]):
        inverse[:,i] = round_func_BPDA(inverse[:,i], self.args.round_decs[i])
    return inverse

def get_not_modif_idx(not_modifiable, scaler):
    not_modifiable_trans = []
    if not_modifiable:
        for idx in not_modifiable:
            if idx in scaler.num_idx:
                not_modifiable_trans.append(scaler.num_idx.index(idx))
            elif idx in scaler.cat_idx:
                cat_st = len(scaler.num_idx)
                if scaler.one_hot_encode:
                    for i, cat in enumerate(scaler.cat_idx):
                        span = len(scaler.ohe.categories_[i])
                        if idx == cat:
                            not_modifiable_trans.extend(np.arange(cat_st, cat_st + span).tolist())
                        cat_st = cat_st + span
                else:
                    not_modifiable_trans.append(cat_st + scaler.cat_idx.index(idx))
    return not_modifiable_trans


class TableGAN(LegacySingleTableBaseline):
    """docstring for TableganSynthesizer??"""

    def __init__(self,
                 target_model, 
                 target_scaler,
                 random_dim=1,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 verbose = True,
                 epochs=300,
                 pert_scale=1,
                 adv_scale=1,
                 path="",
                 version="unconstrained",
                 not_modifiable=[]):
        #self.random_dim = 1
        self.target_model = target_model
        self.target_scaler = target_scaler
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale
        self._verbose = verbose
        self.batch_size = batch_size
        self.epochs = epochs
        self.pert_scale = pert_scale
        self.adv_scale = adv_scale
        self.version = version
        self.path = path
        self.device = 'cpu' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.not_modifiable = not_modifiable


    def get_side(self, train_data):
        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= train_data.shape[1]:
                return i
        
    def add_padding(self, data):
        if self.side * self.side > len(data[0]):
            padding = torch.zeros((len(data), self.side * self.side - len(data[0])))
            data = torch.concat([data, padding], axis=1)
        return data
    
    def fit(self, args, train_data, discrete_columns_idx, num_labels):
        self.args = args
        self.constraints, self.sets_of_constr, self.ordering = get_sets_constraints("tablegan", args.use_case, args.label_ordering, args.constraints_file)
        self.side = self.get_side(train_data)
        self.random_dim = self.side*self.side
        self.transformer = TabScaler(out_min=-1.0, out_max=1.0, one_hot_encode=False)
        self.col_len = train_data.shape[1] 
        self.not_modif_idx = get_not_modif_idx(self.not_modifiable, self.transformer)
        train_data = torch.from_numpy(train_data.values.astype('float32')).to(self.device)
        self.transformer.fit(train_data[:,:-1])
        data = self.transformer.transform(train_data[:,:-1])
        data = self.add_padding(data)
        data_init = data.reshape(-1, 1, self.side, self.side)
        dataset = TensorDataset(data_init, train_data[:,-1])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        layers_D, layers_G, layers_C = determine_layers(
            self.side, self.random_dim, self.num_channels)
        self.generator = Generator(None, self.side, layers_G).to(self.device)
        discriminator = Discriminator(None, self.side, layers_D).to(self.device)

        optimizerG, optimizerD = get_optimizer(self, args, discriminator)
        # factor = 2
        # schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lambda epoch: (epoch+1) * factor, last_epoch=- 1, verbose=False)
        # schedulerG = lr_scheduler.CosineAnnealingLR(optimizerG, T_max = 12, eta_min = 1e-5)
        # schedulerD = lr_scheduler.CosineAnnealingLR(optimizerD,  T_max = 12, eta_min = 1e-5)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)
        pc_adam = PCGrad(optimizerG)
        # adv_decay  = 1.004
        # pert_decay = 1
        for epoch in range(self.epochs):
            loss_g_running,  loss_d_running, pert_loss_running, adv_loss_running = 0, 0, 0, 0

            for id_, data in tqdm(enumerate(loader), total=len(loader)):
                real = data[0].to(self.device)
                labels = data[1]

                real_gen = real.reshape(self.batch_size, self.random_dim, 1, 1)
                adv_gen = self.generator(real_gen)
                fake_cons = _apply_constrained(self, adv_gen)
                fake_cons = fake_cons.reshape(-1, self.random_dim)
                real_random = real.clone().reshape(-1, self.random_dim)
                if self.not_modif_idx:
                    fake_cons[:,self.not_modif_idx] = real_random[:,self.not_modif_idx]
                fake_cons = fake_cons.reshape(-1, 1, self.side, self.side)
                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake_cons.detach())
                loss_d = (-(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                loss_d.backward()
                optimizerD.step()

                ######################## Train GENERATOR ###############################

                adv_gen = self.generator(real_gen)
                fake_cons = _apply_constrained(self, adv_gen)
                fake_cons = fake_cons.reshape(-1, self.random_dim)
                if self.not_modif_idx:
                    fake_cons[:,self.not_modif_idx] = real[:,self.not_modif_idx]
                
                # perturbation = real - fake_cons
                # fake_cons = fake_cons.reshape(-1, 1, self.side, self.side)
                # real = real.reshape(-1, 1, self.side, self.side)
               
                optimizerG.zero_grad()
                # pc_adam.zero_grad()

                weights = F.softmax(torch.randn(2), dim=-1) # RLW is only this!
                # target = torch.where(labels==1)[0]
                perturbation = (real_gen.squeeze() - fake_cons)
                print("Pert",perturbation.shape)
                pert_loss = torch.mean(torch.norm(perturbation[:, :self.col_len-1], 2, dim=-1))
           
                fake_re = fake_cons.reshape(-1, self.random_dim)[:, :self.col_len]
                inverse_adv = self.transformer.inverse_transform(fake_re)
                probs = self.target_model.get_logits(inverse_adv[:,:-1], True)
                adv_loss = adversarial_loss(probs, labels.long(), num_labels, False)
                adv_loss_w = adv_loss*weights[1]

                # real_scaled = self.transformer.inverse_transform(real_random[:, :self.col_len-1])
                # inverse_adv_scaled = self.target_scaler.transform(fake_re[:,:-1])
                # real_scaled = self.target_scaler.transform(real_scaled)
                # pert_loss = torch.mean(torch.norm(real_scaled - inverse_adv_scaled, 2, dim=-1))
                pert_loss_w = pert_loss*weights[0]

                gen_adv_loss = self.pert_scale*weights[0]*pert_loss + self.adv_scale*weights[1]*adv_loss
                gen_adv_loss.backward(retain_graph=True)

                fake_cons = fake_cons.reshape(-1, 1, self.side, self.side)
                y_fake = discriminator(fake_cons)
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True)

                
                loss_mean = torch.norm(torch.mean(fake_cons, dim=0) - torch.mean(real, dim=0), 1)
                loss_std = torch.norm(torch.std(fake_cons, dim=0) - torch.std(real, dim=0), 1)
                loss_info = loss_mean + loss_std
                loss_info.backward()
                # pc_adam.pc_backward([self.pert_scale*pert_loss, self.adv_scale*adv_loss])

                optimizerG.step()

                loss_g_running += loss_g
                loss_d_running += loss_d
                pert_loss_running += pert_loss
                adv_loss_running += adv_loss

            # schedulerG.step()
            # schedulerD.step()

            wandb.log({'epochs/epoch': epoch, 'epochs/loss_gen': loss_g_running/len(loader), 'epochs/loss_disc': loss_d_running/len(loader), 
                       'epochs/loss_pert': pert_loss_running/len(loader), 'epochs/loss_adv': adv_loss_running/len(loader)})

            if self._verbose:
                print(f'Epoch {epoch+1}, Loss G: {loss_g.detach().cpu(): .3f}, '  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .3f}, ')  
                print(f'Epoch {epoch+1}: perturbation_loss {pert_loss_running/len(loader):.3f} adversarial_loss {adv_loss_running/len(loader):.3f}')
    
            if epoch >= 5 and epoch % args.save_every_n_epochs == 0:
                torch.save(self.generator, f"{self.path}/model_{epoch}.pt")

        PATH = f"{self.path}/model_tablegan.pt"
        torch.save(self, PATH)

    def prepare_data_for_sampling(self,x):
        remainder = x.shape[0] % self.batch_size
        padded_rows = torch.zeros(self.batch_size - remainder, x.shape[1])
        x_padded = torch.concat([x, padded_rows])
        data = self.transformer.transform(x_padded[:,:-1])
        data = self.add_padding(data)
        data = data.reshape(-1, 1, self.side, self.side)
        return data, x_padded[:,-1]
    
    def sample(self, X_cand):
        self.generator.eval()
        x = torch.from_numpy(X_cand.values.astype('float32')).to(self.device)
        self.batch_size = min(self.batch_size, x.shape[0])   
        data, labels = self.prepare_data_for_sampling(x)
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.generator.eval()
        out_data = []
        for id_, data in tqdm(enumerate(loader), total=len(loader)):
            real = data[0].reshape(self.batch_size, self.random_dim, 1, 1)
            adv_gen = self.generator(real)
            inverse = _apply_constrained_sample(self, adv_gen)
            if self.not_modifiable:
                real = self.transformer.inverse_transform(real.squeeze(2).squeeze(2)[:, :self.col_len-1])
                inverse[:,self.not_modifiable[:-1]] = real[:,self.not_modifiable[:-1]]
            out_data.append(inverse.detach().cpu().numpy())
        out_data = np.concatenate(out_data, axis=0)[:X_cand.shape[0],:X_cand.shape[1]-1]
        return out_data
