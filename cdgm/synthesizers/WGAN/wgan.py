import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, ReLU, Sequential, functional
from torch.utils.data import Dataset, DataLoader

from cdgm.constraints_code.correct_predictions import correct_preds
from cdgm.data_processors.wgan.tab_scaler import TabScaler
from cdgm.synthesizers.utils import get_sets_constraints, adversarial_loss, round_func_BPDA

# from helpers.eval import  eval
warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)
import wandb
import matplotlib.pyplot as plt

class SingleTaskDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self): 
        return self.data.shape[0]

    def __getitem__(self,idx):
        data_i = self.data[idx]
        return data_i

class Discriminator(nn.Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, pac):
        super(Discriminator, self).__init__()
        discriminator_dim = (256,256)
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)


    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        return self.seq(input_.view(-1, self.pacdim))


    def gradient_penalty(self, real_data, fake_data,  lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // self.pac, 1, 1)
        alpha = alpha.repeat(1, self.pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients_view = gradients.view(-1, self.pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_
        return gradient_penalty



class Residual(nn.Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(nn.Module):
    """Generator for the CTGAN."""

    def __init__(self, args, data_dim,  scaler,  cat_idx):
        super(Generator, self).__init__()
        self.args = args
        self.constraints, self.sets_of_constr, self.ordering = get_sets_constraints("wgan", args.use_case, args.label_ordering, args.constraints_file)
        self.version = args.version
        self.scaler = scaler
        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()
        self.input_length = data_dim
        self.cat_idx = cat_idx
        self.args = args
        generator_dim = (256, 256)
        dim = 128
        dim = data_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        x = data.clone()
        scaled = []
        scaled.append(torch.sigmoid(data[:, :len(self.scaler.num_idx)]))

        st = len(self.scaler.num_idx)
        for i, cat_col in enumerate(self.cat_idx):
            end = st + self.scaler.ohe.categories_[i].shape[0]
            scaled.append(functional.gumbel_softmax(data[:, st:end], tau=0.2, hard=True))
            #scaled.append(round_func_BPDA(torch.softmax(data[:, st:end], dim=1)))
            st = end
        scaled = torch.cat(scaled, dim=1)
        if self.version == "constrained":
            if self.training:
                inverse = self.scaler.inverse_transform(scaled)
                for i in range(inverse.shape[1]):
                    inverse[:,i] = round_func_BPDA(inverse[:,i], self.args.round_decs[i])
                cons_layer = correct_preds(inverse, self.ordering, self.sets_of_constr)
                output_cons = self.scaler.transform(cons_layer)

                ## Do not pass the gradient for last cat column
                target_span = self.scaler.ohe.categories_[-1].shape[0]
                output_cons[:, -target_span:] = scaled[:, -target_span:]
                return output_cons
            else:
                return scaled
        else:
            return scaled


class WGAN():
    def __init__(self, args, generator, discriminator, scaler):
        self.args = args
        self.discriminator = discriminator
        self.generator = generator
        self.scaler = scaler
        self.use_case = args.use_case
        self.clamp = args.clamp
        self.pert_scale = args.pert_scale
        self.adv_scale = args.adv_scale
        self.discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=args.d_lr, alpha=args.alpha, momentum=args.momentum, weight_decay=args.weight_decay)
        self.generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.g_lr, alpha=args.alpha, momentum=args.momentum, weight_decay=args.weight_decay)


    
    def train_discriminator(self, true_data,  generated_data, gp_weight):
        with torch.autograd.set_detect_anomaly(True):
        ## Train Discriminator in real and synthetic data
            self.discriminator_optimizer.zero_grad()
            d_real_loss = torch.mean(self.discriminator(true_data))
            d_syn_loss = torch.mean(self.discriminator(generated_data.detach()))
            gp = self.discriminator.gradient_penalty(true_data, generated_data, gp_weight)
            gp.backward(retain_graph=True)
            discriminator_loss = d_syn_loss - d_real_loss 
            discriminator_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip)  # TODO: add args.clip (different from clamp!)
            self.discriminator_optimizer.step()
        return d_real_loss, d_syn_loss, discriminator_loss


    def train_generator(self, target_model, target_scaler, shape, true_data, num_labels):
        with torch.autograd.set_detect_anomaly(True):
            self.generator_optimizer.zero_grad()
            #noise = torch.rand(size=(shape[0], shape[1])).float()
            generated_data = self.generator(true_data)
            discriminator_out = self.discriminator(generated_data)
            generator_loss = -torch.mean(discriminator_out)  # + torch.nn.functional.l1_loss(generated_data, generated_data[:, torch.randperm(generated_data.shape[1])])
            generator_loss.backward(retain_graph=True)

            weights = F.softmax(torch.randn(2), dim=-1) 

            perturbation = generated_data - true_data
            pert_loss = torch.mean(torch.norm(perturbation[:,:-1], 2, dim=1))
            # pert_loss_w = pert_loss*weights[0]
 
            inverse_adv = self.scaler.inverse_transform(generated_data)
            true_data_inv = self.scaler.inverse_transform(true_data)
            probs = target_model.get_logits(inverse_adv[:,:-1], with_grad=True)
            adv_loss = adversarial_loss(probs, true_data_inv[:, -1].long(), num_labels, False)
            # adv_loss_w = adv_loss*weights[1]

            #print("Adv_loss", adv_loss)
            gen_adv_loss = self.pert_scale*weights[0]*pert_loss + self.adv_scale*weights[1]*adv_loss
            gen_adv_loss.backward()
            self.generator_optimizer.step()
        return generator_loss, pert_loss, adv_loss
    

    def get_not_modif_idx(self, not_modifiable):
        not_modifiable_trans = []
        if not_modifiable:
            for idx in not_modifiable:
                if idx in self.scaler.num_idx:
                    not_modifiable_trans.append(self.scaler.num_idx.index(idx))
                elif idx in self.scaler.cat_idx:
                    cat_st = len(self.scaler.num_idx)
                    if self.scaler.one_hot_encode:
                        for i, cat in enumerate(self.scaler.cat_idx):
                            span = len(self.scaler.ohe.categories_[i])
                            if idx == cat:
                                not_modifiable_trans.extend(np.arange(cat_st, cat_st + span).tolist())
                            cat_st = cat_st + span
                    else:
                        not_modifiable_trans.append(cat_st + self.scaler.cat_idx.index(idx))
        return not_modifiable_trans
    

    def train_step(self, target_model, target_scaler, true_data, disc_repeats=1, gp_weight=1, num_labels=0, not_modifiable_trans=[]):

        mean_d, mean_d_syn, mean_d_real  = 0, 0, 0
        for i in range(disc_repeats):
            # clamp parameters to a cube # https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
            if self.clamp is not None:
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clamp, self.clamp)

            #noise = torch.rand(size=(true_data.shape[0], true_data.shape[1])).float()
            generated_data = self.generator(true_data)
            if not_modifiable_trans:
                generated_data[:,not_modifiable_trans] = true_data[:,not_modifiable_trans]
                # generated_data_inv = self.scaler.inverse_transform(generated_data)
                # real_data_inv = self.scaler.inverse_transform(true_data)
                # generated_data_inv[:, not_modifiable] = real_data_inv[:, not_modifiable]
                # generated_data = self.scaler.transform(generated_data_inv)
            d_real_loss, d_syn_loss, discriminator_loss = self.train_discriminator(true_data, generated_data, gp_weight)
            mean_d_syn += d_syn_loss
            mean_d_real += d_real_loss
            mean_d += discriminator_loss
            # wandb.log({'steps/1step_disc_real': d_real_loss, 'steps/1step_disc_syn': d_syn_loss, 'steps/1step_disc': discriminator_loss})
        generator_loss, pert_loss, adv_loss = self.train_generator(target_model, target_scaler, (true_data.shape[0],true_data.shape[1]), true_data, num_labels)

        loss_d_syn = mean_d_syn/disc_repeats
        loss_d_real = mean_d_real/disc_repeats
        loss_d = mean_d/disc_repeats
        # wandb.log({'steps/gen_loss': generator_loss, 'steps/disc_loss': loss_d})
        return generator_loss.item(), pert_loss.item(), adv_loss.item(), loss_d_syn.item(), loss_d_real.item(),  loss_d.item()



def prepare_data_torch_scaling(train_data, cat_idx):
    train_data = train_data.to_numpy()
    scaler = TabScaler(one_hot_encode=True)
    scaler.fit(train_data, cat_idx = cat_idx)
    #joblib.dump(scaler, f"WGAN_out/{use_case}/{use_case}_torch_scaler.joblib")
    train_data = scaler.transform(train_data)
    return train_data, scaler



def train_model(args, target_model, target_scaler, cat_idx, not_modifiable, path_name, train_data, num_labels):
    #num_labels = train_data.iloc[:,-1].nunique()
    train_data, scaler = prepare_data_torch_scaling(train_data, cat_idx)
    args.input_length = train_data.shape[1]

    train_ds = SingleTaskDataset(train_data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)

    # Models
    generator = Generator(args, args.input_length, scaler, cat_idx)
    discriminator = Discriminator(args.input_length, args.pac)
    gan = WGAN(args, generator, discriminator, scaler)
    gan.not_modifiable = not_modifiable
    if gan.not_modifiable:
        not_modifiable_trans = gan.get_not_modif_idx(gan.not_modifiable)
    else:
        not_modifiable_trans = []
    for epoch in range(args.epochs):
        loss_g_running, pert_loss_running, adv_loss_running, loss_d_syn_running, loss_d_real_running, loss_d_running = 0, 0, 0, 0, 0, 0
        for i, data in enumerate(train_loader):
            real_data = data.float()
            generator_loss, pert_loss_syn, adv_loss_syn, loss_d_syn, loss_d_real,  loss_d = gan.train_step(target_model, target_scaler, real_data, args.disc_repeats, args.gp_weight, num_labels, not_modifiable_trans)
            loss_g_running += generator_loss
            pert_loss_running += pert_loss_syn
            adv_loss_running += adv_loss_syn
            loss_d_syn_running += loss_d_syn
            loss_d_real_running += loss_d_real
            loss_d_running += loss_d 

        wandb.log({'epochs/epoch': epoch, 
                   'epochs/loss_gen': loss_g_running/len(train_loader), 
                   'epochs/loss_pert': pert_loss_running/len(train_loader),
                   'epochs/loss_adv': adv_loss_running/len(train_loader), 
                   'epochs/loss_disc_syn': loss_d_syn_running/len(train_loader), 
                   'epochs/loss_disc_real': loss_d_real_running/len(train_loader), 
                   'epochs/loss_disc': loss_d_running/len(train_loader)})
        print('Epoch {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch, loss_d_running/len(train_loader), loss_g_running/len(train_loader)))
        print('Epoch {}: perturbation_loss {:.3f} adversarial_loss {:.3f}'.format(epoch, pert_loss_running/len(train_loader), adv_loss_running/len(train_loader)))

        print("Discriminator real {}, fake {}".format(loss_d_real_running/len(train_loader), loss_d_syn_running/len(train_loader)))


    PATH = f"{path_name}/wgan_model.pt"
    torch.save(gan, PATH)

    return gan



def sample(wgan, adv_cand_init):
    """Sample data similar to the training data

    Args:
        n (int):
            Number of rows to sample.
    Returns:
        numpy.ndarray or pandas.DataFrame
    """
    if isinstance(adv_cand_init, pd.DataFrame):
        adv_cand_init = torch.tensor(adv_cand_init.values.astype('float32'))
    if isinstance(adv_cand_init, np.ndarray):
        adv_cand_init = torch.tensor(adv_cand_init.astype('float32'))
    adv_cand = wgan.scaler.transform(adv_cand_init)
    wgan.generator.eval()
    #noise = torch.rand(size=(n, input_length)).float()
    with torch.no_grad():
        generated_data = wgan.generator(adv_cand) # it always returns the unconstrained, scaled (so before inverse is applied, even if version was constrained)
    generated_data = wgan.scaler.inverse_transform(generated_data)
    if wgan.args.version == "postprocessing" or wgan.args.version == 'constrained':
        generated_data = correct_preds(generated_data, wgan.generator.ordering, wgan.generator.sets_of_constr)
    if wgan.not_modifiable:
        generated_data[:,wgan.not_modifiable] = adv_cand_init[:,wgan.not_modifiable]
    for i in range(generated_data.shape[1]):
        generated_data[:,i] = round_func_BPDA(generated_data[:,i], wgan.args.round_decs[i])
    sampled_data = generated_data.detach().numpy()[:,:-1]
    return sampled_data