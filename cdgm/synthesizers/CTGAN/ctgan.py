"""CTGAN module."""

import warnings

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import numpy as np
from cdgm.constraints_code.correct_predictions import correct_preds
from cdgm.data_processors.ctgan.data_sampler import DataSampler
from cdgm.data_processors.ctgan.data_transformer import DataTransformer
from cdgm.synthesizers.CTGAN.base_ctgan import BaseSynthesizer, random_state
from cdgm.synthesizers.CTGAN.blocks import Generator, Discriminator, define_optimizer
from cdgm.synthesizers.utils import get_discrete_col, get_not_modif_idx, get_modes_idx
from cdgm.synthesizers.utils import apply_activate, apply_constrained, get_sets_constraints, adversarial_loss, round_func_BPDA
from utils import NumpyArrayIterator
import wandb



class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, target_model, target_scaler, test_data, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=True, epochs=300, pac=10, pert_scale=1, adv_scale=1, cuda=True, path="", bin_cols_idx=[], not_modifiable=[], version="unconstrained", feats_in_constraints=[]):

        assert batch_size % 2 == 0
        self._target_model = target_model
        self._target_scaler = target_scaler
        self.feats_in_constraints = feats_in_constraints
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self._path = path
        self._bin_cols_idx = bin_cols_idx
        self._version = version
        self.target_type = "nn"
        self.pert_scale = pert_scale
        self.adv_scale = adv_scale
        self.targeted = False
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.test_data = test_data
        self.not_modifiable = not_modifiable



    
    def get_train_data(self):
        idx, real = self._data_sampler.sample_data(self._batch_size, None, None)
        real = torch.from_numpy(real.astype('float32')).to(self._device)
        return idx, real

    
    @random_state
    def fit(self, args, train_data_orig,   num_labels, discrete_columns=(),  epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.args = args
        self.constraints, self.sets_of_constr, self.ordering = get_sets_constraints("ctgan", args.use_case, args.label_ordering, args.constraints_file)

        self._transformer = DataTransformer()
        print('Start fit transformer', discrete_columns, train_data_orig.shape)
        self._transformer.fit(train_data_orig.iloc[:,:-1], discrete_columns)
        print('End fit transformer')
        train_data = self._transformer.transform(train_data_orig, None)
        print('End fit data')

        # do_not_touch = np.random.randint(train_data_orig.shape[1]-1, size=62).tolist()
        # self.not_modifiable.extend(do_not_touch)
        self.not_modif_idx = get_not_modif_idx(self._transformer, self.not_modifiable)
        self.modes_idx = get_modes_idx(self._transformer)
        self.args.modes_idx = self.modes_idx
        discrete_cols = get_discrete_col(self._transformer)
        self.args.discrete_cols = discrete_cols
        self._data_sampler = DataSampler(train_data, self._transformer.output_info_list, self._log_frequency)

        data_dim = self._transformer.output_dimensions
        self._embedding_dim = data_dim
        self._generator = Generator(self._embedding_dim , self._generator_dim, data_dim, discrete_cols).to(self._device)
        discriminator = Discriminator( data_dim, self._discriminator_dim, pac=self.pac).to(self._device)
        optimizerG, optimizerD = define_optimizer(self, args, discriminator)
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        # adv_decay  = 1.7
        # pert_decay = 0.99
        # adv_loss_fn = CrossEntropyLoss()
 
        for epoch in range(self._epochs):
            # if epoch > 1:
            #     self.pert_scale = pert_decay*self.pert_scale
            #     self.adv_scale = self.adv_scale*adv_decay
            loss_g_running,  loss_d_syn_running, loss_d_real_running, loss_d_running, adv_loss_running, pert_loss_running = 0, 0, 0, 0, 0, 0
            for id_ in tqdm(range(steps_per_epoch), total=steps_per_epoch):
                mean_d = mean_d_syn = mean_d_real = 0
                self._discriminator_steps = 1

                ################### Train Discriminator ##########################

                for n in range(self._discriminator_steps):

                    idx, real = self.get_train_data()
                    # real = random_sphere_torch(self.args, real)
                    adv = self._generator(real)
                    adv_act = apply_activate(self._transformer, adv)
                    adv_act[:,self.modes_idx] = real[:,self.modes_idx]
                    adv_act[:,self.not_modif_idx] = real[:,self.not_modif_idx]

                    if self._version=="unconstrained" or self._version == "postprocessing":
                        fakecons = adv_act.clone()
                    else:
                        fakecons, _ = apply_constrained(self._transformer, adv_act, self.ordering, self.sets_of_constr, self.args)
                        fakecons[:,self.not_modif_idx] = real[:,self.not_modif_idx]
               

                    y_fake = discriminator(fakecons.squeeze())
                    y_real = discriminator(real)

                    pen = discriminator.calc_gradient_penalty(real, fakecons, self._device, self.pac)
                    loss_syn_d = torch.mean(y_fake.detach())
                    loss_real_d = torch.mean(y_real)
                    loss_d = -(loss_real_d - loss_syn_d)

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()
                    mean_d_syn += loss_syn_d
                    mean_d_real += loss_real_d
                    mean_d += loss_d


                ################### Train Generator ##########################
                optimizerG.zero_grad()
                idx, real = self.get_train_data()
                adv = self._generator(real)
                adv_act = apply_activate(self._transformer, adv)
                adv_act[:,self.modes_idx] = real[:,self.modes_idx]
                adv_act[:,self.not_modif_idx] = real[:,self.not_modif_idx]

                if self._version=="unconstrained" or self._version == "postprocessing":
                    fakecons = adv_act.clone()
                else:
                    fakecons, _ = apply_constrained(self._transformer, adv_act, self.ordering, self.sets_of_constr, self.args)
                    fakecons[:,self.not_modif_idx] = real[:,self.not_modif_idx]
                y_fake = discriminator(fakecons.squeeze())
                loss_g = -torch.mean(y_fake)
                loss_g.backward(retain_graph=True)

                ## Random weighting of the loss RLW technique
                weights = F.softmax(torch.randn(2), dim=-1)

                true_data_inv = torch.from_numpy(train_data_orig.iloc[idx,:].values.astype('float32')).to(self._device)
                self.inverse = self._transformer.inverse_transform(fakecons)
                probs = self._target_model.get_logits(self.inverse, with_grad=True)
                adv_loss = adversarial_loss(probs, true_data_inv[:, -1].long(), num_labels, self.targeted)

                pert_loss = torch.mean(torch.norm(fakecons-real, 2, dim=1))
                gen_adv_loss = self.pert_scale*weights[0]*pert_loss + self.adv_scale*weights[1]*adv_loss
                gen_adv_loss.backward()
             
                optimizerG.step()

                loss_d_syn = mean_d_syn/self._discriminator_steps
                loss_d_real = mean_d_real/self._discriminator_steps
                loss_d = -(loss_d_real - loss_d_syn)
                loss_g_running += loss_g
                loss_d_syn_running += loss_d_syn
                loss_d_real_running += loss_d_real
                loss_d_running += loss_d
                pert_loss_running += pert_loss
                adv_loss_running += adv_loss
           

            wandb.log({'epochs/epoch': epoch, 
                       'epochs/loss_gen': loss_g_running/steps_per_epoch, 
                       'epochs/loss_pert': pert_loss_running/steps_per_epoch,
                       'epochs/loss_adv': adv_loss_running/steps_per_epoch, 
                       'epochs/loss_disc_syn': loss_d_syn_running/steps_per_epoch, 
                       'epochs/loss_disc_real': loss_d_real_running/steps_per_epoch, 
                       'epochs/loss_disc': loss_d_running/steps_per_epoch})

            if self._verbose:
                print(f'Epoch {epoch+1}, Loss G: {loss_g_running/steps_per_epoch: .4f},'  # noqa: T001
                      f'Loss D: {loss_d_running/steps_per_epoch: .4f}',
                      flush=True)
                print('Epoch {}: perturbation_loss {:.3f} adversarial_loss {:.3f}'.format(epoch, pert_loss_running/steps_per_epoch, adv_loss_running/steps_per_epoch))

            if epoch >= 25 and epoch % args.save_every_n_epochs == 0:
                torch.save(self._generator, f"{self._path}/model_{epoch}.pt")

        PATH = f"{self._path}/ctgan_model.pt"
        self.save(PATH)


    @random_state
    def sample(self, adv_cand):
        data = []
        batch_trans = self._transformer.transform(adv_cand, None)
        self._data_sampler = DataSampler(batch_trans, self._transformer.output_info_list, self._log_frequency)
        data_iterator = NumpyArrayIterator(batch_trans, self._batch_size)
        self._generator.eval()

        for batch in data_iterator:
            batch = torch.from_numpy(batch.astype('float32')).to(self._device)
            fake = self._generator(batch)
            fakeact = apply_activate(self._transformer, fake)
            fakeact[:,self.modes_idx] = batch[:,self.modes_idx]
            fakeact[:,self.not_modif_idx] = batch[:,self.not_modif_idx]
            data.append(fakeact)

        data = torch.concat(data, axis=0)
        inverse = self._transformer.inverse_transform(data)
        if self._version == "constrained" or self._version == "postprocessing":
            inverse = correct_preds(inverse, self.ordering, self.sets_of_constr)
        if self.not_modifiable:
            inverse[:, self.not_modifiable] = torch.from_numpy(adv_cand.iloc[:, self.not_modifiable].values.astype('float32'))
        for i in range(inverse.shape[1]):
            inverse[:,i] = round_func_BPDA(inverse[:,i], self.args.round_decs[i])
        return inverse.detach().numpy()

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
