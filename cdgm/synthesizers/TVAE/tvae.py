"""TVAE module."""

import torch
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from cdgm.constraints_code.correct_predictions import correct_preds
from cdgm.data_processors.ctgan.data_transformer import DataTransformer
from cdgm.synthesizers.CTGAN.base_ctgan import BaseSynthesizer, random_state
from cdgm.synthesizers.TVAE.blocks import Encoder, Decoder
from cdgm.synthesizers.utils import get_discrete_col, get_not_modif_idx, get_modes_idx
from cdgm.synthesizers.utils import apply_activate, apply_constrained, get_sets_constraints, adversarial_loss, round_func_BPDA, correct_preds
from utils import NumpyArrayIterator

import wandb


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor, version):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                if version == "constrained":
                    eq = x[:, st] - recon_x[:, st]
                else:
                    eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed
            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]



class TVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
        self,
        target_model,
        target_scaler,
        test_data,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        pert_scale=1,
        adv_scale=1,
        cuda=True,
        path='',
        bin_cols_idx=[], 
        not_modifiable=[],
        version="unconstrained",
        verbose=True
    ):
        self.target_model = target_model
        self.target_scaler = target_scaler
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.pert_scale = pert_scale
        self.adv_scale = adv_scale
        self.epochs = epochs
        self._version = version
        self.not_modifiable=not_modifiable
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        self._path = path
        self._verbose = verbose

    
    def _apply_constrained(self, dec_data, data_bef, sigmas):
        self.transformed, self.inverse = apply_constrained(self.transformer, dec_data, self.ordering, self.sets_of_constr, self.args)

        # data_t = []
        # st = 0
        # for column_info in self.transformer.output_info_list:
        #     for span_info in column_info:
        #         if span_info.activation_fn == 'tanh':
        #             ed = st + span_info.dim
        #             data_t.append(self.transformed[:, st:ed])
        #             st = ed
        #         elif span_info.activation_fn == 'softmax':
        #             ed = st + span_info.dim
        #             data_t.append(data_bef[:, st:ed])
        #             st = ed
        #         else:
        #             raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        # data = torch.cat(data_t, dim=1)
        # return data
        return self.transformed

    @random_state
    def fit(self, args, train_data, num_labels,  discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

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
        self.constraints, self.sets_of_constr, self.ordering = get_sets_constraints("tvae", self.args.use_case, args.label_ordering, args.constraints_file)

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data, None)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        data_dim = self.transformer.output_dimensions

        
        self.not_modif_idx = get_not_modif_idx(self.transformer, self.not_modifiable)
        self.modes_idx = get_modes_idx(self.transformer)

        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        for epoch in range(self.epochs):
            decoder_loss_running, pert_loss_running, adv_loss_running = 0, 0, 0
            # if epoch % self.loss_switch_eps == 0:
            for id_, data in enumerate(loader):
                real = data[0].to(self._device)
  
                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                # emb = mu + std
                emb = eps * std + mu
                rec_, sigmas = self.decoder(emb)
                rec_.retain_grad()
                if args.version=="constrained":
                    rec_act = apply_activate(self.transformer, rec_)
                    rec_act[:,self.modes_idx] = real[:,self.modes_idx]
                    rec = self._apply_constrained(rec_act, rec_, sigmas)
                    rec[:,self.not_modif_idx] = real[:,self.not_modif_idx]

                else:
                    rec = rec_.clone()


                optimizerAE.zero_grad()
                weights = F.softmax(torch.randn(2), dim=-1) # RLW is only this!
                pert_loss = torch.mean(torch.norm(rec - real, 2, dim=1))

                # inverse_adv_scaled = self.target_scaler.transform(self.inverse[:,:-1])
                # real_scaled = self._target_scaler.transform(true_data_inv[:,:-1])
                # perturbation = inverse_adv_scaled - real_scaled
                # pert_loss = torch.mean(torch.norm(perturbation, 2, dim=-1))
                self.inverse = self.transformer.inverse_transform(rec)
                true_data_inv = self.transformer.inverse_transform(real)
                probs = self.target_model.get_logits(self.inverse[:,:-1], with_grad=True)
                adv_loss = adversarial_loss(probs, true_data_inv[:, -1].long(), num_labels, False)

                gen_adv_loss = self.pert_scale*weights[0]*pert_loss + self.adv_scale*weights[1]*adv_loss
                gen_adv_loss.backward(retain_graph=True)

                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor, self._version
                )
                loss = (loss_1 + loss_2)
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
                decoder_loss_running += loss
                pert_loss_running += pert_loss
                adv_loss_running += adv_loss
            wandb.log({'epochs/epoch': epoch, 'epochs/loss':decoder_loss_running/len(loader),
                       'epochs/loss_pert': pert_loss_running/len(loader), 'epochs/loss_adv': adv_loss_running/len(loader)})
            if self._verbose:
                print(f'Epoch {epoch+1}, Loss: {decoder_loss_running/len(loader): .4f}',flush=True)
                print(f'Epoch {epoch+1}: perturbation_loss {pert_loss_running/len(loader):.3f} adversarial_loss {adv_loss_running/len(loader):.3f}')

            if epoch >= 25 and epoch % args.save_every_n_epochs == 0:
                torch.save(self.decoder, f"{self._path}/model_{epoch}.pt")

        PATH = f"{self._path}/model.pt"
        torch.save(self.decoder, PATH)


    @random_state
    def sample(self, adv_cand):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        data = []
        data_act = []
        batch_trans = self.transformer.transform(adv_cand, None)
        data_iterator = NumpyArrayIterator(batch_trans, self.batch_size)

        for batch in data_iterator:
            batch = torch.from_numpy(batch.astype('float32')).to(self._device)
            mu, std, logvar = self.encoder(batch)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            fake, sigmas = self.decoder(emb)
            fakeact = apply_activate(self.transformer, fake)
            fakeact[:,self.modes_idx] = batch[:,self.modes_idx]
            data.append(fakeact)
        data = torch.concat(data, axis=0)
        inverse = self.transformer.inverse_transform(data)
        if self._version == "constrained" or self._version == "postprocessing":
            inverse = correct_preds(inverse, self.ordering, self.sets_of_constr)
        if self.not_modifiable:
            inverse[:, self.not_modifiable] = torch.from_numpy(adv_cand.iloc[:, self.not_modifiable].values.astype('float32'))
        for i in range(inverse.shape[1]):
            inverse[:,i] = round_func_BPDA(inverse[:,i], self.args.round_decs[i])
        return inverse[:,:-1].detach().numpy()


    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)