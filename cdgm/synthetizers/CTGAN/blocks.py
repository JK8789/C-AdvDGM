import torch
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, MSELoss
from torch.optim import Adam, RMSprop, SGD


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


  
class Residual(Module):
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


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim, discrete_cols):
        super(Generator, self).__init__()
        self.discrete_cols = discrete_cols
        dim = embedding_dim
        #dim = data_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)


    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


def define_optimizer(self, args, discriminator):
    if args.optimiser == "adam":
        optimizerG = Adam(self._generator.parameters(), lr=self._generator_lr, betas=(0.9, 0.999), weight_decay=self._generator_decay)
        optimizerD = Adam(discriminator.parameters(), lr=self._discriminator_lr, betas=(0.9, 0.999), weight_decay=self._discriminator_decay)
    elif args.optimiser == "rmsprop":
        optimizerG = RMSprop(self._generator.parameters(), lr=self._generator_lr, alpha=0.9, momentum=0, eps=1e-3, weight_decay=self._generator_decay)
        optimizerD = RMSprop(discriminator.parameters(), lr=self._discriminator_lr, alpha=0.9, momentum=0, eps=1e-3, weight_decay=self._discriminator_decay)
    elif args.optimiser == "sgd":
        optimizerG = SGD(self._generator.parameters(), lr=self._generator_lr, momentum=0.001, weight_decay=self._generator_decay)
        optimizerD = SGD(discriminator.parameters(), lr=self._discriminator_lr, momentum=0.001, weight_decay=self._discriminator_decay)
    else:
        pass
    return optimizerG, optimizerD