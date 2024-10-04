import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class VarianceSchedule(nn.Module):
    """
    A set of hyper-parameters that are directly related to the training of the Loss Function.
    """

    def __init__(self, num_steps=100, beta_1=1e-4, beta_T=0.02):
        super().__init__()
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T

        betas = torch.linspace(beta_1, beta_T, num_steps)
        betas = torch.cat([torch.zeros([1]), betas], dim=0)

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t):
        sigmas = self.sigmas_inflex[t]
        return sigmas

class InterNet(nn.Module):
    """
    simply embeds the time information.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._time_bias = nn.Linear(3, dim_out, bias=False)
        self._time_gate = nn.Linear(3, dim_out)

    def forward(self, x, t):
        gate = torch.sigmoid(self._time_gate(t))
        bias = self._time_bias(t)

        ret = self._layer(x) * gate + bias
        return ret

class SimpleNet(nn.Module):
    """
    A very simple UNet utilizing Residual Connections.
    Using Residual Connections does not make a significant difference.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            InterNet(3, 128),
            InterNet(128, 256),
            InterNet(256, 512),
            InterNet(512, 256),
            InterNet(256, 128),
            InterNet(128, 3)
        ])

    def forward(self, x, t):
        batch_size = x.size(0)
        t = t.view(batch_size, 1, 1)
        time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out, time_emb)
            if i < len(self.layers) - 1:
                out = nn.LeakyReLU()(out)
        return out + x

class BoringNet(nn.Module):
    """
    This is to demonstrate that the model does not perform well without embedding time information.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(3, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Linear(128, 3)
        ])

    def forward(self, x, t):  # t는 그냥 코드 편의상 남겨둠
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = nn.LeakyReLU()(out)
        return x + out
class DiffusionPoint(nn.Module):
    """
    Point Diffusion.
    The code is implemented by excluding the Latent Shape. I only removed unnecessary parts accordingly.
    """
    def __init__(self, net, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x0, t=None):
        batch_size, _, point_dim = x0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]

        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_rand = torch.randn_like(x0)  # initial Gaussian
        e_theta = self.net(c0 * x0 + c1 * e_rand, beta)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, batch_size, num_points, point_dim, device: torch.device, ret_traj=False):
        x_T = torch.randn([batch_size, num_points, point_dim]).to(device)
        traj = {self.var_sched.num_steps: x_T}

        for t in range(self.var_sched.num_steps, 0, -1):
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta)
            x_next = c0 * (x_t - c1 * e_theta)
            traj[t-1] = x_next.detach()
            traj[t] = traj[t].cpu()
        
        if ret_traj:
            return traj
        else:
            return traj[0]


if __name__ == '__main__':
    from dataset import ShapeNetCore

    SEED = 1234
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    path = './data/shapenet.hdf5'
    cate = "airplane"
    split = "train"  # test, train, val

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ShapeNetCore(path, cate)
    data = dataset[0].to(device)
    data = data.view(1, -1, 3)

    diff = DiffusionPoint(
        SimpleNet(),
        VarianceSchedule()
    )
    diff.to(device)

    loss = diff.get_loss(data)
    print(loss)