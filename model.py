import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.poutine as poutine
import torch.nn.functional as F




class FilteringGuideNet(nn.Module):
    def __init__(self, input_dim, x_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + input_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

    def forward(self, z_prev, A_t, X_t):
        h = torch.cat([z_prev, A_t, X_t], dim=-1)
        mu, logvar = self.net(h).chunk(2, dim=-1)
        return mu, logvar



class PyroDVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, x_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ---------- Generative model ----------
        # Transition: p(z_t | z_{t-1}, A_{t-1})
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + input_dim + x_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )


        # Emission: p(A_t | z_t)
        self.emission = nn.Sequential(
            nn.Linear(latent_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # ---------- Filtering guide ----------
        self.guide_transition = FilteringGuideNet(
            input_dim, x_dim, latent_dim, hidden_dim
        )

        # Initial posterior q(z_1 | A_1)
        self.z1_encoder = nn.Sequential(
            nn.Linear(input_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

        def _init_weights(m):
          if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        self.apply(_init_weights)

    def model(self, A, X, obs_mask=None, beta=1.0, **kwargs):
        """
        A: (B, T, D) count tensor
        X: (B, T, Dx) covariates
        obs_mask: (B, T, D) bool mask; True=observed, False=held-out
        """
        batch_size, T, D = A.shape

        if obs_mask is None:
            obs_mask = torch.ones(batch_size, T, D, dtype=torch.bool, device=A.device)

        # Use only observed causes as inputs to avoid leakage from held-out causes
        A_obs = A * obs_mask.to(A.dtype)

        pyro.module("transition", self.transition)
        pyro.module("emission", self.emission)


        with pyro.plate("data", batch_size):

            # p(z_1)
            with poutine.scale(scale=beta):
                z_prev = pyro.sample(
                      "z_1",
                      dist.Normal(
                          torch.zeros(batch_size, self.latent_dim, device=A.device),
                          torch.ones(batch_size, self.latent_dim, device=A.device)
                      ).to_event(1)
                  )


            valid = obs_mask[:, 0, :]  # (B, D), bool

            

            # t = 1
            rate = F.softplus(
                self.emission(torch.cat([z_prev, X[:, 0]], dim=-1))
            )
            rate = torch.clamp(rate, min=1e-6, max=1e3)  
            
            pyro.sample(
                "A_1",
                dist.Poisson(rate).mask(valid).to_event(1),
                obs=A_obs[:, 0]
            )
  

            # t >= 2
            for t in pyro.markov(range(1, T)):
                trans_in = torch.cat([z_prev, A_obs[:, t-1], X[:, t-1]], dim=-1)
                mu, logvar = self.transition(trans_in).chunk(2, dim=-1)
                
                with poutine.scale(scale=beta):
                    z_t = pyro.sample(
                          f"z_{t+1}",
                          dist.Normal(mu, torch.exp(0.5 * logvar)).to_event(1)
                      )


                valid = obs_mask[:, t, :]  # (B, D), bool


                rate = F.softplus(
                    self.emission(torch.cat([z_t, X[:, t]], dim=-1))
                )
                rate = torch.clamp(rate, min=1e-6, max=1e3)  
                
                # Debug invariants (remove after fixing)
                assert not torch.isnan(A).any(), "A contains NaN inside model()"
                assert (A >= 0).all(), "A contains negative values"
                assert torch.all(A == A.floor()), "A contains non-integer values for Poisson"

                pyro.sample(
                    f"A_{t+1}",
                    dist.Poisson(rate).mask(valid).to_event(1),
                    obs=A_obs[:, t]
                )

                # with poutine.mask(mask=obs_mask[:, t, :]):
                #     pyro.sample(
                #         f"A_{t+1}",
                #         dist.Poisson(rate).to_event(1),
                #         obs=A[:, t]
                #     )


                z_prev = z_t


    def guide(self, A, X, obs_mask=None, beta=1.0, **kwargs):
        """
        Structural filtering (one-step / Markov):
        q(z_1 | X_1)            # or prior if you prefer
        q(z_t | z_{t-1}, A_{t-1}, X_{t-1})
        """

        batch_size, T, D = A.shape

        if obs_mask is None:
            obs_mask = torch.ones(batch_size, T, D, dtype=torch.bool, device=A.device)

        A_obs = A * obs_mask.to(A.dtype)
        

        pyro.module("guide_transition", self.guide_transition)
        pyro.module("z1_encoder", self.z1_encoder)

        with pyro.plate("data", batch_size):

            # q(z_1 | A_1, X_1)
            mu1, logvar1 = self.z1_encoder(
                torch.cat([A_obs[:, 0], X[:, 0]], dim=-1)
            ).chunk(2, dim=-1)
            
            logvar1 = torch.clamp(logvar1, min=-10.0, max=10.0)  
            scale1 = torch.exp(0.5 * logvar1)                    
            scale1 = torch.clamp(scale1, min=1e-6)               

            z_prev = pyro.sample(
                "z_1",
                dist.Normal(mu1, scale1).to_event(1)
            )

            # t >= 2
            for t in pyro.markov(range(1, T)):
                mu, logvar = self.guide_transition(
                    z_prev,
                    A_obs[:, t],
                    X[:, t]
                )
                
                logvar = torch.clamp(logvar, min=-10.0, max=10.0)  
                scale = torch.exp(0.5 * logvar)                   
                # scale = torch.clamp(scale, min=1e-6)              

                z_t = pyro.sample(
                    f"z_{t+1}",
                    dist.Normal(mu, scale).to_event(1)
                )

                # z_t = pyro.sample(
                #     f"z_{t+1}",
                #     dist.Normal(mu, scale).to_event(1)
                # )
               
                z_prev = z_t


