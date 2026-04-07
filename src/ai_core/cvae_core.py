import torch
import torch.nn as nn

class MetallurgicCVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for Alloy Design.
    Learns the 'latent physics' of how elements interact.
    """
    def __init__(self, feature_dim, condition_dim, latent_dim=4):
        super(MetallurgicCVAE, self).__init__()
        
        # 1. ENCODER: Compresses (Recipe + Strength) -> Latent Space
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # 2. DECODER: Reconstructs (Latent Space + Strength) -> Recipe
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim) # Predicts: time, temp, mg, si, etc.
        )

    def reparameterize(self, mu, logvar):
        """The 'Sampling' trick that makes the model generative."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        # x = Recipe features, c = Yield Strength (Target)
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Combine the hidden 'dream' (z) with the target strength (c)
        z_cond = torch.cat([z, c], dim=1)
        recon_x = self.decoder(z_cond)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """
    Standard VAE Loss: Reconstruction Loss + KL Divergence.
    Forces the model to be both accurate and creative.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence: Prevents the latent space from becoming chaotic
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss