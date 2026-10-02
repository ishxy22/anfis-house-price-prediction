# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
import itertools

class GaussianMF(nn.Module):
    def __init__(self, num_mfs, input_dim):
        super(GaussianMF, self).__init__()
        # Initialize means to be distributed evenly across [0, 1]
        self.mu = nn.Parameter(torch.linspace(0.1, 0.9, num_mfs).repeat(input_dim, 1))
        # Initialize sigmas
        self.sigma = nn.Parameter(torch.ones(input_dim, num_mfs) * 0.2)
        
    def forward(self, x):
        # x is (batch_size, input_dim)
        # return is (batch_size, input_dim, num_mfs)
        x_expanded = x.unsqueeze(2) # (batch_size, input_dim, 1)
        mu = self.mu.unsqueeze(0) # (1, input_dim, num_mfs)
        sigma = self.sigma.unsqueeze(0) # (1, input_dim, num_mfs)
        return torch.exp(-((x_expanded - mu) ** 2) / (2 * sigma ** 2 + 1e-8))

class ANFIS(nn.Module):
    def __init__(self, input_dim, num_mfs):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        
        # Layer 1: Gaussian Membership Functions
        self.layer1 = GaussianMF(num_mfs, input_dim)
        
        # Total number of rules
        self.num_rules = num_mfs ** input_dim
        
        # Consequent parameters for First-Order Sugeno
        self.consequents = nn.Parameter(torch.randn(self.num_rules, input_dim + 1))
        
        # Generate all combinations of MFs for rules
        self.rule_indices = list(itertools.product(range(num_mfs), repeat=input_dim))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Layer 1: Fuzzification
        mf_outputs = self.layer1(x)
        
        # Layer 2: Rule Firing Strengths (Product T-Norm)
        firing_strengths = torch.ones(batch_size, self.num_rules, device=x.device)
        for i, indices in enumerate(self.rule_indices):
            for j, mf_idx in enumerate(indices):
                firing_strengths[:, i] *= mf_outputs[:, j, mf_idx]
                
        # Layer 3: Normalization
        sum_firing = torch.sum(firing_strengths, dim=1, keepdim=True) + 1e-8
        normalized_firing = firing_strengths / sum_firing
        
        # Layer 4: Consequent Evaluation
        x_aug = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
        rule_outputs = torch.einsum('bi,ri->br', x_aug, self.consequents)
        
        # Layer 5: Defuzzification (Weighted Sum)
        output = torch.sum(normalized_firing * rule_outputs, dim=1, keepdim=True)
        
        return output, normalized_firing
