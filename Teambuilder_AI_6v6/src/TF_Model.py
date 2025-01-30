import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import HyperParameters as H
import math
import Utils as U

with open(U.species_number, 'r') as f:
    num_species = int(f.read())
    species_em_dim = int(math.ceil(math.sqrt(num_species)))

class TeamBuilder(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads=2, num_layers=2, 
                 num_species=num_species, species_em_dim=species_em_dim, 
                 num_arch=2, num_dc=3, num_style=2, num_weather=2, num_type=19, feature_embed_dim=8,
                 dropout=0.1, hidden_dim=H.HIDDEN_UNITS):
        super(TeamBuilder, self).__init__()
        self.species_em = nn.Embedding(num_embeddings=num_species, embedding_dim=species_em_dim)
        self.arch_em = nn.Embedding(num_embeddings=num_arch, embedding_dim=feature_embed_dim)
        self.dc_em = nn.Embedding(num_embeddings=num_dc, embedding_dim=feature_embed_dim)
        self.style_layer = nn.Linear(41, hidden_dim)
        self.weather_em = nn.Linear(8, embed_dim)
        self.type1_em = nn.Embedding(num_type, feature_embed_dim)
        self.type2_em = nn.Embedding(num_type, feature_embed_dim)

        weather_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=1,
            dim_feedforward= 2 * embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.weather_transformer = nn.TransformerEncoder(weather_encoder_layer, num_layers=num_layers)

        species_encoder_layer = nn.TransformerEncoderLayer(
            d_model=species_em_dim,
            nhead=num_heads,
            dim_feedforward= 4 * embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.species_transformer = nn.TransformerEncoder(species_encoder_layer, num_layers=num_layers)

        embed_feature_size = 96
        self.layer_norm = nn.LayerNorm(embed_feature_size)
        self.project = nn.Linear(embed_feature_size, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, input_size)

    def forward(self, src):
        # Get features sizes
        #print(src)
        batch_size, num_pokemon, _ = src.size()

        # Extract features
        species = src[:, :, 0]
        arch = src[:, :, 1]
        dc = src[:, :, 2]
        style = src[:, :, 3:-11]
        weather = src[:, :, -10:-2]
        type1 = src[:, :, -2]
        type2 = src[:, :, -1]

        #feature weighting
        species_weight = 1.5
        weather_weight = 1
        style_weight = 0.8  

        style = style.to(torch.float)
        weather = weather.to(torch.float)

        species = self.species_em(species)
        #species = self.species_project(species)
        species = self.species_transformer(species)
        arch = self.arch_em(arch)
        dc = self.dc_em(dc)
        style = F.relu(self.style_layer(style)) 
        weather = F.relu(self.weather_em(weather))
        weather = self.weather_transformer(weather)
        type1 = self.type1_em(type1) 
        type2 = self.type2_em(type2)
        #x = torch.cat([species, arch, dc, style, weather, type1, type2], dim=-1)
        # Combine all features
        x = torch.cat((
            species, 
            arch,
            dc,
            style,
            weather,
            type1,
            type2
        ), dim=-1)
        #print(x.shape)
        x = F.relu(self.layer_norm(x))
        x = F.relu(self.project(x))
        x = self.transformer(x)
        
        # Final projection to Pokemon IDs
        logits = self.fc_out(x)
        return logits
    
    def generate(self, start_tokens, tokenizer, max_length=6, temperature=1.0, threshold=0, top_k=20, min_k=H.min_k, dynamic_k=False, repetition_penalty=.25, weather_repetition_penalty=.25, hazard_rep_pen=.5):
        self.eval()
        
        current_sequence = start_tokens
        batch_size = current_sequence.size(0)
        
        while current_sequence.size(1) < max_length:
            with torch.no_grad():
                current_sequence = current_sequence.to(torch.long)
                logits = self(current_sequence)
                next_token_logits = logits[:, -1, :]

                # Apply masking first - before any other operations #
                for batch_idx in range(batch_size):
                    # Get type combinations already used in this team
                    used_types = set(
                        (current_sequence[batch_idx, i, -1].item(), 
                        current_sequence[batch_idx, i, -2].item())
                        for i in range(current_sequence.size(1))
                    )
                    
                    # Create and apply mask immediately
                    mask = torch.ones_like(next_token_logits[batch_idx], dtype=torch.bool)
                    for idx in range(len(tokenizer)):
                        features = tokenizer[idx]
                        if (features[-1], features[-2]) in used_types:
                            mask[idx] = False
                    
                    # Apply mask early
                    next_token_logits[batch_idx][~mask] = float('-inf')
                # Apply masking first - before any other operations #

                # REPETITION PENALTY #
                for batch_idx in range(batch_size):
                    current_types = set()
                    weather_setters = 0
                    rockers = 0
                    spikers = 0
                    tspikers = 0
                    for i in range(current_sequence.size(1)):
                        rocker = current_sequence[batch_idx, i, 8].item()
                        spiker = current_sequence[batch_idx, i, 9].item()
                        tspiker = current_sequence[batch_idx, i, 10].item()
                        rain_setter = current_sequence[batch_idx, i, -10].item()
                        sand_setter = current_sequence[batch_idx, i, -8].item()
                        sun_setter = current_sequence[batch_idx, i, -6].item()
                        snow_setter = current_sequence[batch_idx, i, -4].item()
                        weather_setter = sum([rain_setter, sand_setter, sun_setter, snow_setter])
                        type1 = current_sequence[batch_idx, i, -2].item()
                        type2 = current_sequence[batch_idx, i, -1].item()

                        if rocker != 0:
                            rockers = 1
                        if spiker != 0:
                            spikers = 1
                        if tspiker != 0:
                            tspikers = 1
                        if weather_setter != 0:
                            weather_setters = 1
                        if type1 != 18:
                            current_types.add(type1)
                        if type2 != 18:
                            current_types.add(type2)
                    
                    # Only apply penalty to non-masked tokens
                    valid_indices = torch.where(next_token_logits[batch_idx] != float('-inf'))[0]
                    for idx in valid_indices:
                        features = tokenizer[idx.item()]
                        rockers, spikers, tspikers, type1, type2 = features[8], features[9], features[10], features[-2], features[-1]
                        weather_setter = sum([features[-10], features[-8], features[-6], features[-4]])

                        if rockers == 1 and rocker == 1:
                            next_token_logits[batch_idx, idx] *= hazard_rep_pen
                        if spikers == 1 and spiker == 1:
                            next_token_logits[batch_idx, idx] *= hazard_rep_pen
                        if tspikers == 1 and tspiker == 1:
                            next_token_logits[batch_idx, idx] *= hazard_rep_pen
                        if weather_setter >= 1 and weather_setters >= 1:
                            next_token_logits[batch_idx, idx] *= weather_repetition_penalty
                        if type1 in current_types and type1 != 18:
                            next_token_logits[batch_idx, idx] *= repetition_penalty
                        elif type2 in current_types and type2 != 18:
                            next_token_logits[batch_idx, idx] *= repetition_penalty

                # REPETITION PENALTY #

                # Calculate probabilities only for valid options
                probs_before_sampling = F.softmax(next_token_logits, dim=-1)

                
                # Dynamic k calculation #
                thresholds = [0.001, 0.01, 0.05]
                k = top_k
                for i, thresh in enumerate(thresholds):
                    # Only count valid (non-masked) probabilities
                    valid_probs = probs_before_sampling * (next_token_logits != float('-inf'))
                    count = (valid_probs > thresh).sum(dim=-1).float().mean().item()
                    if count >= min_k and count < k and dynamic_k:
                        k = int(count)
                    print(f"Average number of valid labels with >{thresh*100}% chance: {count:.1f}")
                print(f"k: {k}")
                # Dynamic k calculation #

                # Apply top-k only to valid options #
                top_k_values = torch.topk(next_token_logits, min(k, (next_token_logits != float('-inf')).sum(dim=-1).min().item()), dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_values.indices, top_k_values.values)
                # Apply top-k only to valid options #

                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Convert to probabilities and apply threshold
                probs = F.softmax(next_token_logits, dim=-1)
                probs[probs < threshold] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample next Pokemon
                next_indices = torch.multinomial(probs, 1)
                
                # Convert indices to feature representations
                next_features = self.index_to_features(next_indices, tokenizer)
                
                # Add new Pokemon to teams
                current_sequence = torch.cat([current_sequence, next_features], dim=1)
        
        return current_sequence.to(torch.long)

    def index_to_features(self, tokens, tokenizer):
        tokens_features = torch.zeros(H.BATCH_SIZE, 1, H.feature_size)
        tokens_features = tokens_features.to(H.device)
        for i, token in enumerate(tokens):
            #print(tokenizer[token])
            feature_tensor = torch.from_numpy(np.array(tokenizer[token]).astype(np.float32)).to(torch.long)
            tokens_features[i, 0] = feature_tensor
        return tokens_features
