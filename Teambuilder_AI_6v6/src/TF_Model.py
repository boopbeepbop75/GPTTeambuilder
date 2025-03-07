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
    while species_em_dim%H.NUM_HEADS != 0:
        species_em_dim += 1
print(species_em_dim)

# Set Attention Block for permutation invariant processing
class SetAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention without causal mask
        attended, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attended)
        
        # Feed-forward
        x = self.norm2(x + self.ff(x))
        return x
    
class TransformerFeatureBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.2, ff_expansion=4):
        super(TransformerFeatureBlock, self).__init__()
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # First layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_expansion, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Second layer normalization
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key=None, value=None, key_padding_mask=None, attn_mask=None):
        # Use query for key and value if they're not provided
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Multi-head attention with residual connection and layer norm
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
        
        # Residual connection should be with the query
        x = query + attn_output
        x = self.layer_norm1(x)  # Layer normalization
        
        # Feed-forward network with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + ff_output  # Residual connection
        x = self.layer_norm2(x)  # Layer normalization
        
        return x, attn_weights

class TeamBuilder(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads=H.NUM_HEADS, num_layers=H.NUM_LAYERS, 
                 num_species=num_species, species_em_dim=species_em_dim, 
                 num_arch=2, num_dc=3, num_style=2, num_weather=2, num_type=19, feature_embed_dim=4,
                 dropout=0.2, num_removal = 2, num_hazards=2):
        super(TeamBuilder, self).__init__()
        
        # Embedding Layers (unchanged)
        self.species_em = nn.Embedding(num_embeddings=num_species, embedding_dim=species_em_dim)
        self.arch_em = nn.Embedding(num_embeddings=num_arch, embedding_dim=feature_embed_dim)
        self.dc_em = nn.Embedding(num_embeddings=num_dc, embedding_dim=feature_embed_dim)
        self.style_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_style, embedding_dim=feature_embed_dim)
            for _ in range(42)  # number of style features
        ])
        self.weather_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_weather, embedding_dim=feature_embed_dim)
            for _ in range(8)
        ])
        self.type1_em = nn.Embedding(num_type, feature_embed_dim)
        self.type2_em = nn.Embedding(num_type, feature_embed_dim)
        self.removal_em = nn.Embedding(num_removal, feature_embed_dim)

        self.hazards_ems = nn.ModuleList([
            nn.Embedding(num_embeddings=num_hazards, embedding_dim=feature_embed_dim)
            for _ in range(3)  # number of style features
        ])

        # Multihead Attention layers (unchanged)
        self.species_multihead_attn = TransformerFeatureBlock(embed_dim=species_em_dim)

        self.style_projection = nn.Linear(168, species_em_dim)
        self.style_multihead_attn = TransformerFeatureBlock(embed_dim=species_em_dim)

        self.weather_projection = nn.Linear(32, species_em_dim)
        self.weather_multihead_attn = TransformerFeatureBlock(embed_dim=species_em_dim)

        self.types_projection = nn.Linear(8, species_em_dim)
        self.types_multihead_attn = TransformerFeatureBlock(embed_dim=species_em_dim)

        self.hazards_projection = nn.Linear(12, species_em_dim)
        self.hazards_multihead_attn = TransformerFeatureBlock(embed_dim=species_em_dim)

        self.removal_projection = nn.Linear(8, species_em_dim)
        self.removal_multihead_attn = TransformerFeatureBlock(embed_dim=species_em_dim)

        # Keep your existing feature size and projection
        embed_feature_size = feature_embed_dim * 2 + species_em_dim * 6
        self.all_multihead_attn = TransformerFeatureBlock(embed_dim=embed_feature_size,
                                                        num_heads=2)

        self.project = nn.Linear(embed_feature_size, embed_dim)
        
        # Replace encoder+decoder with set-based processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Replace decoder with Set Transformer blocks
        self.set_blocks = nn.ModuleList([
            SetAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Global context attention for permutation invariance
        self.global_context = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Keep output projection
        self.fc_out = nn.Linear(embed_dim, input_size)

    def forward(self, src):
        batch_size, num_pokemon, _ = src.size()

        # Extract features (unchanged)
        species = src[:, :, H.species]
        arch = src[:, :, H.archetypes]
        dc = src[:, :, H.damage_class]
        style = src[:, :, H.style_start:H.style_end]
        weather = src[:, :, H.weather_start:H.weather_end]
        type1 = src[:, :, H.type1]
        type2 = src[:, :, H.type2]
        removal = src[:, :, H.spinner:H.defogger+1]
        hazards = src[:, :, H.rocker:H.tspiker+1]

        ### RUN THROUGH EMBEDDING LAYERS (unchanged) ###
        species = self.species_em(species)
        arch = self.arch_em(arch)
        dc = self.dc_em(dc)

        # Process style embeddings
        style_embeddings = []
        for i in range(style.size(-1)):
            style_feature = style[:, :, i].long()
            embedded = self.style_embeddings[i](style_feature)
            style_embeddings.append(embedded)
        style = torch.cat(style_embeddings, dim=-1)

        # Process weather embeddings
        weather_embeddings = []
        for i in range(weather.size(-1)):
            weather_feature = weather[:, :, i].long()
            embedded = self.weather_embeddings[i](weather_feature)
            weather_embeddings.append(embedded)
        weather = torch.cat(weather_embeddings, dim=-1)

        type1 = self.type1_em(type1)
        type2 = self.type2_em(type2)
        removal = self.removal_em(removal).view(batch_size, num_pokemon, -1)
        hazards_embeddings = []
        for i in range(hazards.size(-1)):
            hazard_feature = hazards[:, :, i].long()
            embedded = self.hazards_ems[i](hazard_feature)
            hazards_embeddings.append(embedded)
        hazards = torch.cat(hazards_embeddings, dim=-1)
        ### FINISHED RUNNING THROUGH EMBEDDING LAYERS ###

        ### FORM INFO FOR EACH FEATURE (unchanged) ###
        types = torch.cat([type1, type2], dim=-1)
        style = self.style_projection(style)
        weather = self.weather_projection(weather)
        types = self.types_projection(types)
        hazards = self.hazards_projection(hazards)
        removal = self.removal_projection(removal)
        ## END FORM INFO ###

        ### RUN THROUGH MULTIHEAD ATTENTION LAYERS (unchanged) ###
        species, _ = self.species_multihead_attn(species, species, species)

        style, _ = self.style_multihead_attn(style, species, species)

        weather, _ = self.weather_multihead_attn(weather, species, species)

        types, _ = self.types_multihead_attn(types, species, species)

        hazards, _ = self.hazards_multihead_attn(hazards, species, species)

        removal, _ = self.removal_multihead_attn(removal, species, species)
        ### FINISHED RUNNING THROUGH MULTIHEAD ATTN LAYERS ###

        # Combine features (unchanged)
        x = torch.cat((
            species,
            arch,
            dc,
            style,
            weather,
            types,
            hazards,
            removal
        ), dim=-1)

        x, _ = self.all_multihead_attn(x, x, x)

        # Project to embedding space (unchanged)
        x = F.relu(self.project(x))

        # Encode with transformer encoder (unchanged)
        memory = self.encoder(x)
        
        # Add global context token for permutation invariance
        global_context = self.global_context.expand(batch_size, -1, -1)
        x_with_context = torch.cat([global_context, x], dim=1)
        
        # Process with set transformer blocks (no causal masking)
        for block in self.set_blocks:
            x_with_context = block(x_with_context)
            
        # Extract team representation without the global context token
        output = x_with_context[:, 1:, :]
        
        # Final projection
        logits = self.fc_out(output)
        return logits

    def generate(self, start_tokens, tokenizer, max_length=H.team_size, temperature=1.0, min_p=.05, top_k=20, min_k=H.min_k, dynamic_k=False, repetition_penalty=.25, weather_repetition_penalty=.25, hazard_rep_pen=.5, track_gradients=False):
        self.eval()
        
        current_sequence = start_tokens
        batch_size = current_sequence.size(0)
        log_probs_list = []

        context_manager = torch.enable_grad() if track_gradients else torch.no_grad()
        
        while current_sequence.size(1) < max_length:
            with context_manager:
                current_sequence = current_sequence.to(torch.long)
                
                # Get encoder memory for full sequence
                logits = self(current_sequence)
                next_token_logits = logits[:, -1, :]
                mask = torch.ones_like(next_token_logits, dtype=torch.bool, device=H.device) #Keep track of what needs to be masked out

                # Masking Dupes
                for i, batch in enumerate(next_token_logits):
                    used_mons = torch.tensor([mon[H.species].item() for mon in current_sequence[i]], device=H.device)
                    mask[i] = ~torch.isin(torch.tensor([features[H.species] for features in tokenizer], device=H.device), used_mons)
                #Finish Masking Dupes
                
                p_probs = F.softmax(next_token_logits, dim=-1) * mask
                highest_p, _ = torch.max(p_probs, dim=-1)  # Get max probability per batch

                for batch_idx in range(batch_size):
                    batch_min_p = highest_p[batch_idx] * min_p
                    batch_min_p_mask = (p_probs[batch_idx] >= batch_min_p)
                    mask[batch_idx] = mask[batch_idx] & batch_min_p_mask

                # Temperature adjustment
                if current_sequence.size(1) > 1:
                    temperature_reduction_factor = H.temp_redux
                    adjusted_temperature = temperature * (temperature_reduction_factor ** (current_sequence.size(1) - 1))
                else:
                    adjusted_temperature = temperature

                next_token_logits = next_token_logits / adjusted_temperature

                # Convert to probabilities
                next_token_logits.masked_fill_(~mask, float('-inf'))
                probs = F.softmax(next_token_logits, dim=-1)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample next Pokemon
                next_indices = torch.multinomial(probs, 1)
                
                # Convert indices to features
                next_features = self.index_to_features(next_indices, tokenizer)
                
                # Add new Pokemon to teams
                current_sequence = torch.cat([current_sequence, next_features], dim=1)

        if not track_gradients:
            return current_sequence.to(torch.long)
        else:
            return current_sequence.to(torch.long), probs

    def index_to_features(self, tokens, tokenizer):
        tokens_features = torch.zeros(H.BATCH_SIZE, 1, H.feature_size)
        tokens_features = tokens_features.to(H.device)
        for i, token in enumerate(tokens):
            feature_tensor = torch.from_numpy(np.array(tokenizer[token]).astype(np.float32)).to(torch.long)
            tokens_features[i, 0] = feature_tensor
        return tokens_features

'''class TeamBuilder(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads=H.NUM_HEADS, num_layers=H.NUM_LAYERS, 
                 num_species=num_species, species_em_dim=species_em_dim, 
                 num_arch=2, num_dc=3, num_style=2, num_weather=2, num_type=19, feature_embed_dim=4,
                 dropout=0.2, num_removal = 2, num_hazards=2):
        super(TeamBuilder, self).__init__()
        
        # Embedding Layers
        self.species_em = nn.Embedding(num_embeddings=num_species, embedding_dim=species_em_dim)
        self.arch_em = nn.Embedding(num_embeddings=num_arch, embedding_dim=feature_embed_dim)
        self.dc_em = nn.Embedding(num_embeddings=num_dc, embedding_dim=feature_embed_dim)
        self.style_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_style, embedding_dim=feature_embed_dim)
            for _ in range(42)  # number of style features
        ])
        self.weather_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_weather, embedding_dim=feature_embed_dim)
            for _ in range(8)
        ])
        self.type1_em = nn.Embedding(num_type, feature_embed_dim)
        self.type2_em = nn.Embedding(num_type, feature_embed_dim)
        self.removal_em = nn.Embedding(num_removal, feature_embed_dim)

        self.hazards_ems = nn.ModuleList([
            nn.Embedding(num_embeddings=num_hazards, embedding_dim=feature_embed_dim)
            for _ in range(3)  # number of style features
        ])

        # Multihead Attention layers
        self.species_multihead_attn = nn.MultiheadAttention(embed_dim=species_em_dim,
                                                            num_heads=1,
                                                            dropout=dropout,
                                                            batch_first=True)
        self.species_layer_norm = nn.LayerNorm(species_em_dim)

        self.style_projection = nn.Linear(168, species_em_dim)
        self.style_multihead_attn = nn.MultiheadAttention(embed_dim=species_em_dim,
                                                          num_heads=1,
                                                          dropout=dropout,
                                                          batch_first=True)
        self.style_layer_norm = nn.LayerNorm(species_em_dim)

        self.weather_projection = nn.Linear(32, species_em_dim)
        self.weather_multihead_attn = nn.MultiheadAttention(embed_dim=species_em_dim, 
                                                    num_heads=1, 
                                                    dropout=dropout, 
                                                    batch_first=True)
        self.weather_layer_norm = nn.LayerNorm(species_em_dim)

        self.types_projection = nn.Linear(8, species_em_dim)
        self.types_multihead_attn = nn.MultiheadAttention(embed_dim=species_em_dim,
                                                          num_heads=1,
                                                          dropout=dropout,
                                                          batch_first=True)
        self.types_layer_norm = nn.LayerNorm(species_em_dim)

        self.hazards_projection = nn.Linear(12, species_em_dim)
        self.hazards_multihead_attn = nn.MultiheadAttention(embed_dim=species_em_dim,
                                                            num_heads=1,
                                                            dropout=dropout,
                                                            batch_first=True)
        self.hazards_layer_norm = nn.LayerNorm(species_em_dim)

        self.removal_projection = nn.Linear(8, species_em_dim)
        self.removal_multihead_attn = nn.MultiheadAttention(embed_dim=species_em_dim,
                                                            num_heads=1,
                                                            dropout=dropout,
                                                            batch_first=True)
        self.removal_layer_norm = nn.LayerNorm(species_em_dim)

        # Keep your existing feature size and projection
        embed_feature_size = feature_embed_dim * 2 + species_em_dim * 6
        self.all_multihead_attn = nn.MultiheadAttention(embed_dim=embed_feature_size,
                                                        num_heads=2,
                                                        dropout=dropout,
                                                        batch_first=True)

        self.layer_norm = nn.LayerNorm(embed_feature_size)
        self.project = nn.Linear(embed_feature_size, embed_dim)
        
        # Replace encoder+fc with encoder+decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Keep output projection
        self.fc_out = nn.Linear(embed_dim, input_size)

    def forward(self, src):
        batch_size, num_pokemon, _ = src.size()

        # Extract features (same as before)
        species = src[:, :, H.species]
        arch = src[:, :, H.archetypes]
        dc = src[:, :, H.damage_class]
        style = src[:, :, H.style_start:H.style_end]
        weather = src[:, :, H.weather_start:H.weather_end]
        type1 = src[:, :, H.type1]
        type2 = src[:, :, H.type2]
        removal = src[:, :, H.spinner:H.defogger+1]
        hazards = src[:, :, H.rocker:H.tspiker+1]

        ### RUN THROUGH EMBEDDING LAYERS ###
        # Process embeddings (same as before)
        species = self.species_em(species)
        arch = self.arch_em(arch)
        dc = self.dc_em(dc)

        # Process style embeddings
        style_embeddings = []
        for i in range(style.size(-1)):
            style_feature = style[:, :, i].long()
            embedded = self.style_embeddings[i](style_feature)
            style_embeddings.append(embedded)
        style = torch.cat(style_embeddings, dim=-1)

        # Process weather embeddings
        weather_embeddings = []
        for i in range(weather.size(-1)):
            weather_feature = weather[:, :, i].long()
            embedded = self.weather_embeddings[i](weather_feature)
            weather_embeddings.append(embedded)
        weather = torch.cat(weather_embeddings, dim=-1)

        type1 = self.type1_em(type1)
        type2 = self.type2_em(type2)
        removal = self.removal_em(removal).view(batch_size, num_pokemon, -1)
        hazards_embeddings = []
        for i in range(hazards.size(-1)):
            hazard_feature = hazards[:, :, i].long()
            embedded = self.hazards_ems[i](hazard_feature)
            hazards_embeddings.append(embedded)
        hazards = torch.cat(hazards_embeddings, dim=-1)
        ### FINISHED RUNNING THROUGH EMBEDDING LAYERS ###

        ### FORM INFO FOR EACH FEATURE ###
        types = torch.cat([type1, type2], dim=-1)
        style = self.style_projection(style)
        weather = self.weather_projection(weather)
        types = self.types_projection(types)
        hazards = self.hazards_projection(hazards)
        removal = self.removal_projection(removal)
        ## END FORM INFO ###

        ### RUN THROUGH MULTIHEAD ATTENTION LAYERS ###
        species, _ = self.species_multihead_attn(species, species, species)
        species = self.species_layer_norm(species)

        style, _ = self.style_multihead_attn(style, species, species)
        style = self.style_layer_norm(style)

        weather, _ = self.weather_multihead_attn(weather, species, species)
        weather = self.weather_layer_norm(weather)

        types, _ = self.types_multihead_attn(types, species, species)
        types = self.types_layer_norm(types)

        hazards, _ = self.hazards_multihead_attn(hazards, species, species)
        hazards = self.hazards_layer_norm(hazards)

        removal, _ = self.removal_multihead_attn(removal, species, species)
        removal = self.removal_layer_norm(removal)
        
        ### FINISHED RUNNING THROUGH MULTIHEAD ATTN LAYERS ###


        # Combine features
        x = torch.cat((
            species,
            arch,
            dc,
            style,
            weather,
            types,
            hazards,
            removal
        ), dim=-1)

        x, _ = self.all_multihead_attn(x, x, x)

        # Project to embedding space
        x = self.layer_norm(x)
        x = F.relu(self.project(x))

        # Encode
        memory = self.encoder(x)
        
        # Generate causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
    
        # Decode
        output = self.decoder(x, memory, tgt_mask=tgt_mask)
        
        # Final projection
        logits = self.fc_out(output)
        return logits

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate(self, start_tokens, tokenizer, max_length=H.team_size, temperature=1.0, min_p=.05, top_k=20, min_k=H.min_k, dynamic_k=False, repetition_penalty=.25, weather_repetition_penalty=.25, hazard_rep_pen=.5, track_gradients=False):
        self.eval()
        
        current_sequence = start_tokens
        batch_size = current_sequence.size(0)
        log_probs_list = []

        context_manager = torch.enable_grad() if track_gradients else torch.no_grad()
        
        while current_sequence.size(1) < max_length:
            with context_manager:
                current_sequence = current_sequence.to(torch.long)
                
                # Get encoder memory for full sequence
                logits = self(current_sequence)
                next_token_logits = logits[:, -1, :]
                mask = torch.ones_like(next_token_logits, dtype=torch.bool, device=H.device) #Keep track of what needs to be masked out

                # Masking Dupes
                for i, batch in enumerate(next_token_logits):
                    used_mons = torch.tensor([mon[H.species].item() for mon in current_sequence[i]], device=H.device)
                    mask[i] = ~torch.isin(torch.tensor([features[H.species] for features in tokenizer], device=H.device), used_mons)
                #Finish Masking Dupes
                
                p_probs = F.softmax(next_token_logits, dim=-1) * mask
                highest_p, _ = torch.max(p_probs, dim=-1)  # Get max probability per batch

                for batch_idx in range(batch_size):
                    batch_min_p = highest_p[batch_idx] * min_p
                    batch_min_p_mask = (p_probs[batch_idx] >= batch_min_p)
                    mask[batch_idx] = mask[batch_idx] & batch_min_p_mask

                # Temperature adjustment
                if current_sequence.size(1) > 1:
                    temperature_reduction_factor = H.temp_redux
                    adjusted_temperature = temperature * (temperature_reduction_factor ** (current_sequence.size(1) - 1))
                else:
                    adjusted_temperature = temperature

                next_token_logits = next_token_logits / adjusted_temperature

                # Convert to probabilities
                next_token_logits.masked_fill_(~mask, float('-inf'))
                probs = F.softmax(next_token_logits, dim=-1)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample next Pokemon
                next_indices = torch.multinomial(probs, 1)
                
                # Convert indices to features
                next_features = self.index_to_features(next_indices, tokenizer)
                
                # Add new Pokemon to teams
                current_sequence = torch.cat([current_sequence, next_features], dim=1)

        if not track_gradients:
            return current_sequence.to(torch.long)
        else:
            return current_sequence.to(torch.long), probs

    def index_to_features(self, tokens, tokenizer):
        tokens_features = torch.zeros(H.BATCH_SIZE, 1, H.feature_size)
        tokens_features = tokens_features.to(H.device)
        for i, token in enumerate(tokens):
            feature_tensor = torch.from_numpy(np.array(tokenizer[token]).astype(np.float32)).to(torch.long)
            tokens_features[i, 0] = feature_tensor
        return tokens_features'''