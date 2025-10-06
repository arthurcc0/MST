import torch 
from .base_model import BasicClassifier
# from transformers import Dinov2Model
from transformers import AutoImageProcessor, AutoModel
from .utils.transformer_blocks import TransformerEncoderLayer
import torch.nn as nn
from einops import rearrange
from .extern.dinov2.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

def slices2rgb(tensor):
    # [B, 1, D, H, W] -> [B*D//3, 3, H, W]
    B, C, D, H, W = tensor.shape

    assert C == 1, "More than one channel"

    # If D is not divisible by 3, we need to pad by repeating the first slice
    if D % 3 != 0:
        padding_size = 3 - (D % 3)  # Find out how much padding is needed
        padding = tensor[:, :, :padding_size]  # Take the first slices to pad
        tensor = torch.cat([tensor, padding], dim=2)  # Concatenate along D axis
    
    # Reshape the tensor from [B, 1, D, H, W] to [B * (D // 3), 3, H, W]
    B, _, D, H, W = tensor.shape
    tensor = tensor.view(B, D // 3, 3, H, W)  # Reshape to [B, D//3, 3, H, W]
    tensor = tensor.reshape(-1, 3, H, W)  # [B*D//3, 3, H, W]
    
    return tensor 

   


class DinoV2ClassifierSlice(BasicClassifier):
    def __init__(
            self, 
            in_ch,
            out_ch,
            spatial_dims=2,
            pretrained=True,
            save_attn = False,
            rotary_positional_encoding=None,
            optimizer_kwargs={'lr': 1e-6, 'weight_decay': 1e-2},
            model_size = 's', # [s, b, l, 'g']
            use_registers = True,
            use_bottleneck=False,
            use_slice_pos_emb=False,
            enable_linear = True,
            enable_trans = True, # Deprecated 
            slice_fusion='transformer',
            freeze=False,
            **kwargs
        ):
        super().__init__(in_ch, out_ch, spatial_dims=spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)
        self.save_attn = save_attn
        self.attention_maps = []
        self.attention_maps_slice = []
        self.use_registers = use_registers
        self.slice_fusion_type = slice_fusion

        if pretrained:
            if use_registers:
                self.encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14_reg')
            else:
                self.encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
        else:
            Model = {'s': vit_small, 'b': vit_base, 'l':vit_large, 'g':vit_giant2 }[model_size]
            self.encoder = Model(patch_size=14, num_register_tokens=0)
   
        # Freeze backbone 
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    
        emb_ch = self.encoder.num_features 
        if use_bottleneck:
            self.bottleneck = nn.Linear(emb_ch, emb_ch//4)
            emb_ch = emb_ch//4 
        self.emb_ch = emb_ch

        if slice_fusion == 'transformer':
            if use_slice_pos_emb:
                self.slice_pos_emb = nn.Embedding(256, emb_ch) # WARNING: Assuming max. 256 slices

            self.slice_fusion = nn.TransformerEncoder(
                encoder_layer=TransformerEncoderLayer(
                    d_model=emb_ch,
                    nhead=12, 
                    dim_feedforward=1*emb_ch,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                    rotary_positional_encoding=rotary_positional_encoding
                ),
                num_layers=1,
                norm=nn.LayerNorm(emb_ch)
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion == 'linear':
            emb_ch = emb_ch*32
        elif slice_fusion == 'average':
            pass 

        self.linear = nn.Linear(emb_ch, out_ch) if enable_linear else nn.Identity()



        


    def forward(self, source, save_attn=False, src_key_padding_mask=None, **kwargs):   

        if save_attn:
            fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            self.attention_maps_slice = []
            self.attention_maps = []
            self.hooks = []
            self.register_hooks()


        x = source.to(self.device) # [B, C, D, H, W]
        B, C, *_ = x.shape
 

        x = rearrange(x, 'b c d h w -> (b d c) h w')
        x = x[:, None]
        x = x.repeat(1, 3, 1, 1) # Gray to RGB

        # x = slices2rgb(x) # [B, 1, D, H, W] -> [B*D//3, 3, H, W]

        x = self.encoder(x) # [(B D), C, H, W] -> [(B D), out] 

        # Bottleneck: force to focus on relevant features for classification 
        if hasattr(self, 'bottleneck'):
            x = self.bottleneck(x)
        
        # Slice fusion 
        x = rearrange(x, '(b d) e -> b d e', b=B)

        if hasattr(self, 'slice_pos_emb'):
            pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
            x += self.slice_pos_emb(pos)
        
        if self.slice_fusion_type == 'transformer':
            x = torch.concat([self.cls_token.repeat(B, 1, 1), x], dim=1)
 
            if src_key_padding_mask is not None: 
                src_key_padding_mask = src_key_padding_mask.to(self.device)
                src_key_padding_mask_cls = torch.zeros((B, 1), device=self.device, dtype=bool)
                src_key_padding_mask = torch.concat([src_key_padding_mask_cls, src_key_padding_mask], dim=1)# [Batch, L]
       
            x = self.slice_fusion(x, src_key_padding_mask=src_key_padding_mask)
            x = x[:, 0]
        elif self.slice_fusion_type == 'linear':
            x = rearrange(x, 'b d e -> b (d e)')
        elif self.slice_fusion_type == 'average':
            x = x.mean(dim=1, keepdim=False)

        if save_attn:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            self.deregister_hooks()

        # Logits 
        if kwargs.get('without_linear', False):
            return x 
        x = self.linear(x) 
        return x
    



    
    def get_slice_attention(self):
        attention_map_slice = self.attention_maps_slice[-1] # [B, Heads, 1+D(+regs), 1+D(+regs)]
        attention_map_slice = attention_map_slice[:, :, 0, 1:] # [B, Heads, D]
        attention_map_slice /= attention_map_slice.sum(dim=-1, keepdim=True)

        # Option 1:
        attention_map_slice = attention_map_slice.mean(dim=1)  # [B, D]
        attention_map_slice = attention_map_slice.view(-1) # [B*D]
        attention_map_slice = attention_map_slice[:, None, None] # [B*D, 1, 1]

        # Option 2:
        # attention_map_slice = rearrange(attention_map_slice, 'b d e -> (b e) d') # [B*D, Heads]
        # attention_map_slice = attention_map_slice[:, :, None] # [B*D, Heads, 1]

        return attention_map_slice

    def get_plane_attention(self):
        attention_map_dino = self.attention_maps[-1] # [B*D, Heads, 1+HW, 1+HW]
        img_slice = slice(5, None) if self.use_registers else slice(1, None) # see https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L264 
        attention_map_dino = attention_map_dino[:,:, 0, img_slice] # [B*D, Heads, HW]
        attention_map_dino[:,:,0] = 0
        attention_map_dino /= attention_map_dino.sum(dim=-1, keepdim=True)
        return attention_map_dino

    def get_attention_maps(self):
        attention_map_dino = self.get_plane_attention()
        attention_map_slice = self.get_slice_attention()
        
        attention_map = attention_map_slice*attention_map_dino
        return attention_map
    
    def get_attention_cls(self):
        """ Calculate the attention in the first layer starting from the CLS token in the last layer. """
        attention_to_cls = self.attention_maps[-1]
        # Propagate the attention backwards
        for attn in reversed(self.attention_maps[:-1]):
            attention_to_cls = torch.matmul(attn, attention_to_cls)
        
        # The attention to the first layer from the CLS token
        return attention_to_cls
    
    def register_hooks(self):
        def enable_attention(module):
            forward_orig = module.forward
            def forward_wrap(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return forward_orig(*args, **kwargs)
            module.forward = forward_wrap
            module.foward_orig = forward_orig

        def enable_attention2(mod):
                forward_orig = mod.forward
                def forward_wrap(self2, x):
                    # forward_orig.__self__
                    B, N, C = x.shape
                    qkv = self2.qkv(x).reshape(B, N, 3, self2.num_heads, C // self2.num_heads).permute(2, 0, 3, 1, 4)
                    
                    q, k, v = qkv[0] * self2.scale, qkv[1], qkv[2]
                    attn = q @ k.transpose(-2, -1)
           
                    attn = attn.softmax(dim=-1)
                    attn = attn if isinstance(self2.attn_drop, float) else self2.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self2.proj(x)
                    x = self2.proj_drop(x)

                    # Hook attention map 
                    self.attention_maps.append(attn)

                    return x
                
                mod.forward = lambda x: forward_wrap(mod, x)
                mod.foward_orig = forward_orig

        def append_attention_maps(module, input, output):
            self.attention_maps_slice.append(output[1])

        # Hook Dino Attention
        for name, mod in self.encoder.named_modules():
            if name.endswith('.attn'):
                enable_attention2(mod)

        # Hook Slice Attention
        for _, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                enable_attention(mod)
                self.hooks.append(mod.register_forward_hook(append_attention_maps))


    def deregister_hooks(self):
        for handle in self.hooks:
            handle.remove()

        # Dino Attention
        for name, mod in self.encoder.named_modules():
            if name.endswith('.attn'):
                mod.forward = mod.foward_orig
    
        # Slice Attention
        for _, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                mod.forward = mod.foward_orig



class DinoV3ClassifierSlice(BasicClassifier):
    def __init__(
            self, 
            in_ch,
            out_ch,
            spatial_dims=2,
            pretrained=True,
            save_attn = False,
            rotary_positional_encoding=None,
            optimizer_kwargs={'lr': 1e-6, 'weight_decay': 1e-2},
            model_size = 's', # [s, b, l, 'g']
            use_bottleneck=False,
            use_slice_pos_emb=False,
            enable_linear = True,
            enable_trans = True, # Deprecated 
            slice_fusion='transformer',
            freeze=False,
            **kwargs
        ):
        super().__init__(in_ch, out_ch, spatial_dims=spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)
        self.save_attn = save_attn
        self.attention_maps = []
        self.attention_maps_slice = []
        self.slice_fusion_type = slice_fusion
        self.model_size = model_size

        if pretrained:
            # Use official Meta DinoV3 weights via torch.hub for better attention extraction
            model_urls = {
                's': 'https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiYjNsY3Y3aDRsNGM0M2tzamV1b3J1cXViIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTg3MzY0ODJ9fX1dfQ__&Signature=uDC5hc1DpqbPP0EDRyVfibgojYt9CzQ3a3q9Hpfx1B%7E5IUhFHnhS3kaS25xF8mIIO5O20bodnF1BFNAUNfjr6rZzm1qJwfbUjgHw1RTYBV5b7c5lEFwMg7oFRz6qliOKBjePSgj78wstu82pnOrNqgdTRMW4moVYNJU1P5V1Y2ALXSQSXoQ0Y4llCzZeCECAAvTw3Tyutkygm3FqPrvty14xBNgUFJoSXGSSQ3r2ty%7ECgFoDsy1hhUEhtD3TIlhJEOGawS8kci2vxMzQSuFSEwNa8kibelXnOF1IEYY7x8yusLOb4A1psljuTJDNEG2BrcP08L4Ve66TpesUD3KYXw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=4145608132324923',
                'b': 'https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiYjNsY3Y3aDRsNGM0M2tzamV1b3J1cXViIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTg3MzY0ODJ9fX1dfQ__&Signature=uDC5hc1DpqbPP0EDRyVfibgojYt9CzQ3a3q9Hpfx1B%7E5IUhFHnhS3kaS25xF8mIIO5O20bodnF1BFNAUNfjr6rZzm1qJwfbUjgHw1RTYBV5b7c5lEFwMg7oFRz6qliOKBjePSgj78wstu82pnOrNqgdTRMW4moVYNJU1P5V1Y2ALXSQSXoQ0Y4llCzZeCECAAvTw3Tyutkygm3FqPrvty14xBNgUFJoSXGSSQ3r2ty%7ECgFoDsy1hhUEhtD3TIlhJEOGawS8kci2vxMzQSuFSEwNa8kibelXnOF1IEYY7x8yusLOb4A1psljuTJDNEG2BrcP08L4Ve66TpesUD3KYXw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=4145608132324923',
                'l': 'https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiYjNsY3Y3aDRsNGM0M2tzamV1b3J1cXViIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTg3MzY0ODJ9fX1dfQ__&Signature=uDC5hc1DpqbPP0EDRyVfibgojYt9CzQ3a3q9Hpfx1B%7E5IUhFHnhS3kaS25xF8mIIO5O20bodnF1BFNAUNfjr6rZzm1qJwfbUjgHw1RTYBV5b7c5lEFwMg7oFRz6qliOKBjePSgj78wstu82pnOrNqgdTRMW4moVYNJU1P5V1Y2ALXSQSXoQ0Y4llCzZeCECAAvTw3Tyutkygm3FqPrvty14xBNgUFJoSXGSSQ3r2ty%7ECgFoDsy1hhUEhtD3TIlhJEOGawS8kci2vxMzQSuFSEwNa8kibelXnOF1IEYY7x8yusLOb4A1psljuTJDNEG2BrcP08L4Ve66TpesUD3KYXw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=4145608132324923',
                'g': 'https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiYjNsY3Y3aDRsNGM0M2tzamV1b3J1cXViIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTg3MzY0ODJ9fX1dfQ__&Signature=uDC5hc1DpqbPP0EDRyVfibgojYt9CzQ3a3q9Hpfx1B%7E5IUhFHnhS3kaS25xF8mIIO5O20bodnF1BFNAUNfjr6rZzm1qJwfbUjgHw1RTYBV5b7c5lEFwMg7oFRz6qliOKBjePSgj78wstu82pnOrNqgdTRMW4moVYNJU1P5V1Y2ALXSQSXoQ0Y4llCzZeCECAAvTw3Tyutkygm3FqPrvty14xBNgUFJoSXGSSQ3r2ty%7ECgFoDsy1hhUEhtD3TIlhJEOGawS8kci2vxMzQSuFSEwNa8kibelXnOF1IEYY7x8yusLOb4A1psljuTJDNEG2BrcP08L4Ve66TpesUD3KYXw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=4145608132324923'
            }
            
            try:
                # Try loading official Meta weights first
                model_url = model_urls[model_size]
                self.encoder = torch.hub.load('facebookresearch/dinov3', f'dinov3_vit{model_size}16', weights=model_url)
                self.encoder.num_features = self.encoder.embed_dim
                self.use_official_weights = True
                print(f"Loaded official DinoV3 {model_size} weights from Meta")
            except Exception as e:
                print(f"Failed to load official weights, falling back to Hugging Face: {e}")
                # Fallback to Hugging Face transformers
                model_name = {
                    's': 'facebook/dinov3-vits16-pretrain-lvd1689m',
                    'b': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                    'l': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
                    'g': 'facebook/dinov3-vitg14-pretrain-lvd1689m'
                }[model_size]
                self.image_processor = AutoImageProcessor.from_pretrained(model_name)
                self.encoder = AutoModel.from_pretrained(model_name)
                self.encoder.num_features = self.encoder.config.hidden_size
                self.use_official_weights = False
        else:
            Model = {'s': vit_small, 'b': vit_base, 'l':vit_large, 'g':vit_giant2 }[model_size]
            self.encoder = Model(patch_size=14, num_register_tokens=0)
   
        # Freeze backbone 
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        emb_ch = self.encoder.num_features 
        if use_bottleneck:
            self.bottleneck = nn.Linear(emb_ch, emb_ch//4)
            emb_ch = emb_ch//4 
        self.emb_ch = emb_ch

        if slice_fusion == 'transformer':
            if use_slice_pos_emb:
                self.slice_pos_emb = nn.Embedding(256, emb_ch) # WARNING: Assuming max. 256 slices

            self.slice_fusion = nn.TransformerEncoder(
                encoder_layer=TransformerEncoderLayer(
                    d_model=emb_ch,
                    nhead=12, 
                    dim_feedforward=1*emb_ch,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                    rotary_positional_encoding=rotary_positional_encoding
                ),
                num_layers=1,
                norm=nn.LayerNorm(emb_ch)
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion == 'linear':
            emb_ch = emb_ch*32
        elif slice_fusion == 'average':
            pass 

        self.linear = nn.Linear(emb_ch, out_ch) if enable_linear else nn.Identity()
        
        # Store attention hooks
        self.attention_hooks = []

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights from DinoV3 encoder"""
        self.attention_maps = []
        
        def attention_hook(module, input, output):
            # For DinoV3, attention weights might be in different positions
            # Try to capture attention from the multi-head attention modules
            if hasattr(module, 'attention') and hasattr(module.attention, 'get_attention_map'):
                attn_weights = module.attention.get_attention_map()
                if attn_weights is not None:
                    self.attention_maps.append(attn_weights.detach())
            elif len(output) > 1 and isinstance(output[1], torch.Tensor):
                # Some attention modules return (output, attention_weights)
                attn_weights = output[1]
                if attn_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    self.attention_maps.append(attn_weights.detach())
        
        # Hook into transformer blocks
        for name, module in self.encoder.named_modules():
            if 'attention' in name or 'attn' in name:
                handle = module.register_forward_hook(attention_hook)
                self.attention_hooks.append(handle)

    def _remove_attention_hooks(self):
        """Remove all attention hooks"""
        for handle in self.attention_hooks:
            handle.remove()
        self.attention_hooks = []

    def forward(self, source, save_attn=False, src_key_padding_mask=None, **kwargs):   
        
        if save_attn:
            fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            self.attention_maps_slice = []
            self.attention_maps = []
            self.hooks = []
            # Try both HF method and hooks
            self._register_attention_hooks()
            self.register_hooks()

        x = source.to(self.device) # [B, C, D, H, W]
        B, C, *_ = x.shape

        # Fix preprocessing - handle dimensions properly
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        
        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if hasattr(self, 'use_official_weights') and self.use_official_weights:
            # Use official Meta DinoV3 weights - works like DinoV2
            x = self.encoder(x)  # Direct forward pass like DinoV2
        elif hasattr(self, 'image_processor'):
            try:
                # Method 1: Try standard HF approach
                inputs = self.image_processor(x, return_tensors="pt")
                
                # Handle different return types from image processor
                if hasattr(inputs, 'pixel_values'):
                    inputs = inputs.pixel_values.to(self.device)
                elif isinstance(inputs, dict) and 'pixel_values' in inputs:
                    inputs = inputs['pixel_values'].to(self.device)
                elif hasattr(inputs, 'data') and isinstance(inputs.data, dict) and 'pixel_values' in inputs.data:
                    inputs = inputs.data['pixel_values'].to(self.device)
                elif isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                else:
                    raise ValueError(f"Unexpected input type from processor: {type(inputs)}")
                
                # Force attention output
                outputs = self.encoder(inputs, output_attentions=True, return_dict=True)
                x = outputs.pooler_output
                
                # Debug attention extraction
                if save_attn:
                    print(f"DinoV3 Debug: Model type: {type(self.encoder)}")
                    print(f"DinoV3 Debug: Output type: {type(outputs)}")
                    print(f"DinoV3 Debug: Output keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'No keys'}")
                    
                    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        print(f"DinoV3 Debug: Found {len(outputs.attentions)} attention layers")
                        for i, attn in enumerate(outputs.attentions):
                            print(f"DinoV3 Debug: Layer {i} attention shape: {attn.shape}")
                        # Store all attention layers
                        self.attention_maps.extend([attn.detach() for attn in outputs.attentions])
                    else:
                        print("DinoV3 Debug: No attentions found in outputs")
                        print("DinoV3 Debug: Trying alternative extraction...")
                        
                        # Method 2: Try to extract from last_hidden_state or other outputs
                        if hasattr(outputs, 'last_hidden_state'):
                            print(f"DinoV3 Debug: Found last_hidden_state: {outputs.last_hidden_state.shape}")
                        
                        # Method 3: Check if hooks captured anything
                        if self.attention_maps:
                            print(f"DinoV3 Debug: Hooks captured {len(self.attention_maps)} attention maps")
                        else:
                            print("DinoV3 Debug: No attention captured through hooks either")
                            
            except Exception as e:
                print(f"DinoV3 Debug: Error in processing: {e}")
                # Fallback: process images individually
                processed_images = []
                for i in range(x.shape[0]):
                    img = x[i:i+1]  # Keep batch dimension
                    try:
                        processed = self.image_processor(img, return_tensors="pt")
                        if hasattr(processed, 'pixel_values'):
                            processed = processed.pixel_values
                        elif isinstance(processed, dict) and 'pixel_values' in processed:
                            processed = processed['pixel_values']
                        elif hasattr(processed, 'data') and 'pixel_values' in processed.data:
                            processed = processed.data['pixel_values']
                        
                        # Ensure it's a tensor
                        if isinstance(processed, torch.Tensor):
                            processed_images.append(processed)
                        else:
                            print(f"DinoV3 Debug: Unexpected processed type: {type(processed)}")
                            # Use direct tensor if processor fails
                            processed_images.append(img)
                            
                    except Exception as img_e:
                        print(f"DinoV3 Debug: Error processing image {i}: {img_e}")
                        # Use direct tensor if processor fails
                        processed_images.append(img)
                
                if processed_images:
                    inputs = torch.cat(processed_images, dim=0).to(self.device)
                    outputs = self.encoder(inputs, output_attentions=True, return_dict=True)
                    x = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        else:
            # Non-pretrained model path
            x = self.encoder(x)

        # Bottleneck: force to focus on relevant features for classification 
        if hasattr(self, 'bottleneck'):
            x = self.bottleneck(x)
        
        # Slice fusion 
        x = rearrange(x, '(b d) e -> b d e', b=B)

        if hasattr(self, 'slice_pos_emb'):
            pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
            x += self.slice_pos_emb(pos)
        
        if self.slice_fusion_type == 'transformer':
            x = torch.concat([self.cls_token.repeat(B, 1, 1), x], dim=1)
 
            if src_key_padding_mask is not None: 
                src_key_padding_mask = src_key_padding_mask.to(self.device)
                src_key_padding_mask_cls = torch.zeros((B, 1), device=self.device, dtype=bool)
                src_key_padding_mask = torch.concat([src_key_padding_mask_cls, src_key_padding_mask], dim=1)# [Batch, L]
       
            x = self.slice_fusion(x, src_key_padding_mask=src_key_padding_mask)
            x = x[:, 0]
        elif self.slice_fusion_type == 'linear':
            x = rearrange(x, 'b d e -> b (d e)')
        elif self.slice_fusion_type == 'average':
            x = x.mean(dim=1, keepdim=False)

        if save_attn:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            self.deregister_hooks()
            self._remove_attention_hooks()

        # Logits 
        if kwargs.get('without_linear', False):
            return x 
        x = self.linear(x) 
        return x
    
    def get_attention_spatial_shape(self):
        """Get the spatial dimensions for attention reshaping based on model size"""
        if self.model_size == 'g':
            return 16, 16  # 224/14 = 16 patches per side for giant model
        else:
            return 14, 14  # 224/16 = 14 patches per side for other models
    
    def get_slice_attention(self):
        if not self.attention_maps_slice:
            print("Warning: No slice attention maps available")
            return None
            
        attention_map_slice = self.attention_maps_slice[-1] # [B, Heads, 1+D(+regs), 1+D(+regs)]
        attention_map_slice = attention_map_slice[:, :, 0, 1:] # [B, Heads, D]
        attention_map_slice /= (attention_map_slice.sum(dim=-1, keepdim=True) + 1e-8)

        # Option 1:
        attention_map_slice = attention_map_slice.mean(dim=1)  # [B, D]
        attention_map_slice = attention_map_slice.view(-1) # [B*D]
        attention_map_slice = attention_map_slice[:, None, None] # [B*D, 1, 1]

        return attention_map_slice

    def get_plane_attention(self):
        if not self.attention_maps:
            print("ERROR: No attention maps stored!")
            return None
            
        print(f"Processing attention maps: {len(self.attention_maps)} maps available")
        attention_map_dino = self.attention_maps[-1] # [B*D, Heads, seq_len, seq_len]
        print(f"Raw attention shape: {attention_map_dino.shape}")
        
        # Verify attention shape
        if len(attention_map_dino.shape) != 4:
            print(f"ERROR: Expected 4D attention tensor, got {len(attention_map_dino.shape)}D")
            return None
        
        batch_size, num_heads, seq_len, seq_len2 = attention_map_dino.shape
        
        # Calculate expected number of patches based on model
        h_patches, w_patches = self.get_attention_spatial_shape()
        expected_tokens = h_patches * w_patches + 1  # +1 for CLS token
        
        print(f"Expected tokens: {expected_tokens}, Actual: {seq_len}")
        
        if seq_len < 2:
            print(f"ERROR: Too few tokens in attention: {seq_len}")
            return None
        
        # For DinoV3, extract attention from CLS token to image patches
        cls_to_patches = attention_map_dino[:, :, 0, 1:]  # [B*D, Heads, num_patches]
        print(f"After CLS extraction: {cls_to_patches.shape}")
        
        # Handle edge case where we have fewer patches than expected
        if cls_to_patches.shape[-1] == 0:
            print("ERROR: No patch tokens found after CLS extraction")
            return None
        
        # Set first patch to 0 and normalize (optional - remove if not needed)
        cls_to_patches_normalized = cls_to_patches.clone()
        if cls_to_patches_normalized.shape[-1] > 0:
            cls_to_patches_normalized[:, :, 0] = 0
        cls_to_patches_normalized = cls_to_patches_normalized / (cls_to_patches_normalized.sum(dim=-1, keepdim=True) + 1e-8)
        
        print(f"Final attention shape: {cls_to_patches_normalized.shape}")
        print(f"Attention stats - min: {cls_to_patches_normalized.min():.6f}, max: {cls_to_patches_normalized.max():.6f}")
        
        # Average across attention heads
        cls_to_patches_normalized = cls_to_patches_normalized.mean(dim=1)  # [B*D, num_patches]
        
        return cls_to_patches_normalized

    def get_attention_maps(self):
        attention_map_dino = self.get_plane_attention()
        if attention_map_dino is None:
            return None
            
        attention_map_slice = self.get_slice_attention()
        if attention_map_slice is None:
            # Return just plane attention if slice attention is not available
            return attention_map_dino
        
        # Debug shapes
        print(f"Plane attention shape: {attention_map_dino.shape}")
        print(f"Slice attention shape: {attention_map_slice.shape}")
        
        # Ensure compatible dimensions for multiplication
        if attention_map_dino.dim() == 2:  # [B*D, num_patches]
            B_times_D, num_patches = attention_map_dino.shape
            
            # attention_map_slice should be [B*D, 1, 1] from get_slice_attention
            if attention_map_slice.dim() == 3 and attention_map_slice.shape[1:] == (1, 1):
                # Broadcast slice attention to match plane attention
                attention_map_slice = attention_map_slice.squeeze(-1).squeeze(-1)  # [B*D]
                attention_map_slice = attention_map_slice.unsqueeze(1)  # [B*D, 1]
                attention_map_slice = attention_map_slice.expand(-1, num_patches)  # [B*D, num_patches]
            elif attention_map_slice.dim() == 1:
                # If slice attention is [B*D], expand to [B*D, num_patches]
                attention_map_slice = attention_map_slice.unsqueeze(1).expand(-1, num_patches)
            elif attention_map_slice.dim() == 3:
                # Handle [B*D, 1, 1] case
                attention_map_slice = attention_map_slice.view(B_times_D, -1)
                if attention_map_slice.shape[1] == 1:
                    attention_map_slice = attention_map_slice.expand(-1, num_patches)
        
        print(f"After reshaping - Plane: {attention_map_dino.shape}, Slice: {attention_map_slice.shape}")
        
        # Now multiply element-wise
        attention_map = attention_map_slice * attention_map_dino
        return attention_map
    
    def get_attention_cls(self):
        """ Calculate the attention in the first layer starting from the CLS token in the last layer. """
        if not self.attention_maps:
            return None
            
        attention_to_cls = self.attention_maps[-1]
        # Propagate the attention backwards
        for attn in reversed(self.attention_maps[:-1]):
            attention_to_cls = torch.matmul(attn, attention_to_cls)
        
        # The attention to the first layer from the CLS token
        return attention_to_cls
    
    def register_hooks(self):
        def enable_attention(module):
            if hasattr(module, 'forward'):
                forward_orig = module.forward
                def forward_wrap(*args, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    return forward_orig(*args, **kwargs)
                module.forward = forward_wrap
                module.forward_orig = forward_orig

        def enable_attention_dinov3(mod):
            """Hook for official DinoV3 attention layers (handles rope parameter)"""
            forward_orig = mod.forward
            def forward_wrap(self2, x, rope=None):
    # DinoV3 attention capture with rope support
                B, N, C = x.shape
                qkv = self2.qkv(x).reshape(B, N, 3, self2.num_heads, C // self2.num_heads).permute(2, 0, 3, 1, 4)
                
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Apply rotary positional encoding if provided
                if rope is not None:
                    # rope is typically a tuple (cos, sin) for DinoV3
                    if hasattr(self2, 'rope') and hasattr(self2.rope, 'apply_rotary_emb'):
                        q = self2.rope.apply_rotary_emb(q, rope)
                        k = self2.rope.apply_rotary_emb(k, rope)
                    elif isinstance(rope, tuple) and len(rope) == 2:
                        # Handle rope as (cos, sin) tuple - apply manually if needed
                        cos, sin = rope
                        # For now, skip rope application to avoid errors
                        pass
                
                # Scale query
                q = q * self2.scale
                
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = attn if isinstance(self2.attn_drop, float) else self2.attn_drop(attn)
                
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self2.proj(x)
                x = self2.proj_drop(x)

                # Hook attention map 
                self.attention_maps.append(attn.detach())

                return x
            mod.forward = lambda x, rope=None: forward_wrap(mod, x, rope)
            mod.forward_orig = forward_orig

        def append_attention_maps(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if isinstance(attn_weights, torch.Tensor) and attn_weights.dim() == 4:
                    self.attention_maps_slice.append(attn_weights.detach())

        # Hook DinoV3 Attention based on model type
        if hasattr(self, 'use_official_weights') and self.use_official_weights:
            # Official Meta weights - hook like DinoV2
            for name, mod in self.encoder.named_modules():
                if name.endswith('.attn'):
                    enable_attention_dinov3(mod)
        
        # Hook Slice Attention
        if hasattr(self, 'slice_fusion'):
            for _, mod in self.slice_fusion.named_modules():
                if isinstance(mod, nn.MultiheadAttention):
                    enable_attention(mod)
                    handle = mod.register_forward_hook(append_attention_maps)
                    self.hooks.append(handle)

    def deregister_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
        # Restore original forward methods for official DinoV3
        if hasattr(self, 'use_official_weights') and self.use_official_weights:
            for name, mod in self.encoder.named_modules():
                if name.endswith('.attn') and hasattr(mod, 'forward_orig'):
                    mod.forward = mod.forward_orig
                    delattr(mod, 'forward_orig')
        
        # Restore original forward methods for slice fusion
        if hasattr(self, 'slice_fusion'):
            for _, mod in self.slice_fusion.named_modules():
                if isinstance(mod, nn.MultiheadAttention) and hasattr(mod, 'forward_orig'):
                    mod.forward = mod.forward_orig
                    delattr(mod, 'forward_orig')

    def test_attention_extraction(self):
        """Test function to debug attention extraction"""
        print("Testing attention extraction...")
        
        # Create a simple test input
        test_input = torch.randn(1, 1, 4, 224, 224).to(self.device)  # Small test case
        
        # Force save_attn=True
        with torch.no_grad():
            output = self.forward(test_input, save_attn=True)
        
        print(f"Test completed. Output shape: {output.shape}")
        print(f"DinoV3 attention maps collected: {len(self.attention_maps)}")
        print(f"Slice attention maps collected: {len(self.attention_maps_slice)}")
        
        if self.attention_maps:
            for i, attn in enumerate(self.attention_maps):
                print(f"Attention map {i} shape: {attn.shape}")
            
            # Test attention processing
            plane_attn = self.get_plane_attention()
            if plane_attn is not None:
                print(f"Plane attention extracted successfully: {plane_attn.shape}")
                return True
            else:
                print("Failed to extract plane attention")
                return False
        else:
            print("No attention maps captured - check model compatibility")
            return False