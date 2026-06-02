import torch 
from .base_model import BasicClassifier
from torchmetrics import AUROC, Accuracy
from transformers import pipeline
from .utils.transformer_blocks import TransformerEncoderLayer
import torch.nn as nn
from einops import rearrange
from .extern.dinov2.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from .extern.dinov2.tokenizer import Tokenizer
import requests
from io import BytesIO
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def tensor_to_pil(tensors):
    """Convert a batch of tensors to a list of PIL images."""
    # Detach tensors and move to CPU before converting to PIL images
    return [to_pil_image(tensor.detach().cpu()) for tensor in tensors]

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

# def merge_pre_bn(module, pre_bn_1, pre_bn_2): 
#     weight = module.weight.data
#     if module.bias is None:
#         zeros = torch.zeros(module.out_channels, device=module.weight.device).type(weight.type())
#         module.bias = nn.Parameter(zeros)
#     bias = module.bias.data
#     if pre_bn_2 is None:
#         assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
#         assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"   
#         scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
#         extra_weight = scale * pre_bn_1.weight
#         extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
#     else:
# class DinoV2ClassifierSliceKAN(BaseClassifier):
#     def __init__(self):


class DinoClassifierSlice(BasicClassifier):
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
            model_version='v2',
            use_registers = False,
            use_bottleneck=False,
            use_slice_pos_emb=False,
            enable_linear = True,
            enable_trans = True, # Deprecated 
            slice_fusion='transformer',
            freeze=False,
            encoder_lr=None,
            **kwargs
        ):
        kwargs.pop('paired_sampling', None)
        kwargs.pop('freeze_backbone', None)
        kwargs.pop('unfreeze_encoder_blocks', None)
        kwargs.pop('learning_rate', None)
        kwargs.pop('class_weight', None)
        loss_kwargs = kwargs.pop('loss_kwargs', {})
        super().__init__(
            in_ch,
            out_ch,
            spatial_dims=spatial_dims,
            optimizer_kwargs=optimizer_kwargs,
            loss_kwargs=loss_kwargs,
        )
        self.encoder_lr = encoder_lr
        self.save_attn = save_attn
        self.attention_maps = []
        self.attention_maps_slice = []
        self.use_registers = use_registers
        self.slice_fusion_type = slice_fusion
        self.model_version = model_version

        if pretrained:
            if model_version == 'v2':
                if use_registers:
                    self.encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14_reg')
                else:
                    self.encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
            elif model_version == 'v3':
                self.encoder = pipeline(
                    task="image-feature-extraction",
                    model="facebook/dinov3-vits16-pretrain-lvd1689m",
                    device=self.device,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    token=None
                )
        else:
            Model = {'s': vit_small, 'b': vit_base, 'l':vit_large, 'g':vit_giant2 }[model_size]
            self.encoder = Model(patch_size=14, num_register_tokens=0)
   
        # Freeze backbone 
        if freeze:
            # For DINOv3 pipeline, the underlying model is frozen by default.
            # For DINOv2, we freeze manually.
            if hasattr(self.encoder, 'parameters'):
                for param in self.encoder.parameters():
                    param.requires_grad = False

        if model_version == 'v2':
            emb_ch = self.encoder.num_features 
        else:
            # DINOv3 ViT-S models have a hidden size of 384
            emb_ch = 384

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

    def unfreeze_encoder_last_blocks(self, n_blocks: int) -> int:
        """Unfreeze the last ``n_blocks`` DINOv2 ViT blocks (MST + classifier stay trainable)."""
        if n_blocks <= 0:
            return 0
        if self.model_version != 'v2' or not hasattr(self.encoder, 'blocks'):
            print(
                "Warning: partial encoder unfreeze is only implemented for DINOv2 hub ViT "
                f"(model_version={self.model_version!r})."
            )
            return 0
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = self.encoder.blocks[-n_blocks:]
        for block in blocks:
            for param in block.parameters():
                param.requires_grad = True
        return len(blocks)

    def configure_optimizers(self):
        encoder_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('encoder.'):
                encoder_params.append(param)
            else:
                head_params.append(param)

        opt_kwargs = dict(self.optimizer_kwargs)
        head_lr = float(opt_kwargs.pop('lr', 1e-6))
        enc_lr = float(self.encoder_lr) if self.encoder_lr is not None else head_lr * 0.1

        if encoder_params and head_params:
            param_groups = [
                {'params': encoder_params, 'lr': enc_lr},
                {'params': head_params, 'lr': head_lr},
            ]
        else:
            param_groups = [p for p in self.parameters() if p.requires_grad]

        optimizer = self.optimizer(param_groups, **opt_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}]
        return optimizer

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

        if self.model_version == 'v2':
            x = self.encoder(x) # [(B D), C, H, W] -> [(B D), out] 
        elif self.model_version == 'v3':
            pil_images = tensor_to_pil(x)
            features = self.encoder(pil_images, pool=True)
            # The pipeline returns a list of lists; we extract the tensor from the inner list
            # The pipeline returns a list of lists of tensors, so we flatten and stack them
            x = torch.stack([torch.tensor(f[0]) for f in features]).to(self.device)

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
                    # Fix: attn_drop is a float, not a callable
                    if self2.training and self2.attn_drop > 0:
                        attn = torch.nn.functional.dropout(attn, p=self2.attn_drop)

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


class DinoClassifierPaired(BasicClassifier):
    def __init__(self, in_ch, out_ch, spatial_dims=2, **kwargs):
        kwargs.pop('paired_sampling', None)
        super().__init__(in_ch, out_ch, spatial_dims=spatial_dims, **kwargs)
        self.feature_extractor = DinoClassifierSlice(in_ch, out_ch, spatial_dims, **kwargs)
        emb_ch = self.feature_extractor.emb_ch
        self.classifier_head = nn.Linear(emb_ch * 2, 1) # Binary classifier for the pair
        self.loss_func = nn.BCEWithLogitsLoss() # Binary classifier for the pair

        # Re-configure metrics for binary classification, overriding the base class default
        binary_auc_kwargs = {"task": "binary"}
        binary_acc_kwargs = {"task": "binary"}
        self.auc_roc = nn.ModuleDict({state: AUROC(**binary_auc_kwargs) for state in ["train_", "val_", "test_"]})
        self.acc = nn.ModuleDict({state: Accuracy(**binary_acc_kwargs) for state in ["train_", "val_", "test_"]})

        self.batch_size = kwargs.get('batch_size', 2)

    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        # Remap the keys from the checkpoint to match the feature_extractor structure
        new_state_dict = {}
        for key, value in pretrained_weights.items():
            if not key.startswith('feature_extractor.'):
                new_key = f'feature_extractor.{key}'
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load the remapped weights
        # Set strict=False because the classifier head will be different
        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, source_malignant, source_benign, **kwargs):
        # Extract features for both images
        features_malignant = self.feature_extractor(source_malignant, without_linear=True, **kwargs)
        features_benign = self.feature_extractor(source_benign, without_linear=True, **kwargs)
        
        # Concatenate features
        combined_features = torch.cat([features_malignant, features_benign], dim=1)
        
        # Get logits from the classifier head
        logits = self.classifier_head(combined_features)
        return logits

    def training_step(self, batch, batch_idx):
        source_malignant = batch['source_malignant']
        source_benign = batch['source_benign']
        
        target = batch['target'].float().unsqueeze(1)

        output = self(source_malignant=source_malignant, source_benign=source_benign)
        loss = self.loss_func(output, target)

        # Update and log training metrics
        preds = torch.sigmoid(output)
        self.acc["train_"].update(preds, target.long())
        self.auc_roc["train_"].update(preds, target.long())

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source_malignant = batch['source_malignant']
        source_benign = batch['source_benign']
        target = batch['target'].float().unsqueeze(1)

        output = self(source_malignant=source_malignant, source_benign=source_benign)
        loss = self.loss_func(output, target)

        # For metrics, we need to provide predictions and targets
        preds = torch.sigmoid(output)
        self.acc["val_"].update(preds, target.long())
        self.auc_roc["val_"].update(preds, target.long())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'preds': output, 'target': target}




class DinoTxtClassifier(BasicClassifier):
    def __init__(
        self,
        in_ch,
        out_ch,
        spatial_dims=2,
        optimizer_kwargs={'lr': 1e-6, 'weight_decay': 1e-2},
        freeze_vision=True,
        freeze_text=False,
        use_slice_fusion=True,
        slice_fusion='transformer',
        model_size='l',
        model_version='v2',
        **kwargs
    ):
        kwargs.pop('paired_sampling', None)
        super().__init__(in_ch, out_ch, spatial_dims=spatial_dims, optimizer_kwargs=optimizer_kwargs)
        
        # Set batch_size attribute for logging
        self.batch_size = kwargs.get('batch_size', 16)
        self.model_version = model_version
        
        # Initialize vision encoder
        if model_version == 'v2':
            if model_size == 'l':
                self.vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
                vision_embed_dim = 1024
            elif model_size == 'b':
                self.vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                vision_embed_dim = 768
            else:
                self.vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                vision_embed_dim = 384
        elif model_version == 'v3':
            self.vision_encoder = pipeline(
                task="image-feature-extraction",
                model="facebook/dinov3-vits16-pretrain-lvd1689m",
                device=self.device,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                token=None
            )
            # DINOv3 ViT-S models have a hidden size of 384
            vision_embed_dim = 384
        
        # Initialize text encoder with standard PyTorch transformer
        self.text_embedding = nn.Embedding(49408, 512)  # vocab_size, embed_dim
        self.text_pos_embedding = nn.Parameter(torch.randn(77, 512))  # max_seq_len, embed_dim
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Initialize tokenizer
        self.tokenizer = self._get_tokenizer()
        
        # Freeze/unfreeze components
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Set embedding dimensions
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = 512
        
        # Projection layers to align dimensions
        self.vision_projection = nn.Linear(vision_embed_dim, 512)
        self.text_projection = nn.Linear(512, 512)
        
        # Slice fusion for 3D medical images
        self.use_slice_fusion = use_slice_fusion
        self.slice_fusion_type = slice_fusion
        
        if use_slice_fusion and slice_fusion == 'transformer':
            self.slice_fusion = nn.TransformerEncoder(
                encoder_layer=TransformerEncoderLayer(
                    d_model=512,
                    nhead=8,
                    dim_feedforward=512,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True
                ),
                num_layers=1,
                norm=nn.LayerNorm(512)
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        
        # Classification head
        final_dim = 512 * 2  # Combined image + text features
        self.classifier = nn.Linear(final_dim, out_ch)
        self.loss_func = nn.CrossEntropyLoss()
        
    def _get_tokenizer(self):
        """Initialize tokenizer with fallback"""
        try:
            from .extern.dinov2.simple_tokenizer import SimpleTokenizer
            url = "https://dl.fbaipublicfiles.com/dinov2/thirdparty/bpe_simple_vocab_16e6.txt.gz"
            response = requests.get(url)
            response.raise_for_status()
            file_buf = BytesIO(response.content)
            return Tokenizer(vocab_path=file_buf)
        except Exception as e:
            print(f"Warning: Could not initialize tokenizer: {e}")
            return None
        
    def preprocess_image(self, source):
        """Convert 3D medical image to 2D slices for DinoV2 processing"""
        x = source.to(self.device)  # [B, C, D, H, W]
        B, C, D, H, W = x.shape
        
        # Convert to 2D slices and repeat to RGB
        x = rearrange(x, 'b c d h w -> (b d) c h w')  # [B*D, C, H, W]
        x = x.repeat(1, 3, 1, 1)  # [B*D, 3, H, W] - repeat grayscale to RGB
        
        return x, B, D
    
    def process_image_slices(self, x, B, D, src_key_padding_mask=None):
        """Process image slices and fuse them"""
        # Get image features from vision encoder
        if self.model_version == 'v2':
            image_features = self.vision_encoder(x)  # [B*D, vision_embed_dim]
        elif self.model_version == 'v3':
            pil_images = tensor_to_pil(x)
            features = self.vision_encoder(pil_images, pool=True)
            # The pipeline returns a list of lists; we extract the tensor from the inner list
            image_features = torch.stack([torch.tensor(f[0]) for f in features]).to(self.device)
        image_features = self.vision_projection(image_features)  # [B*D, 512]
        
        if self.use_slice_fusion:
            # Reshape back to batch and slice dimensions
            image_features = rearrange(image_features, '(b d) e -> b d e', b=B)
            
            if self.slice_fusion_type == 'transformer':
                # Add CLS token and apply transformer fusion
                image_features = torch.cat([self.cls_token.repeat(B, 1, 1), image_features], dim=1)
                
                if src_key_padding_mask is not None:
                    src_key_padding_mask = src_key_padding_mask.to(self.device)
                    src_key_padding_mask_cls = torch.zeros((B, 1), device=self.device, dtype=bool)
                    src_key_padding_mask = torch.cat([src_key_padding_mask_cls, src_key_padding_mask], dim=1)
                
                image_features = self.slice_fusion(image_features, src_key_padding_mask=src_key_padding_mask)
                image_features = image_features[:, 0]  # Use CLS token
            elif self.slice_fusion_type == 'average':
                image_features = image_features.mean(dim=1)
        else:
            # Simple average pooling across slices
            image_features = rearrange(image_features, '(b d) e -> b d e', b=B)
            image_features = image_features.mean(dim=1)
        
        return image_features
    
    def process_text(self, text_reports):
        """Process text reports using standard PyTorch transformer"""
        if self.tokenizer is None:
            # Return zero features if tokenizer is not available
            batch_size = len(text_reports) if isinstance(text_reports, list) else text_reports.size(0)
            return torch.zeros(batch_size, 512, device=self.device)
        
        if isinstance(text_reports[0], str):
            # Tokenize text reports
            tokenized_text = self.tokenizer.tokenize(text_reports, context_length=77)
            tokenized_text = tokenized_text.to(self.device)
        else:
            # Assume already tokenized
            tokenized_text = text_reports.to(self.device)
        
        # Embed tokens and add positional encoding
        seq_len = tokenized_text.size(1)
        text_embeds = self.text_embedding(tokenized_text)  # [B, seq_len, 512]
        text_embeds = text_embeds + self.text_pos_embedding[:seq_len]  # Add positional encoding
        
        # Pass through transformer encoder
        text_features = self.text_encoder(text_embeds)  # [B, seq_len, 512]
        
        # Pool text features (use first token)
        text_features = text_features[:, 0]  # [B, 512]
        text_features = self.text_projection(text_features)
        
        return text_features
    
    def forward(self, source, text_reports=None, src_key_padding_mask=None, **kwargs):
        """
        Forward pass for multimodal classification
        
        Args:
            source: Medical images [B, C, D, H, W]
            text_reports: List of text reports or tokenized text [B, seq_len]
            src_key_padding_mask: Padding mask for slices [B, D]
        """
        # Process images
        x, B, D = self.preprocess_image(source)
        image_features = self.process_image_slices(x, B, D, src_key_padding_mask)
        
        # Process text if provided
        if text_reports is not None:
            text_features = self.process_text(text_reports)
            
            # Combine image and text features
            combined_features = torch.cat([image_features, text_features], dim=-1)
        else:
            # Use only image features (duplicate to maintain dimension)
            combined_features = torch.cat([image_features, image_features], dim=-1)
        
        # Classification
        if kwargs.get('without_linear', False):
            return combined_features
        
        logits = self.classifier(combined_features)
        return logits
    
    def training_step(self, batch, batch_idx):
        source = batch['source']
        target = batch['target']
        text_reports = batch.get('text_reports', None)
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        
        output = self(source, text_reports=text_reports, src_key_padding_mask=src_key_padding_mask)
        loss = self.loss_func(output, target)

        # Update and log metrics
        preds = torch.softmax(output, dim=1)
        self.acc["train_"].update(preds, target)
        self.auc_roc["train_"].update(preds, target)
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        source = batch['source']
        target = batch['target']
        text_reports = batch.get('text_reports', None)
        src_key_padding_mask = batch.get('src_key_padding_mask', None)
        
        output = self(source, text_reports=text_reports, src_key_padding_mask=src_key_padding_mask)
        loss = self.loss_func(output, target)

        # Update and log metrics
        preds = torch.softmax(output, dim=1)
        self.acc["val_"].update(preds, target)
        self.auc_roc["val_"].update(preds, target)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss, 'preds': output, 'target': target}
