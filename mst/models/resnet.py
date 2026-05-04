import torch 
import torch.nn as nn 
import torch.nn.functional as F
import monai.networks.nets as nets
from .base_model import BasicClassifier
import torchvision.models as models
from einops import rearrange
from .utils.transformer_blocks import TransformerEncoderLayer

def _get_resnet_monai(model):
    return {
        18: nets.resnet18, 34: nets.resnet34, 50: nets.resnet50, 101: nets.resnet101, 152: nets.resnet152
    }.get(model)
    
def _get_resnet_torch(model):
    return {
        18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152
    }.get(model)

class GetLast(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input[-1]


class ResNet(BasicClassifier):
    def __init__(self, in_ch, out_ch, spatial_dims=3, model=34, pretrained=False, kwargs_resnet={}, **kwargs):
        emb_ch = kwargs.pop('emb_ch', out_ch)
#        kwargs.pop('model', None)
#        kwargs.pop('save_attn', None) # Remove save_attn if present
#        kwargs.pop('model_size', None) # Remove save_attn if present
#        kwargs.pop('use_registers', None) # Remove use_registers if present
#        kwargs.pop('use_bottleneck', None) # Remove use_registers if present
#        kwargs.pop('use_slice_pos_emb', None)
#        kwargs.pop('enable_linear', None) # Remove enable_linear if present
#        kwargs.pop('enable_trans', None)
#        kwargs.pop('slice_fusion', None)
#        kwargs.pop('freeze', None) # Remove freeze if present
        super().__init__(in_ch, out_ch, spatial_dims, **kwargs)
        
        self.attention_maps = []

        if pretrained:
            if spatial_dims==3:
                resnet = nets.ResNetFeatures(model_name=f'resnet{model}',  spatial_dims=spatial_dims, in_channels=in_ch)
                resnet_out_ch = max([ mod.num_features for name, mod in resnet.layer4[-1]._modules.items() if "bn" in name])
                self.model = nn.Sequential(
                    resnet,
                    GetLast(),
                    nn.AdaptiveAvgPool3d(1),
                    nn.Flatten(1),
                    nn.Linear(resnet_out_ch, emb_ch)
                )
            elif spatial_dims==2:
                Model = _get_resnet_torch(model)
                self.model = Model(weights="DEFAULT")
                resnet_out_ch = self.model.fc.in_features
                if emb_ch is None:
                    self.model.fc = nn.Identity()
                elif emb_ch != 1000:
                    self.model.fc = nn.Linear(resnet_out_ch, emb_ch)
        else:
            Model = _get_resnet_monai(model)
            self.model = Model(n_input_channels=in_ch, spatial_dims=spatial_dims, num_classes=emb_ch, **kwargs_resnet)
        
   
    def forward(self, source, save_attn=False, **kwargs):
        if save_attn:
            self.attention_maps = []
            self.activations = []     # reset activations 
            self.gradients = []       # reset gradients
            self.hooks = []
            self.register_hooks()

        output = self.model(source.to(self.device))

        if save_attn:
            self.model.zero_grad()
            loss = output.gather(1, output.argmax(dim=1, keepdim=True)).sum()
            loss.backward(retain_graph=True)
            self.compute_attention_maps()
            for handle in self.hooks:
                handle.remove()
        
        return output

    def get_attention_maps(self):
        return self.attention_maps[-1] # [B, C=1, D, H, W]
    
    def register_hooks(self):
        def save_gradient(module, input, output):
            def _store_grad(grad):
                self.gradients = [grad] + self.gradients
            self.hooks.append(output.register_hook(_store_grad))
     
        def save_activation(module, input, output):
            self.activations.append(output)

        for name, mod in self.model.named_modules():
            if isinstance(mod,  nn.ReLU):
                self.hooks.append(mod.register_forward_hook(save_activation))
                self.hooks.append(mod.register_forward_hook(save_gradient)) # Because of https://github.com/pytorch/pytorch/issues/61519
    
    def compute_attention_maps(self):
        # Compute Grad-CAM
        for grad, act in zip(self.gradients, self.activations):
            weights = self.compute_grad_cam_weights(grad, act)
            gradcam = torch.sum(weights * act, dim=1, keepdim=True)
            gradcam = F.relu(gradcam) # = maximum(cam, 0)

            # Normalize the attention map
            gradcam = gradcam - gradcam.min()
            gradcam = gradcam / gradcam.max()
            self.attention_maps.append(gradcam)
        
    def compute_grad_cam_weights(self, grads, act, cam_mode='gradcam++'):
        spatial_dims = list(range(2, grads.ndim))
        if cam_mode == 'gradcam':
            return torch.mean(grads, dim=spatial_dims, keepdim=True) 
        elif cam_mode == 'gradcam++':
            grads_power_2 = grads**2
            grads_power_3 = grads_power_2 * grads
            # Equation 19 in https://arxiv.org/abs/1710.11063
            sum_activations = torch.sum(act, axis=spatial_dims, keepdim=True)
            denom =  (2 * grads_power_2 + sum_activations * grads_power_3 + 1e-6)
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            aij = grads_power_2/denom

            weights = F.relu(grads) * aij
            weights = torch.sum(weights, axis=spatial_dims, keepdim=True)
            return weights
        else:
            raise ValueError('Unknown CAM mode')



    
class ResNetSliceTrans(ResNet):
    def __init__(
            self, 
            in_ch,
              out_ch, 
              spatial_dims=2, 
              model=34, 
              pretrained=True, 
              kwargs_resnet={}, 
              rotary_positional_encoding=None,
              optimizer_kwargs={'lr':1e-5, 'weight_decay':1e-2}, 
              **kwargs
            ):
        super().__init__(
            in_ch, 
            out_ch, 
            spatial_dims, 
            emb_ch=None,
            model=model, 
            pretrained=pretrained, 
            kwargs_resnet=kwargs_resnet, 
            optimizer_kwargs=optimizer_kwargs, 
            **kwargs
        )
        
        emb_ch = 512 if model <= 34 else 2048
        self.attention_maps_slice = []

        self.slice_fusion = nn.TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=emb_ch,
                nhead=16,
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
        self.linear = nn.Linear(emb_ch, out_ch)


    def forward(self, source, src_key_padding_mask=None, **kwargs):
        x = source.to(self.device) # [B, C, D, H, W]
        if kwargs.get('save_attn'):
            self.attention_maps_slice = []
            self.slice_hooks = []
            self.register_hooks_trans()

        B, C, *_ = x.shape
        x = x.repeat(1, 3, 1, 1, 1) # Gray to RGB 
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        x = super().forward(x, **kwargs) # [(B D), C, H, W] -> [(B D), out] 
        x = rearrange(x, '(b d) e -> b d e', b=B)
        x = torch.concat([self.cls_token.repeat(B, 1, 1), x], dim=1)
        
        if src_key_padding_mask is not None: 
            src_key_padding_mask = src_key_padding_mask.to(self.device)
            src_key_padding_mask_cls = torch.zeros((B, 1), device=self.device, dtype=bool)
            src_key_padding_mask = torch.concat([src_key_padding_mask_cls, src_key_padding_mask], dim=1)# [Batch, L]
        
        x = self.slice_fusion(x, src_key_padding_mask=src_key_padding_mask)
        x = x[:, 0]
        x = self.linear(x)
        
        if kwargs.get('save_attn'):
            self.deregister_hooks()
        
        return x
    
    def get_slice_attention(self):
        attention_map_slice = self.attention_maps_slice[-1] # [B, Heads, 1+D, 1+D]
        attention_map_slice = attention_map_slice[:, :, 0, 1:] # [B, Heads, D]
        attention_map_slice /= attention_map_slice.sum(dim=-1, keepdim=True)

        attention_map_slice = attention_map_slice.mean(dim=1)  # [B, D]
        attention_map_slice = attention_map_slice.view(-1) # [B*D]
        attention_map_slice = attention_map_slice[:, None, None] # [B*D, 1, 1]
        
        return attention_map_slice

    def get_attention_maps(self):
        attention_map_resnet = super().get_attention_maps() # [B*D, 1, H, W]

        attention_map_slice = self.get_slice_attention() # [B*D, 1, 1]
        attention_map = attention_map_slice.unsqueeze(-1)*attention_map_resnet 
        return attention_map #  [B*D, 1, H, W]
    
    def register_hooks_trans(self):
        def enable_attention(module):
            forward_orig = module.forward
            def forward_wrap(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return forward_orig(*args, **kwargs)
            module.forward = forward_wrap
            module.foward_orig = forward_orig

        def get_attention_maps(module, input, output):
            self.attention_maps_slice.append(output[1])

        for name, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                enable_attention(mod)
                self.slice_hooks.append(mod.register_forward_hook(get_attention_maps))


    def deregister_hooks(self):
        for handle in self.slice_hooks:
            handle.remove()

        # Restore forward function
        for _, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                mod.forward = mod.foward_orig