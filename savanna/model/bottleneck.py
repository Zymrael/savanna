# class Hyena(nn.Module):
#     def __init__(
#         self,
#         global_config,
#         attention_mask_func,
#         init_method,
#         output_layer_init_method,
#         layer_number,
#         rpe=None,
#         rotary=False,
#         use_cache=False,
#     ):

#         super().__init__()
#         self.layer_number = layer_number

#         norm, eps = get_norm(global_config)
#         self.prenorm, self.postnorm = global_config.prenorm, global_config.postnorm
#         if global_config.prenorm:
#             self.input_layernorm = norm(global_config.hidden_size, eps=eps)
#         if global_config.postnorm:
#             self.post_attention_layernorm = norm(global_config.hidden_size, eps=eps)

#         self.use_cache = use_cache

#         self.hidden_dropout = global_config.hidden_dropout
#         self.bias_dropout_fusion = global_config.bias_dropout_fusion

#         if self.gpt_j_residual:
#             self.reduce = mpu.mappings.reduce_from_model_parallel_region

#         # Self attention.
#         self.attention = ParallelSelfAttention(
#             global_config=global_config,
#             attention_mask_func=attention_mask_func,
#             init_method=init_method,
#             output_layer_init_method=output_layer_init_method,
#             layer_number=layer_number,
#             rpe=rpe,
#             use_cache=self.use_cache,
#             rotary=rotary,
#             parallel_output=self.gpt_j_residual,
#         )

#         # MLP
#         if global_config.all_config["identity_mlp"]:
#             self.mlp = nn.Identity()
#         else:
#             self.mlp = ParallelMLP(
#                 global_config=global_config,
#                 init_method=init_method,
#                 output_layer_init_method=output_layer_init_method,
#                 parallel_output=self.gpt_j_residual,
#             )

#         self.layer_past = None  # used to cache k/v pairs in inference

#     def _get_bias_dropout(self):
#         if self.bias_dropout_fusion:
#             fn = (
#                 bias_dropout_add_fused_train
#                 if self.training
#                 else bias_dropout_add_fused_inference
#             )
#         else:
#             fn = get_bias_dropout_add(self.training)
#         return fn

#     def forward(self, x, attention_mask, layer_past=None):
#         layer_past = layer_past if layer_past is not None else self.layer_past
#         bias_dropout_fn = self._get_bias_dropout()

#         residual = x

#         if self.prenorm:
#             x = self.input_layernorm(x)

#         attention_output, attention_bias = self.attention(
#             x, attention_mask, layer_past=layer_past
#         )

#         with torch.enable_grad():
#             attention_output = bias_dropout_fn(
#                 attention_output,
#                 bias=attention_bias.expand_as(residual),
#                 residual=residual,
#                 prob=self.hidden_dropout,
#             )


#         if isinstance(self.mlp, nn.Identity):
#             output = attention_output
#             if self.postnorm:
#                 output = self.post_attention_layernorm(output)

#         else:
#             # output = x + mlp(ln2(x))
#             mlp_output, mlp_bias = self.mlp(attention_output)
#             with torch.enable_grad():
#                 output = bias_dropout_fn(
#                     mlp_output,
#                     bias=mlp_bias.expand_as(mlp_output),
#                     residual=attention_output,
#                     prob=self.hidden_dropout,
#                 )

#         return output


# class HyenaBottleneck(Hyena):
#     def __init__


# class HyenaPipe(Hyena):
#     """Extends Hyena to forward attention_mask through the pipeline."""

#     def forward(self, args):
#         assert (
#             len(args) == 2
#         ), "TransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
#         hidden_states, attention_mask = args
#         # we are returning just [hidden_states, mask]
#         return super().forward(hidden_states, attention_mask), attention_mask
