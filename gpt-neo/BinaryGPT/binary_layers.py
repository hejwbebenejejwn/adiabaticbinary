import math

import torch
import torch.nn as nn
import torch.nn.init as init
from BinaryGPT.binary_functions import BinaryLinearFunction
from BinaryGPT.state_storage import Config
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from transformers import GPTNeoConfig
from transformers.activations import ACT2FN
from transformers.utils import logging

logger = logging.get_logger(__name__)


# %% binary linear layer
class BinaryLinear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Definition of binarize training parameters: kk and aa
        self.kk = Parameter(torch.tensor(1.), requires_grad=False)
        self.aa = Parameter(torch.tensor(1.), requires_grad=False)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return BinaryLinearFunction.apply(input, self.weight, self.bias, self.kk, self.aa)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    # External modification of kk & aa stage 1
    def set_kk_stage1(self, kknew: list):
        with torch.no_grad():
            kknew = torch.tensor(kknew, dtype=self.weight.dtype, device=self.weight.device)
            self.kk = Parameter(kknew[0], requires_grad=False)
            self.aa = Parameter(kknew[1], requires_grad=False)

    # External modification of kk & aa stage 2
    def set_kk_stage2(self, ratio: float):
        with torch.no_grad():
            q = 1 - ratio
            ratio = torch.tensor(ratio, dtype=self.weight.dtype, device=self.weight.device)
            weight_flatten = self.weight.flatten().abs().float()
            weight_indice = weight_flatten < 2 * torch.pow(self.kk.to(device=ratio.device, dtype=ratio.dtype), -1)
            total_elements = weight_indice.sum().item()
            weight_indice = torch.nonzero(weight_indice, as_tuple=False).view(-1)
            # Sample approximately 10% of the indices that meet the condition
            sample_size = int(total_elements * 0.025)
            selected_indices = weight_indice[torch.randint(0, total_elements, (sample_size,))]
            # Using sampled indices to calculate quantile
            quantile_value = torch.quantile(weight_flatten[selected_indices], q).to(ratio.device, ratio.dtype)
            # Update kk
            self.kk = Parameter(2 * torch.pow(quantile_value, -1).to(ratio.device, ratio.dtype), requires_grad=False)

            if self.kk < 1.:
                self.aa = Parameter(torch.pow(self.kk, -1), requires_grad=False)
            else:
                self.aa = Parameter(torch.tensor([1.]).to(ratio.device, ratio.dtype), requires_grad=False)

            Config.kk_list.append(self.kk)


# %% gpt neo block
class BinaryGPTNeoBlock(Module):
    def __init__(self, config: GPTNeoConfig, layer_id):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = BinaryGPTNeoAttention(config, layer_id)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = BinaryGPTNeoMLP(inner_dim, config)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

    # External modification of kk & aa stage 1
    def set_kk_stage1(self, kknew):
        self.attn.set_kk_stage1(kknew)
        self.mlp.set_kk_stage1(kknew)

    # External modification of kk & aa stage 2
    def set_kk_stage2(self, ratio):
        self.attn.set_kk_stage2(ratio)
        self.mlp.set_kk_stage2(ratio)


# %% gpt neo attention
class BinaryGPTNeoSelfAttention(Module):
    def __init__(self, config: GPTNeoConfig, attention_type):
        super().__init__()

        max_positions = config.max_position_embeddings
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(
            1, 1, max_positions, max_positions
        )

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))

        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.k_proj = BinaryLinear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = BinaryLinear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = BinaryLinear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = BinaryLinear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_past=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

    # External modification of kk & aa stage 1
    def set_kk_stage1(self, kknew):
        self.k_proj.set_kk_stage1(kknew)
        self.v_proj.set_kk_stage1(kknew)
        self.q_proj.set_kk_stage1(kknew)
        self.out_proj.set_kk_stage1(kknew)

    # External modification of kk & aa stage 2
    def set_kk_stage2(self, ratio):
        self.k_proj.set_kk_stage2(ratio)
        self.v_proj.set_kk_stage2(ratio)
        self.q_proj.set_kk_stage2(ratio)
        self.out_proj.set_kk_stage2(ratio)


class BinaryGPTNeoAttention(Module):
    def __init__(self, config: GPTNeoConfig, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type in ["global", "local"]:
            self.attention = BinaryGPTNeoSelfAttention(config, self.attention_type)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `gpt_config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

    # External modification of kk & aa stage 1
    def set_kk_stage1(self, kknew):
        self.attention.set_kk_stage1(kknew)

    # External modification of kk & aa stage 2
    def set_kk_stage2(self, ratio):
        self.attention.set_kk_stage2(ratio)


# %% gpt neo mlp
class BinaryGPTNeoMLP(Module):
    def __init__(self, intermediate_size, config: GPTNeoConfig):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = BinaryLinear(embed_dim, intermediate_size)
        self.c_proj = BinaryLinear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    # External modification of kk & aa stage 1
    def set_kk_stage1(self, kknew):
        self.c_fc.set_kk_stage1(kknew)
        self.c_proj.set_kk_stage1(kknew)

    # External modification of kk & aa stage 2
    def set_kk_stage2(self, ratio):
        self.c_fc.set_kk_stage2(ratio)
        self.c_proj.set_kk_stage2(ratio)
