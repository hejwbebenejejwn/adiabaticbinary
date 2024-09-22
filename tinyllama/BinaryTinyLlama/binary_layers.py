import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from tinyllama.BinaryTinyLlama.binary_functions import BinaryLinearFunction
from tinyllama.BinaryTinyLlama.state_storage import Config

from transformers.activations import ACT2FN
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
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
    def set_kk_stage1(self, kknew):
        with torch.no_grad():
            self.kk = Parameter(kknew[0], requires_grad=False)
            self.aa = Parameter(kknew[1], requires_grad=False)

    # External modification of kk & aa stage 2
    def set_kk_stage2(self, ratio):
        with torch.no_grad():
            q = (1 - ratio).float()
            weight_flatten = self.weight.flatten().abs().float()
            weight_indice = weight_flatten < 2 * torch.pow(self.kk.to(device=ratio.device, dtype=ratio.dtype), -1)
            # Sample approximately 10% of the indices that meet the condition
            total_elements = weight_indice.sum().item()
            sample_size = int(total_elements * 0.025)
            selected_indices = torch.randint(0, total_elements, (sample_size,))
            # Using sampled indices to calculate quantile
            quantile_value = torch.quantile(weight_flatten[selected_indices], q).to(ratio.device, ratio.dtype)
            # Update kk
            self.kk = Parameter(2 * torch.pow(quantile_value, -1), requires_grad=False)

            if self.kk < 1:
                self.aa = Parameter(torch.pow(self.kk, -1).to(ratio.device, ratio.dtype), requires_grad=False)
            else:
                self.aa = Parameter(torch.tensor([1.]).to(ratio.device, ratio.dtype), requires_grad=False)

            Config.kk_list.append(self.kk)


# %% llama decoder
class BinaryLlamaDecoderLayer(Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BinaryLlamaAttention(config=config)
        self.mlp = BinaryLlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    # External modification of kk & aa stage 1
    def set_kk_stage1(self, kknew):
        self.self_attn.set_kk_stage1(kknew)
        self.mlp.set_kk_stage1(kknew)

    # External modification of kk & aa stage 2
    def set_kk_stage2(self, ratio):
        self.self_attn.set_kk_stage2(ratio)
        self.mlp.set_kk_stage2(ratio)


# %% llama mlp
class BinaryLlamaMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = BinaryLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = BinaryLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = BinaryLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # Definition of binarize training parameters: kk and aa
        self.kk = Parameter(torch.tensor(1.), requires_grad=False)
        self.aa = Parameter(torch.tensor(1.), requires_grad=False)

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([BinaryLinearFunction.apply(x, gate_proj_slices[i], None, self.gate_proj.kk, self.gate_proj.aa) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([BinaryLinearFunction.apply(x, up_proj_slices[i], None, self.up_proj.kk, self.up_proj.aa) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [BinaryLinearFunction.apply(intermediate_states[i], down_proj_slices[i], None, self.down_proj.kk, self.down_proj.aa) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

    # External modification of kk & aa stage 1
    def set_kk_stage1(self, kknew):
        self.gate_proj.set_kk_stage1(kknew)
        self.up_proj.set_kk_stage1(kknew)
        self.down_proj.set_kk_stage1(kknew)

    # External modification of kk & aa stage 2
    def set_kk_stage2(self, ratio):
        self.gate_proj.set_kk_stage2(ratio)
        self.up_proj.set_kk_stage2(ratio)
        self.down_proj.set_kk_stage2(ratio)


# %% attention layer
class BinaryLlamaAttention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = BinaryLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = BinaryLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = BinaryLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = BinaryLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [BinaryLinearFunction.apply(hidden_states, query_slices[i], None, self.q_proj.kk, self.q_proj.aa) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [BinaryLinearFunction.apply(hidden_states, key_slices[i], None, self.k_proj.kk, self.k_proj.aa) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [BinaryLinearFunction.apply(hidden_states, value_slices[i], None, self.v_proj.kk, self.v_proj.aa) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([BinaryLinearFunction.apply(attn_output[i], o_proj_slices[i], None, self.o_proj.kk, self.o_proj.aa) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    # External modification of kk & aa stage 1
    def set_kk_stage1(self, kknew):
        self.q_proj.set_kk_stage1(kknew)
        self.k_proj.set_kk_stage1(kknew)
        self.v_proj.set_kk_stage1(kknew)
        self.o_proj.set_kk_stage1(kknew)

    # External modification of kk & aa stage 2
    def set_kk_stage2(self, ratio):
        self.q_proj.set_kk_stage2(ratio)
        self.k_proj.set_kk_stage2(ratio)
        self.v_proj.set_kk_stage2(ratio)
        self.o_proj.set_kk_stage2(ratio)