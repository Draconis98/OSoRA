import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from loguru import logger

class OSoraLayer(BaseTunerLayer):
    adapter_layer_names = ("osora_S", "osora_O") # tranable
    other_param_names = ("osora_U", "osora_V") # fixed

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.osora_dropout = nn.ModuleDict({})

        self.osora_S = nn.ParameterDict({})
        self.osora_O = nn.ParameterDict({})
        
        self.osora_U = nn.ModuleDict({})
        self.osora_V = nn.ModuleDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)
    
    def update_layer(
            self, 
            adapter_name, 
            r,
            osora_dropout
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r

        if osora_dropout > 0.0:
            osora_dropout_layer = nn.Dropout(p=osora_dropout)
        else:
            osora_dropout_layer = nn.Identity()

        self.osora_dropout.update(nn.ModuleDict({adapter_name: osora_dropout_layer}))

        self.osora_O[adapter_name] = nn.Parameter(torch.ones(self.out_features))
        self.osora_U[adapter_name] = nn.Linear(self.out_features, r, bias=False)
        self.osora_S[adapter_name] = nn.Parameter(torch.ones(r))
        self.osora_V[adapter_name] = nn.Linear(r, self.in_features, bias=False)
        
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize OSoRA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)

        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        Ur = U[:, : self.r[adapter_name]]
        Sr = S[: self.r[adapter_name]]
        Vhr = Vh[: self.r[adapter_name], :]

        self.osora_U[adapter_name].weight.data = Ur.contiguous()
        self.osora_S[adapter_name].data = Sr
        self.osora_V[adapter_name].weight.data = Vhr.contiguous()

        weight = weight.data - Ur @ torch.diag(Sr) @ Vhr
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)


class Linear(nn.Linear, OSoraLayer):
    # OSora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        osora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        OSoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, osora_dropout)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.osora_S.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)
                
    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.osora_S.keys():
                base_layer = self.get_base_layer()
                base_layer.weight.data -= self.get_delta_weight(active_adapter)
    
    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        osora_O = self.osora_O[adapter]
        osora_U = self.osora_U[adapter]
        osora_S = self.osora_S[adapter]
        osora_V = self.osora_V[adapter]

        device = osora_V.device
        dtype = osora_V.dtype

        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        if cast_to_fp32:
            osora_U = osora_U.float()
            osora_S = osora_S.float()
            osora_V = osora_V.float()
            osora_O = osora_O.float()

        delta_weight = transpose(torch.diag(osora_O) @ osora_U @ torch.diag(osora_S) @ osora_V, self.fan_in_fan_out)

        if cast_to_fp32:
            delta_weight = delta_weight.to(dtype)

            # cast back the weights
            self.osora_U[adapter].data = osora_U.to(dtype)
            self.osora_S[adapter].data = osora_S.to(dtype)
            self.osora_V[adapter].weight.data = osora_V.to(dtype)
            self.osora_O[adapter].data = osora_O.to(dtype)
            
        return delta_weight
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.osora_S.keys():
                    continue

                osora_U = self.osora_U[active_adapter].weight
                osora_S = self.osora_S[active_adapter]
                osora_V = self.osora_V[active_adapter].weight
                osora_O = self.osora_O[active_adapter]

                dropout = self.osora_dropout[active_adapter]
                x = x.to(osora_S.dtype)
                # result = result + F.linear(dropout(x),torch.diag(osora_O) @ osora_U @ torch.diag(osora_S) @ osora_V
                result = result + osora_O * F.linear(osora_S * F.linear(dropout(x), osora_V), osora_U)

        result = result.to(previous_dtype)
        return result
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "osora." + rep
