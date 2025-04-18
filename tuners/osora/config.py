import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class OSoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`OSoraModel`].
    """
    r: int = field(default=256, metadata={"help": "The rank of the OSora layer."})

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with OSora."
                "Only linear layers are supported."
            )
        }
    )
    
    osora_dropout: float = field(default=0.0, metadata={"help": "The dropout rate of the OSora layer."})

    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"}
    )

    bias: str = field(default="none", metadata={"help": "Bias type for OSora. Can be 'none', 'all' or 'osora_only'"})

    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from OSora layers to be set as trainable and saved in the final checkpoint."
                "For example, in Sequence Classification or Token Classification tasks, the final layer"
                "`classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        }
    )
    
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, if this argument is specified, it will apply the OSora transformations on"
                "the layer indexes that are specified inside this list. If a single integer is passed, PEFT will transform"
                "only the layer at this index."
            )
        }
    )
    
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer"
                "pattern is not in the common layers pattern."
            )
        }
    )

    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using LoRA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate LoRA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
                "The format is a list of [start, end) pairs which specify the layer ranges to stack. For example:\n"
                "   Original model has 5 layers labelled by their position in the model: `[0, 1, 2, 3, 4]`\n"
                "   layer_replication: `[[0, 4], [2, 5]]`\n"
                "   Final model will have this arrangement of original layers: `[0, 1, 2, 3, 2, 3, 4]`\n"
                "This format is based on what is used for pass-through merges in mergekit. It makes it simple to select sequential "
                "ranges of a model and stack them while reusing layers at either end of each sequence."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.OSORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )