import json
from typing import Dict, Tuple
from transformers import TrainingArguments, HfArgumentParser
import argparse
from transformers.hf_argparser import DataClass

import dataclasses
from typing import Tuple, Union, Iterable, NewType, Any

DataClassType = NewType("DataClassType", Any)

def parse_args(file_path):
    # This is going to need a lot of work to fix tbh
    argp = CustomArgumentParser(TrainingArguments)
    training_args, remianing_args = argp.parse_json_file(file_path)
    
    args = argparse.Namespace(**remianing_args)

    return training_args, args


def training_metrics():
    return


class CustomArgumentParser(HfArgumentParser):
    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        super().__init__(dataclass_types, **kwargs)
    
    def parse_dict(self, args: Dict[str, Any], allow_extra_keys: bool = False) -> Tuple[DataClass]:
        unused_keys = set(args.keys())
        outputs = []
        
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            #print(keys)
            inputs = {k: v for k,v in args.items() if k in keys}
            unused_keys.difference_update(inputs.keys())
            obj = dtype(**inputs)
            outputs.append(obj)
        
        remaining_dict = {k: v for k, v in args.items() if k in unused_keys}

        return (*outputs, remaining_dict)
