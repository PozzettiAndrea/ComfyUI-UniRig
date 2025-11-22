from typing import Dict, List, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass
import yaml
from box import Box

from .spec import ConfigSpec

@dataclass
class OrderConfig(ConfigSpec):
    '''
    Config to handle bones re-ordering.
    '''
    
    # {skeleton_name: path}
    skeleton_path: Dict[str, str]
    
    # {cls: {part_name: [bone_name_1, bone_name_2, ...]}}
    parts: Dict[str, Dict[str, List[str]]]
    
    # {cls: parts of bones to be arranged in [part_name_1, part_name_2, ...]}
    parts_order: Dict[str, List[str]]
    
    @classmethod
    def parse(cls, config):
        cls.check_keys(config)
        skeleton_path = config.skeleton_path
        parts = {}
        parts_order = {}
        for (cls, path) in skeleton_path.items():
            assert cls not in parts, 'cls conflicts'
            d = Box(yaml.safe_load(open(path, 'r')))
            parts[cls] = d.parts
            parts_order[cls] = d.parts_order
        return OrderConfig(
            skeleton_path=skeleton_path,
            parts=parts,
            parts_order=parts_order,
        )

class Order():
    
    # {part_name: [bone_name_1, bone_name_2, ...]}
    parts: Dict[str, Dict[str, List[str]]]
    
    # parts of bones to be arranged in [part_name_1, part_name_2, ...]
    parts_order: Dict[str, List[str]]
    
    def __init__(self, config: OrderConfig):
        self.parts          = config.parts
        self.parts_order    = config.parts_order
    
    def part_exists(self, cls: str, part: str, names: List[str]) -> bool:
        '''
        Check if part exists.
        '''
        if part not in self.parts[cls]:
            return False
        for name in self.parts[cls][part]:
            if name not in names:
                return False
        return True
    
    def make_names(self, cls: Union[str, None], parts: List[Union[str, None]], num_bones: int) -> List[str]:
        '''
        Get names for specified cls.
        '''
        names = []

        # Auto-infer parts from cls when parts list is empty
        # This handles cases where the model doesn't generate part tokens
        if len(parts) == 0 and cls is not None and cls in self.parts_order:
            print(f"[Order] Auto-inferring parts for cls='{cls}' (parts list was empty)")
            parts = list(self.parts_order[cls])
            print(f"[Order] Inferred parts: {parts}")

        for part in parts:
            if part is None: # spring
                continue
            if cls in self.parts and part in self.parts[cls]:
                names.extend(self.parts[cls][part])

        assert len(names) <= num_bones, f"Expected {len(names)} bones for cls='{cls}', got {num_bones} bones"
        for i in range(len(names), num_bones):
            names.append(f"bone_{i}")
        return names
    
    def arrange_names(self, cls: str, names: List[str], parents: List[Union[int, None]]) -> Tuple[List[str], Dict[int, Union[str]]]:
        '''
        Arrange names according to required parts order.
        '''
        if cls not in self.parts_order:
            return names, {0: None} # add a spring token
        vis = defaultdict(bool)
        name_to_id = {name: i for (i, name) in enumerate(names)}
        new_names = []
        parts_bias = {}
        for part in self.parts_order[cls]:
            if self.part_exists(cls=cls, part=part, names=names):
                for name in self.parts[cls][part]:
                    vis[name] = True
                flag = False
                for name in self.parts[cls][part]:
                    pid = parents[name_to_id[name]]
                    if pid is None:
                        continue
                    if not vis[names[pid]]:
                        flag = True
                        break
                if flag: # incorrect parts order and should immediately add a spring token
                    break
                parts_bias[len(new_names)] = part
                new_names.extend(self.parts[cls][part])
        parts_bias[len(new_names)] = None # add a spring token
        for name in names:
            if name not in new_names:
                new_names.append(name)
        return new_names, parts_bias

def get_order(config: OrderConfig) -> Order:
    return Order(config=config)