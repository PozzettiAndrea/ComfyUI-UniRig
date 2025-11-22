import numpy as np
from numpy import ndarray

from typing import Dict, Tuple, Union, List

from .spec import TokenizerSpec, TokenizeInput, DetokenizeOutput, TokenizerConfig
from .spec import make_skeleton
from ..data.order import get_order

class TokenizerPart(TokenizerSpec):
    def __init__(
        self,
        config: TokenizerConfig,
    ):
        super().__init__()
        
        self._num_discrete      = config.num_discrete
        self._continuous_range  = config.continuous_range
        self.cls_token_id       = config.cls_token_id.copy()
        self.parts_token_id     = config.parts_token_id.copy()
        self.order              = get_order(config.order_config)
        _offset                 = config.num_discrete
        
        self.token_id_branch = _offset + 0
        self.token_id_bos    = _offset + 1
        self.token_id_eos    = _offset + 2
        self.token_id_pad    = _offset + 3
        _offset += 4
        
        self.token_id_spring   = _offset + 0
        _offset += 1
        
        assert None not in self.parts_token_id
        for i in self.parts_token_id:
            self.parts_token_id[i] += _offset
        _offset += len(self.parts_token_id)
        
        self.token_id_cls_none = _offset + 0
        _offset += 1
        
        for i in self.cls_token_id:
            self.cls_token_id[i] += _offset
        _offset += len(self.cls_token_id)
        
        self._vocab_size = _offset
        
        self.parts_token_id_name = [x for x in self.parts_token_id]
        
        self.part_token_to_name = {v: k for k, v in self.parts_token_id.items()}
        assert len(self.part_token_to_name) == len(self.parts_token_id), 'names with same token found in parts_token_id'
        self.part_token_to_name[self.token_id_spring] = None
        
        self.cls_token_to_name = {v: k for k, v in self.cls_token_id.items()}
        assert len(self.cls_token_to_name) == len(self.cls_token_id), 'names with same token found in cls_token_id'

    def cls_name_to_token(self, cls: str) -> int:
        if cls not in self.cls_token_id:
            return self.token_id_cls_none
        return self.cls_token_id[cls]
    
    def part_name_to_token(self, part: str) -> int:
        assert part in self.parts_token_id, f"do not find part name    `{part}` in tokenizer"
        return self.parts_token_id[part]
    
    def next_posible_token(self, ids: ndarray) -> List[int]:
        if ids.shape[0] == 0 or ids.ndim == 0:
            return [self.token_id_bos]
        assert ids.ndim == 1, "expect an array"
        state = 'expect_bos'
        for id in ids:
            if state == 'expect_bos':
                assert id == self.token_id_bos, 'ids do not start with bos'
                state = 'expect_cls_or_part_or_joint'
            elif state == 'expect_cls_or_part_or_joint':
                if id < self.num_discrete:
                    state = 'expect_joint_2'
                elif id == self.token_id_cls_none or id in self.cls_token_id.values():
                    state = 'expect_part_or_joint'
                else: # a part
                    state = 'expect_joint'
            elif state == 'expect_part_or_joint':
                if id < self.num_discrete:
                    state = 'expect_joint_2'
                else:
                    state = 'expect_part_or_joint'
            elif state == 'expect_joint_2':
                state = 'expect_joint_3'
            elif state == 'expect_joint_3':
                state = 'expect_branch_or_part_or_joint'
            elif state == 'expect_branch_or_part_or_joint':
                if id == self.token_id_branch:
                    state = 'expect_joint'
                elif id < self.num_discrete:
                    state = 'expect_joint_2'
                else: # find a part
                    state = 'expect_joint'
            elif state == 'expect_joint':
                state = 'expect_joint_2'
            else:
                assert 0, state
        s = []
        def add_cls():
            s.append(self.token_id_cls_none)
            for v in self.cls_token_id.values():
                s.append(v)
        def add_part():
            s.append(self.token_id_spring)
            for v in self.parts_token_id.values():
                s.append(v)
        def add_joint():
            for i in range(self.num_discrete):
                s.append(i)
        def add_branch():
            s.append(self.token_id_branch)
        def add_eos():
            s.append(self.token_id_eos)
        def add_bos():
            s.append(self.token_id_bos)
        if state == 'expect_bos':
            add_bos()
        elif state == 'expect_cls_or_part_or_joint':
            add_cls()
            add_part()
            add_joint()
        elif state == 'expect_cls':
            add_cls()
        elif state == 'expect_part_or_joint':
            add_part()
            add_joint()
            add_eos()
        elif state == 'expect_joint_2':
            add_joint()
        elif state == 'expect_joint_3':
            add_joint()
        elif state == 'expect_branch_or_part_or_joint':
            add_joint()
            add_part()
            add_branch()
            add_eos()
        elif state == 'expect_joint':
            add_joint()
        else:
            assert 0, state
        return s
    
    def bones_in_sequence(self, ids: ndarray):
        assert ids.ndim == 1, "expect an array"
        s = 0
        state = 'expect_bos'
        for id in ids:
            if state == 'expect_bos':
                assert id == self.token_id_bos, 'ids do not start with bos'
                state = 'expect_cls_or_part_or_joint'
            elif state == 'expect_cls_or_part_or_joint':
                if id < self.num_discrete:
                    state = 'expect_joint_2'
                elif id == self.token_id_cls_none or id in self.cls_token_id.values():
                    state = 'expect_part_or_joint'
                else: # a part
                    state = 'expect_joint'
            elif state == 'expect_part_or_joint':
                if id < self.num_discrete:
                    state = 'expect_joint_2'
                else:
                    state = 'expect_part_or_joint'
            elif state == 'expect_joint_2':
                state = 'expect_joint_3'
            elif state == 'expect_joint_3':
                s += 1
                state = 'expect_branch_or_part_or_joint'
            elif state == 'expect_branch_or_part_or_joint':
                if id == self.token_id_branch:
                    state = 'expect_joint'
                elif id < self.num_discrete:
                    state = 'expect_joint_2'
                else: # find a part
                    state = 'expect_joint'
            elif state == 'expect_joint':
                state = 'expect_joint_2'
            else:
                assert 0, state
            if id == self.token_id_eos:
                break
        return s
    
    def tokenize(self, input: TokenizeInput) -> ndarray:
        num_bones = input.num_bones
        bones = discretize(t=input.bones, continuous_range=self.continuous_range, num_discrete=self.num_discrete)
        tails = discretize(t=input.tails, continuous_range=self.continuous_range, num_discrete=self.num_discrete)
        
        branch = input.branch
        is_leaf = input.is_leaf
        
        tokens = [self.token_id_bos]
        if input.cls is None or input.cls not in self.cls_token_id:
            tokens.append(self.token_id_cls_none)
        else:
            tokens.append(self.cls_token_id[input.cls])
        use_leaf = False
        for i in range(num_bones):
            # add parts token id
            if i in input.parts_bias:
                part = input.parts_bias[i]
                if part is None:
                    tokens.append(self.token_id_spring)
                else:
                    assert part in self.parts_token_id, f"do not find part name {part} in tokenizer {self.__class__}"
                    tokens.append(self.parts_token_id[part])
            if branch[i]:
                tokens.append(self.token_id_branch)
                tokens.append(bones[i, 0])
                tokens.append(bones[i, 1])
                tokens.append(bones[i, 2])
                tokens.append(bones[i, 3])
                tokens.append(bones[i, 4])
                tokens.append(bones[i, 5])
            else:
                tokens.append(bones[i, 3])
                tokens.append(bones[i, 4])
                tokens.append(bones[i, 5])
        tokens.append(self.token_id_eos)
        return np.array(tokens, dtype=np.int64)
            

    def detokenize(self, ids: ndarray, **kwargs) -> DetokenizeOutput:
        assert isinstance(ids, ndarray), 'expect ids to be ndarray'
        if ids[0] != self.token_id_bos:
            raise ValueError(f"first token is not bos")
        trailing_pad = 0
        while trailing_pad < ids.shape[0] and ids[-trailing_pad-1] == self.token_id_pad:
            trailing_pad += 1
        if ids[-1-trailing_pad] != self.token_id_eos:
            raise ValueError(f"last token is not eos")
        ids = ids[1:-1-trailing_pad]
        joints = []
        p_joints = []
        tails_dict = {}
        parts = []
        i = 0
        is_branch = False
        last_joint = None
        num_bones = 0
        cls = None
        while i < len(ids):
            if ids[i] < self.num_discrete:
                if is_branch:
                    p_joint = undiscretize(t=ids[i:i+3], continuous_range=self.continuous_range, num_discrete=self.num_discrete)
                    current_joint = undiscretize(t=ids[i+3:i+6], continuous_range=self.continuous_range, num_discrete=self.num_discrete)
                    joints.append(current_joint)
                    p_joints.append(p_joint)
                    i += 6
                else:
                    current_joint = undiscretize(t=ids[i:i+3], continuous_range=self.continuous_range, num_discrete=self.num_discrete)
                    joints.append(current_joint)
                    if len(p_joints) == 0: # root
                        p_joints.append(current_joint)
                        p_joint = current_joint
                    else:
                        assert last_joint is not None
                        p_joints.append(last_joint)
                        p_joint = last_joint
                    i += 3
                if last_joint is not None:
                    tails_dict[num_bones-1] = current_joint
                last_joint = current_joint
                num_bones += 1
                is_branch = False
            elif ids[i]==self.token_id_branch:
                is_branch = True
                last_joint = None
                i += 1
            elif ids[i]==self.token_id_spring or ids[i] in self.parts_token_id.values():
                parts.append(self.part_token_to_name[ids[i]])
                i += 1
            elif ids[i] in self.cls_token_id.values():
                cls = ids[i]
                i += 1
            elif ids[i] == self.token_id_cls_none:
                cls = None
                i += 1
            else:
                raise ValueError(f"unexpected token found: {ids[i]}")
        joints = np.stack(joints)
        p_joints = np.stack(p_joints)
        # leaf is ignored in this tokenizer so need to extrude tails for leaf and branch
        bones, tails, available_bones_id, parents = make_skeleton(
            joints=joints,
            p_joints=p_joints,
            tails_dict=tails_dict,
            convert_leaf_bones_to_tails=False,
            extrude_tail_for_leaf=True,
            extrude_tail_for_branch=True,
        )
        bones = bones[available_bones_id]
        tails = tails[available_bones_id]
        if cls in self.cls_token_to_name:
            cls = self.cls_token_to_name[cls]
        else:
            cls = None

        # Auto-infer parts if they're all None/empty
        if self.order is not None:
            # Filter out None values from parts list (spring tokens)
            cleaned_parts = [p for p in parts if p is not None]

            # If no real parts were generated, infer from cls and parts_order
            if len(cleaned_parts) == 0 and cls is not None and cls in self.order.parts_order:
                print(f"[Tokenizer] Model generated no part tokens for cls='{cls}', auto-inferring from parts_order")
                cleaned_parts = list(self.order.parts_order[cls])
                print(f"[Tokenizer] Inferred parts: {cleaned_parts}")

            names = self.order.make_names(cls=cls, parts=cleaned_parts, num_bones=num_bones)
        else:
            names = [f"bone_{i}" for i in range(num_bones)]
        return DetokenizeOutput(
            tokens=ids,
            parents=parents,
            bones=bones,
            tails=tails,
            no_skin=None,
            cls=cls,
            parts=parts,
            names=names,
            continuous_range=self.continuous_range,
        )
    
    def get_require_parts(self) -> List[str]:
        return self.parts_token_id_name
            
    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def pad(self):
        return self.token_id_pad
    
    @property
    def bos(self):
        return self.token_id_bos

    @property
    def eos(self):
        return self.token_id_eos
    
    @property
    def num_discrete(self):
        return self._num_discrete
    
    @property
    def continuous_range(self) -> Tuple[float, float]:
        return self._continuous_range

def discretize(
    t: ndarray,
    continuous_range: Tuple[float, float],
    num_discrete: int,
) -> ndarray:
    lo, hi = continuous_range
    assert hi >= lo
    t = (t - lo) / (hi - lo)
    t *= num_discrete
    return np.clip(t.round(), 0, num_discrete - 1).astype(np.int64)

def undiscretize(
    t: ndarray,
    continuous_range: Tuple[float, float],
    num_discrete: int,
) -> ndarray:
    lo, hi = continuous_range
    assert hi >= lo
    t = t.astype(np.float32) + 0.5
    t /= num_discrete
    return t * (hi - lo) + lo
