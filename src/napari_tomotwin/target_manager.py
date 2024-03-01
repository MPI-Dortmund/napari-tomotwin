from dataclasses import dataclass, field
from itertools import count
import numpy as np
@dataclass
class Target:

    embeddings_path: str
    embeddings_mask: np.array
    target_color: list[int] = field(compare=False)
    target_id: int = field(default_factory=count().__next__, compare=False)
    target_name: str = field(compare=False, default=f"None")


    def __eq__(self, other):
        return np.array_equal(self.embeddings_mask, other.embeddings_mask) and self.embeddings_path == other.embeddings_path



class TargetManager:

    def __init__(self):
        self.targets : list[Target] = []


    def add_target(self, target: Target) -> bool:

        for t in self.targets:
            print(type(t), type(target))
            if target.__eq__(t):
                return False
        self.targets.append(target)
        return True


    def remove_target(self, target_ids: list[int]):
        self.targets = [t for t in self.targets if t.target_id not in target_ids]
