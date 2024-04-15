import os
from dataclasses import dataclass, field
from itertools import count

import numpy as np
import pandas as pd
from napari.layers import Labels

from .make_targets import _get_medoid_embedding


@dataclass
class Target:

    embeddings_path: str
    embeddings_mask: np.array
    layer: Labels = field(compare=False)
    cluster_id: int = field(compare=False)
    target_color: list[int] = field(compare=False)
    target_id: int = field(default_factory=count().__next__, compare=False)
    target_name: str = field(compare=False, default=f"None")

    def __eq__(self, other):
        return (
            np.array_equal(self.embeddings_mask, other.embeddings_mask)
            and self.embeddings_path == other.embeddings_path
        )


class TargetManager:

    def __init__(self):
        self.targets: list[Target] = []

    def add_target(self, target: Target) -> bool:

        for t in self.targets:
            print(type(t), type(target))
            if target.__eq__(t):
                return False
        self.targets.append(target)
        return True

    def remove_target(self, target_ids: list[int]):
        self.targets = [
            t for t in self.targets if t.target_id not in target_ids
        ]

    def get_target_by_id(self, id: int):
        for t in self.targets:
            if t.target_id == id:
                return t
        return None

    def save_to_disk(self, output_folder):

        alL_medoids = []
        all_sub_embeddings = []
        all_positions = []
        all_target_names = []
        for target in self.targets:
            embeddings = pd.read_pickle(target.embeddings_path)
            cluster_embeddings = embeddings.loc[target.embeddings_mask, :]
            medoid, position = _get_medoid_embedding(cluster_embeddings)

            all_sub_embeddings.append(cluster_embeddings)
            medoid = medoid.to_frame().T
            alL_medoids.append(medoid)

            all_positions.append(position)
            if target.target_name == "None":
                all_target_names.append(f"cluser_{target.target_id}")
            else:
                all_target_names.append(target.target_name)

        df_targets = pd.concat(alL_medoids, ignore_index=True)
        df_targets["filepath"] = all_target_names
        print(df_targets)

        print("Write targets")
        os.makedirs(output_folder, exist_ok="True")
        pth_ref = os.path.join(output_folder, "cluster_targets.temb")
        df_targets.to_pickle(pth_ref)
        for pos_i, df_loc in enumerate(all_positions):
            clname = (
                df_targets[["filepath"]]
                .iloc[pos_i]
                .to_string(header=False, index=False)
            )
            if df_loc is not None and len(df_loc) > 0:
                pth_loc = os.path.join(
                    output_folder, f"medoid_{clname}.coords"
                )
                df_loc[["X", "Y", "Z"]].to_csv(
                    pth_loc, sep=" ", header=None, index=None
                )

        print("Write custer embeddings")
        for emb_i, emb in enumerate(all_sub_embeddings):
            clname = (
                df_targets[["filepath"]]
                .iloc[emb_i]
                .to_string(header=False, index=False)
            )
            pth_emb = os.path.join(output_folder, f"embeddings_{clname}.temb")
            emb.to_pickle(pth_emb)
