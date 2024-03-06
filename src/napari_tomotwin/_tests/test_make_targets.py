import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from napari_tomotwin.make_targets import _make_targets, _get_medoid_embedding

from glob import glob

class MyTestCase(unittest.TestCase):
    def test_make_targets_single_cluster_medoid(self):
        fake_embedding = {
            "X": [0, 1, 2],
            "Y": [0, 1, 2],
            "Z": [0, 1, 2],
            "1": [5, 6, 7],
            "2": [5, 6, 7],
            "filepath": ["a.mrc","b.mrc","c.mrc"]
        }
        cluster = pd.Series(np.array([1,1,1]))
        with tempfile.TemporaryDirectory() as tmpdirname:
            _, _, target_locations = _make_targets(embeddings=pd.DataFrame(fake_embedding),
                          clusters=cluster,
                          avg_func=_get_medoid_embedding,
                          target_cluster= None)


            self.assertEqual(target_locations[1].iloc[0, 0], 1)
            self.assertEqual(target_locations[1].iloc[0, 1], 1)
            self.assertEqual(target_locations[1].iloc[0, 2], 1)

    def test_make_targets_two_clusters_medoid(self):
        range(6)
        fake_embedding = {
            "X": [0, 1, 2, 8, 9, 10],
            "Y": [0, 1, 2, 8, 9, 10],
            "Z": [0, 1, 2, 8, 9, 10],
            "1": [-1, 0, 1, -1, 0, 1],
            "2": [1, 1, 1, 2, 2, 2],
        }
        fake_embedding['filepath'] = [f"{i}.mrc" for i in range(len(fake_embedding["X"]))]
        cluster = pd.Series(np.array([1,1,1,2,2,2]))
        with tempfile.TemporaryDirectory() as tmpdirname:
            _, _, target_locations = _make_targets(embeddings=pd.DataFrame(fake_embedding),
                                                   clusters=cluster,
                                                   avg_func=_get_medoid_embedding,
                                                   target_cluster=None)

            box_data: pd.DataFrame = target_locations[1]
            self.assertEqual(box_data.iloc[0, 0], 1)
            self.assertEqual(box_data.iloc[0, 1], 1)
            self.assertEqual(box_data.iloc[0, 2], 1)

            box_data: pd.DataFrame = target_locations[2]
            self.assertEqual(box_data.iloc[0, 0], 9)
            self.assertEqual(box_data.iloc[0, 1], 9)
            self.assertEqual(box_data.iloc[0, 2], 9)

    def test_make_targets_single_cluster_average(self):
        fake_embedding = {
            "X": [0, 1, 2],
            "Y": [0, 1, 2],
            "Z": [0, 1, 2],
            "1": [5, 6, 7],
            "2": [5, 6, 7],
            "filepath": ["a.mrc","b.mrc","c.mrc"]
        }
        cluster = pd.Series(np.array([1,1,1]))
        targets_emb, _, _ = _make_targets(embeddings=pd.DataFrame(fake_embedding),
                                               clusters=cluster,
                                               avg_func=_get_medoid_embedding,
                                               target_cluster=None)

        self.assertEqual(targets_emb["1"].iloc[0],6)
        self.assertEqual(targets_emb["2"].iloc[0],6)






if __name__ == '__main__':
    unittest.main()
