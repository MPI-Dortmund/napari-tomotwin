import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from napari_tomotwin.make_targets import _run

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
            _run(clusters=cluster,
                 embeddings=pd.DataFrame(fake_embedding),
                 average_method_name="Medoid",
                 output_folder=tmpdirname)

            box_data: pd.DataFrame = pd.read_csv(
                os.path.join(tmpdirname,"cluster_1_medoid.coords"),
                delim_whitespace=True,
                index_col=False,
                header=None,
                dtype=float,
                names=["X","Y","Z"]
            ).astype(np.int32)  # type: ignore
            self.assertEqual(box_data.iloc[0, 0], 1)
            self.assertEqual(box_data.iloc[0, 1], 1)
            self.assertEqual(box_data.iloc[0, 2], 1)

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
            _run(clusters=cluster,
                 embeddings=pd.DataFrame(fake_embedding),
                 average_method_name="Medoid",
                 output_folder=tmpdirname)

            box_data: pd.DataFrame = pd.read_csv(
                os.path.join(tmpdirname,"cluster_1_medoid.coords"),
                delim_whitespace=True,
                index_col=False,
                header=None,
                dtype=float,
                names=["X","Y","Z"]
            ).astype(np.int32)  # type: ignore
            self.assertEqual(box_data.iloc[0, 0], 1)
            self.assertEqual(box_data.iloc[0, 1], 1)
            self.assertEqual(box_data.iloc[0, 2], 1)

            box_data: pd.DataFrame = pd.read_csv(
                os.path.join(tmpdirname, "cluster_2_medoid.coords"),
                delim_whitespace=True,
                index_col=False,
                header=None,
                dtype=float,
                names=["X", "Y", "Z"]
            ).astype(np.int32)  # type: ignore
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
        with tempfile.TemporaryDirectory() as tmpdirname:
            _run(clusters=cluster,
                 embeddings=pd.DataFrame(fake_embedding),
                 average_method_name="Average",
                 output_folder=tmpdirname)

            targets_emb: pd.DataFrame = pd.read_pickle(
                os.path.join(tmpdirname,"cluster_targets.temb"),
            )
            self.assertEqual(targets_emb["1"].iloc[0],6)
            self.assertEqual(targets_emb["2"].iloc[0],6)

    def test_make_targets_single_cluster_no_coords_written(self):
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
            _run(clusters=cluster,
                 embeddings=pd.DataFrame(fake_embedding),
                 average_method_name="Average",
                 output_folder=tmpdirname)

            r = glob(os.path.join(tmpdirname,"*.coords"))
            print(r)
            self.assertEqual(len(r),0)






if __name__ == '__main__':
    unittest.main()
