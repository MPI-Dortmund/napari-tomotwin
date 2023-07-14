import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from napari_tomotwin.make_targets import _run


class MyTestCase(unittest.TestCase):
    def test_make_targets_single_cluster(self):
        fake_embedding = {
            "X": [0, 1, 2],
            "Y": [0, 1, 2],
            "Z": [0, 1, 2],
            "1": [5, 6, 7],
            "2": [5, 6, 7],
            "filepath": ["a.mrc","b.mrc","c.mrc"]
        }
        cluster = np.array([1,1,1])
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


if __name__ == '__main__':
    unittest.main()
