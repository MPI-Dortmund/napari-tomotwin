import tempfile
import unittest

import mrcfile
import napari
import napari_tomotwin.load_umap as lumap
import numpy as np
import pandas as pd
from napari_tomotwin.load_umap import LoadUmapTool


class MyTestCase(unittest.TestCase):
    def test_something(self):
        viewer = napari.Viewer()
        tool = LoadUmapTool()

        rand_volint = np.arange(0,100*100).reshape(100,100).astype(np.float32)

        umap = {
            "X": np.random.randint(0,100,size=100*100),
            "Y": np.random.randint(0,100,size=100 * 100),
            "Z": np.random.randint(0,100,size=100 * 100),
            "umap_0": np.random.rand(100*100),
            "umap_1": np.random.rand(100 * 100)
        }



        print(rand_volint.shape)
        with tempfile.TemporaryDirectory() as tmpdirname:

            with mrcfile.new(f"{tmpdirname}/label_mask.mrci") as mrc:
                mrc.set_data(rand_volint)
            #viewer.open(plugin='napari-boxmanager',
            #                   path=[f"{tmpdirname}/label_mask.mrci"])

            umap_df = pd.DataFrame(umap)
            umap_df.attrs['tomogram_input_shape'] = (100, 100, 100)
            umap_df.attrs['embeddings_attrs'] = {
                "stride": (1,1,1),
                "tomogram_input_shape": (100,100,100)
            }
            umap_df.attrs['embeddings_path'] = ""
            umap_df.to_pickle(f"{tmpdirname}/umap.tumap")

            widget, _ = viewer.window.add_plugin_dock_widget('napari-tomotwin', widget_name='Cluster UMAP embeddings')
            lyr = tool.load_umap(filename=f"{tmpdirname}/umap.tumap")
            viewer.add_layer(lyr)

            assert True # just make sure that now exception is raised

if __name__ == '__main__':
    unittest.main()
