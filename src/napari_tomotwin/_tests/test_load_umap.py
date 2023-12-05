import tempfile
import unittest

import mrcfile
import napari
import napari_tomotwin.load_umap as lumap
import numpy as np
import pandas as pd
from napari_tomotwin.load_umap import load_umap, _draw_circle


class MyTestCase(unittest.TestCase):
    def test_something(self):
        viewer = napari.Viewer()


        rand_volint = np.arange(0,100*100).reshape(100,100).astype(np.float32)

        umap = {
            "umap_0": np.random.rand(100*100),
            "umap_1": np.random.rand(100 * 100)
        }

        print(rand_volint.shape)
        with tempfile.TemporaryDirectory() as tmpdirname:

            with mrcfile.new(f"{tmpdirname}/label_mask.mrci") as mrc:
                mrc.set_data(rand_volint)
            viewer.open(plugin='napari-boxmanager',
                               path=[f"{tmpdirname}/label_mask.mrci"])

            umap_df = pd.DataFrame(umap)
            umap_df.to_pickle(f"{tmpdirname}/umap.tumap")

            widget, _ = viewer.window.add_plugin_dock_widget('napari-tomotwin', widget_name='Cluster UMAP embeddings')
            worker = load_umap(label_layer=viewer.layers[0], filename=f"{tmpdirname}/umap.tumap")

            def draw():
                _draw_circle((52.60765063119348, 33.616464739122755),viewer.layers[0], lumap.umap)

            worker.returned.connect(_draw_circle)
            worker.start()
            assert True # just make sure that now exception is raised

if __name__ == '__main__':
    unittest.main()
