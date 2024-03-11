from VQ_Loss_Plotter import VQ_Loss_Plotter
import unittest


class TestThresholds(unittest.TestCase):
    def test_thresholds_ex(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.thresholds(
                data_splits=["ex"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Thresholds/",
            ),
            None,
        )

    def test_thresholds_vs(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.thresholds(
                data_splits=["vs"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Thresholds/",
            ),
            None,
        )

    def test_thresholds_vd(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.thresholds(
                data_splits=["vd"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Thresholds/",
            ),
            None,
        )

    def test_thresholds_vs_inv(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.thresholds(
                data_splits=["vs-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Thresholds/",
            ),
            None,
        )

    def test_thresholds_vd_inv(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.thresholds(
                data_splits=["vd-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Thresholds/",
            ),
            None,
        )


class TestHeatmaps(unittest.TestCase):
    def test_heatmaps_ex(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.heatmaps(
                data_splits=["ex"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Heatmap/",
            ),
            None,
        )

    def test_heatmaps_vs(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.heatmaps(
                data_splits=["vs"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Heatmap/",
            ),
            None,
        )

    def test_heatmaps_vd(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.heatmaps(
                data_splits=["vd"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Heatmap/",
            ),
            None,
        )

    def test_heatmaps_vs_inv(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.heatmaps(
                data_splits=["vs-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Heatmap/",
            ),
            None,
        )

    def test_heatmaps_vd_inv(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.heatmaps(
                data_splits=["vd-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Heatmap/",
            ),
            None,
        )


class TestBoxplots(unittest.TestCase):
    def test_boxplots_ex(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.boxplots(
                data_splits=["ex"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )

    def test_boxplots_vs(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.boxplots(
                data_splits=["vs"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )

    def test_boxplots_vd(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.boxplots(
                data_splits=["vd"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )

    def test_boxplots_vs_inv(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.boxplots(
                data_splits=["vs-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )

    def test_boxplots_vd_inv(self):
        plotter = VQ_Loss_Plotter()
        self.assertEqual(
            plotter.boxplots(
                data_splits=["vd-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )


if __name__ == "__main__":
    unittest.main()
