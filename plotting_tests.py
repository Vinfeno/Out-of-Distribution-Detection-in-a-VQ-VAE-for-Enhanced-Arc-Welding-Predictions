from VQ_Loss_Plotter import VQ_Loss_Plotter
import unittest
import torch
import matplotlib.pyplot as plt


class TestThresholds(unittest.TestCase):
    def setUp(self) -> None:
        self.plotter = VQ_Loss_Plotter()

    def test_thresholds_ex(self):
        self.assertEqual(
            self.plotter.thresholds(
                data_splits=["ex"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Thresholds/",
            ),
            None,
        )

    def test_thresholds_vs(self):
        self.assertEqual(
            self.plotter.thresholds(
                data_splits=["vs"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Thresholds/",
            ),
            None,
        )

    def test_thresholds_vd(self):
        self.assertEqual(
            self.plotter.thresholds(
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

    def test_thresholds_all(self):
        self.assertEqual(
            self.plotter.thresholds(
                data_splits=["ex"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Thresholds/",
                all=True,
            ),
            None,
        )


class TestBoxplots(unittest.TestCase):
    def setUp(self) -> None:
        self.plotter = VQ_Loss_Plotter()

    def test_boxplots_ex(self):
        self.assertEqual(
            self.plotter.boxplots(
                data_splits=["ex"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )

    def test_boxplots_vs(self):
        self.assertEqual(
            self.plotter.boxplots(
                data_splits=["vs"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )

    def test_boxplots_vd(self):
        self.assertEqual(
            self.plotter.boxplots(
                data_splits=["vd"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )

    def test_boxplots_vs_inv(self):
        self.assertEqual(
            self.plotter.boxplots(
                data_splits=["vs-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )

    def test_boxplots_vd_inv(self):
        self.assertEqual(
            self.plotter.boxplots(
                data_splits=["vd-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Boxplot/",
            ),
            None,
        )


class TestNoise(unittest.TestCase):
    def setUp(self) -> None:
        self.plotter = VQ_Loss_Plotter()

    def test_noise_generation(self):
        noise_data = self.plotter.get_noise_data()
        sample = noise_data[0][0]

        # Plotting both time series on the same graph
        plt.figure(figsize=(10, 6))  # Set the figure size for better readability
        plt.plot(sample[:, 0], color="blue")  # Plot the first column
        plt.plot(sample[:, 1], color="red")  # Plot the second column

        plt.title("Random Cycle")  # Title of the plot
        plt.xlabel("Time")  # X-axis label
        plt.ylabel("Simulated Current/Voltage")  # Y-axis label
        # plt.legend()  # Show legend to identify the lines
        plt.savefig("images/Noise/RandomCycle.png")

    def test_on_noise_ex(self):
        self.assertEqual(
            self.plotter.test_on_noise(
                data_splits=["ex"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Noise/",
            ),
            None,
        )

    def test_on_noise_ex_inv(self):
        self.assertEqual(
            self.plotter.test_on_noise(
                data_splits=["ex-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Noise/",
            ),
            None,
        )

    def test_on_noise_vs(self):
        self.assertEqual(
            self.plotter.test_on_noise(
                data_splits=["vs"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Noise/",
            ),
            None,
        )

    def test_on_noise_vd(self):
        self.assertEqual(
            self.plotter.test_on_noise(
                data_splits=["vd"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Noise/",
            ),
            None,
        )

    def test_on_noise_vs_inv(self):
        self.assertEqual(
            self.plotter.test_on_noise(
                data_splits=["vs-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Noise/",
            ),
            None,
        )

    def test_on_noise_vd_inv(self):
        self.assertEqual(
            self.plotter.test_on_noise(
                data_splits=["vd-inv"],
                epochs=[10],
                embeddings=[16],
                betas=[0.25],
                image_path="images/tests/Noise/",
            ),
            None,
        )


if __name__ == "__main__":
    unittest.main()

# class TestHeatmaps(unittest.TestCase):
#     def setUp(self) -> None:
#         self.plotter = VQ_Loss_Plotter()

#     def test_heatmaps_ex(self):
#         self.assertEqual(
#             self.plotter.heatmaps(
#                 data_splits=["ex"],
#                 epochs=[10],
#                 embeddings=[16],
#                 betas=[0.25],
#                 image_path="images/tests/Heatmap/",
#             ),
#             None,
#         )

#     def test_heatmaps_vs(self):
#         self.assertEqual(
#             self.plotter.heatmaps(
#                 data_splits=["vs"],
#                 epochs=[10],
#                 embeddings=[16],
#                 betas=[0.25],
#                 image_path="images/tests/Heatmap/",
#             ),
#             None,
#         )

#     def test_heatmaps_vd(self):
#         self.assertEqual(
#             self.plotter.heatmaps(
#                 data_splits=["vd"],
#                 epochs=[10],
#                 embeddings=[16],
#                 betas=[0.25],
#                 image_path="images/tests/Heatmap/",
#             ),
#             None,
#         )

#     def test_heatmaps_vs_inv(self):
#         self.assertEqual(
#             self.plotter.heatmaps(
#                 data_splits=["vs-inv"],
#                 epochs=[10],
#                 embeddings=[16],
#                 betas=[0.25],
#                 image_path="images/tests/Heatmap/",
#             ),
#             None,
#         )

#     def test_heatmaps_vd_inv(self):
#         self.assertEqual(
#             self.plotter.heatmaps(
#                 data_splits=["vd-inv"],
#                 epochs=[10],
#                 embeddings=[16],
#                 betas=[0.25],
#                 image_path="images/tests/Heatmap/",
#             ),
#             None,
#         )
