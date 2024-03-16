import unittest
from Split_Trainer import Split_Trainer


class TestSplitTrainer(unittest.TestCase):
    def setUp(self):
        # Initialize the SplitTrainer object
        self.split_trainer = Split_Trainer()
        self.epochs_list = [1]
        self.num_embeddings = [16]
        self.betas = [0.25]

    def test_train_multi_models_ex(self):
        # Define the test data
        data_split = ["ex"]
        self.split_trainer.train_multi_models(
            data_splits=data_split,
            epochs_list=self.epochs_list,
            num_embeddings=self.num_embeddings,
            betas=self.betas,
        )
        self.assertEqual(
            self.split_trainer.train_multi_models(
                data_splits=data_split,
                epochs_list=self.epochs_list,
                num_embeddings=self.num_embeddings,
                betas=self.betas,
            ),
            None,
        )

    def test_train_multi_models_vs(self):
        # Define the test data
        data_split = ["vs"]
        self.split_trainer.train_multi_models(
            data_splits=data_split,
            epochs_list=self.epochs_list,
            num_embeddings=self.num_embeddings,
            betas=self.betas,
        )
        self.assertEqual(
            self.split_trainer.train_multi_models(
                data_splits=data_split,
                epochs_list=self.epochs_list,
                num_embeddings=self.num_embeddings,
                betas=self.betas,
            ),
            None,
        )

    def test_train_multi_models_vd(self):
        # Define the test data
        data_split = ["vd"]
        self.split_trainer.train_multi_models(
            data_splits=data_split,
            epochs_list=self.epochs_list,
            num_embeddings=self.num_embeddings,
            betas=self.betas,
        )
        self.assertEqual(
            self.split_trainer.train_multi_models(
                data_splits=data_split,
                epochs_list=self.epochs_list,
                num_embeddings=self.num_embeddings,
                betas=self.betas,
            ),
            None,
        )

    def test_train_multi_models_vs_inv(self):
        # Define the test data
        data_split = ["vs-inv"]
        self.split_trainer.train_multi_models(
            data_splits=data_split,
            epochs_list=self.epochs_list,
            num_embeddings=self.num_embeddings,
            betas=self.betas,
        )
        self.assertEqual(
            self.split_trainer.train_multi_models(
                data_splits=data_split,
                epochs_list=self.epochs_list,
                num_embeddings=self.num_embeddings,
                betas=self.betas,
            ),
            None,
        )

    def test_train_multi_models_vd_inv(self):
        # Define the test data
        data_split = ["vd-inv"]
        self.split_trainer.train_multi_models(
            data_splits=data_split,
            epochs_list=self.epochs_list,
            num_embeddings=self.num_embeddings,
            betas=self.betas,
        )
        self.assertEqual(
            self.split_trainer.train_multi_models(
                data_splits=data_split,
                epochs_list=self.epochs_list,
                num_embeddings=self.num_embeddings,
                betas=self.betas,
            ),
            None,
        )


if __name__ == "__main__":
    unittest.main()
