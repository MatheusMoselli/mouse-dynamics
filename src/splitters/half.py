"""
There will be the same amount of authentic data and unauthentic. Also, the unauthentic data will be populated
by taking the same amount of data from each of the other users.
"""
import gc
import logging
import pandas as pd
from pathlib import Path
from src.dto import ExtractionData, EnumTypeOfSession
from src.splitters import BaseSplitter

logger = logging.getLogger(__name__)

_FEATURES_PATH = Path("../datasets/features")
_SPLIT_PATH    = Path("../datasets/split")


class HalfSplitter(BaseSplitter):
    """
    Split the dataset into train/test sets.
    The features should already be extracted at this point.
    """

    def split(self, extraction_data: ExtractionData) -> ExtractionData:
        """
        Split the dataset into train/test sets.
        :param extraction_data:  the list of datasets to be split
        :return: The list of train/test sets
        """
        if self.is_memory_efficiency_necessary:
            return self._split_from_disk(extraction_data)
        
        return self._split_in_memory(extraction_data)
    
    @staticmethod
    def _read_user_training_from_disk(user_id: str) -> pd.DataFrame:
        """
        Re-read all training session Parquets written by BasePreprocessor
        and return them as a single concatenated DataFrame.

        :param user_id: the user whose files to read
        :return: merged training DataFrame
        :raises FileNotFoundError: if the expected directory does not exist
        """
        training_dir = _FEATURES_PATH / f"user{user_id}" / "training"
        
        if not training_dir.exists():
            raise FileNotFoundError(
                f"Debug Parquet directory not found for user {user_id}: {training_dir}."
            )
            
        parquet_files = sorted(training_dir.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found in {training_dir} for user {user_id}."
            )
            
        return pd.concat(
            (pd.read_parquet(p) for p in parquet_files), ignore_index=True
        )

    @staticmethod
    def _read_split_from_disk(user_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Re-read the training/testing Parquets written by _write_debug_file.
        """
        
        user_dir = _SPLIT_PATH / f"user{user_id}"
        
        train_df = pd.concat(
            (pd.read_parquet(p) for p in sorted((user_dir / "training").glob("*.parquet"))),
            ignore_index=True,
        )
        
        test_df = pd.concat(
            (pd.read_parquet(p) for p in sorted((user_dir / "testing").glob("*.parquet"))),
            ignore_index=True,
        )
        
        return train_df, test_df

    def _split_from_disk(self, extraction_data: ExtractionData) -> ExtractionData:
        """
        Processes ONE user at a time.
        The assembled final_training_df is written to disk and immediately
        freed before the next user starts — never holding more than one
        large merged DataFrame in RAM at the same time.
        """
        users = extraction_data.users
        n_support = len(users) - 1

        logger.info("HalfSplitter (disk mode): reading training sizes.")
        authentic_sizes: dict[str, int] = {}
        for user in users:
            df = self._read_user_training_from_disk(user.id)
            authentic_sizes[user.id] = len(df)
            del df
        gc.collect()

        testing_cache: dict[str, pd.DataFrame] = {}
        for user in users:
            testing_cache[user.id] = user.merged_sessions(EnumTypeOfSession.TESTING)
            user.training_sessions = {}
            user.testing_sessions  = {}
        gc.collect()

        for user in users:
            logger.info(f"HalfSplitter: splitting user {user.id}")
            per_support_size = authentic_sizes[user.id] // n_support

            final_training_df = self._read_user_training_from_disk(user.id)

            for support_user in users:
                if support_user.id == user.id:
                    continue
                
                support_df = self._read_user_training_from_disk(support_user.id)
                support_df = support_df.iloc[:per_support_size].copy()
                support_df["authentic"] = 0
                
                final_training_df = pd.concat(
                    [final_training_df, support_df], ignore_index=True
                )
                del support_df
                gc.collect()

            user.training_sessions = {"_merged": final_training_df}
            user.testing_sessions  = {"_merged": testing_cache[user.id]}
            self._write_debug_file(user)

            user.training_sessions = {}
            user.testing_sessions  = {}
            del final_training_df
            testing_cache[user.id] = pd.DataFrame() 
            gc.collect()

        del testing_cache
        gc.collect()

        logger.info("HalfSplitter: reloading split files into UserDataDtos.")
        for user in users:
            train_df, test_df = self._read_split_from_disk(user.id)
            user.training_sessions = {"_merged": train_df}
            user.testing_sessions  = {"_merged": test_df}

        gc.collect()
        return extraction_data

    def _split_in_memory(self, extraction_data: ExtractionData) -> ExtractionData:
        users = extraction_data.users
        n_support = len(users) - 1

        training_cache: dict[str, pd.DataFrame] = {}
        for user in users:
            training_cache[user.id] = user.merged_sessions(EnumTypeOfSession.TRAINING)
            user.training_sessions  = {}
        gc.collect()

        for user in users:
            true_user_df = training_cache[user.id].copy()
            per_support_size = len(true_user_df) // n_support
            all_training_dfs = [true_user_df]

            for support_user in users:
                if support_user.id == user.id:
                    continue
                
                support_df = training_cache[support_user.id].iloc[:per_support_size].copy()
                support_df["authentic"] = 0
                all_training_dfs.append(support_df)

            final_training_df = pd.concat(all_training_dfs, ignore_index=True)
            del all_training_dfs
            
            merged_testing_df = user.merged_sessions(EnumTypeOfSession.TESTING)
            user.training_sessions = {"_merged": final_training_df}
            user.testing_sessions = {"_merged": merged_testing_df}

            if self.is_debug:
                self._write_debug_file(user)

        training_cache.clear()
        gc.collect()
        
        return extraction_data
