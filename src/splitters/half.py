"""
There will be the same amount of authentic data and unauthentic. Also, the unauthentic data will be populated
by taking the same amount of data from each of the other users.
"""
import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.dto import ExtractionData, EnumTypeOfSession
from src.splitters import BaseSplitter
from src.utils.split_cache import split_dir, write_merged

logger = logging.getLogger(__name__)

try:
    import psutil
    _process = psutil.Process()

    def _mem_mb() -> float:
        return _process.memory_info().rss / (1024 ** 2)

except ImportError:
    import resource

    def _mem_mb() -> float:
        # ru_maxrss is KB on Linux, bytes on macOS -- assume Linux (typical prod VM).
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _log_mem(stage: str) -> None:
    try:
        logger.info("[HalfSplitter] %s | RSS memory: %.1f MB", stage, _mem_mb())
    except Exception:
        # Memory logging must never break the actual pipeline.
        logger.debug("[HalfSplitter] %s | (failed to read memory usage)", stage)


def _df_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast float64 -> float32 and int64 -> the smallest safe int dtype, in place.
    Cuts memory footprint of numeric feature columns roughly in half without
    touching non-numeric columns (ids, labels handled elsewhere, etc).
    Skip this if you need float64 precision downstream (e.g. some stats tests).
    """
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype(np.float32)

    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


class HalfSplitter(BaseSplitter):
    """
    Split the dataset into train/test sets.
    The features should already be extracted at this point.

    Each user's final train/test split is written to disk (parquet, via
    src.utils.split_cache) right after it's built, then dropped from memory.
    Peak RAM during split() stays roughly flat regardless of how many users
    are in the dataset -- classifiers read it back one user at a time via
    BaseClassifier._prepare_user_data (see set_split_location there).
    """

    DEFAULT_OUTPUT_DIR = Path("../datasets/split")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = self.DEFAULT_OUTPUT_DIR

    def split(
        self,
        extraction_data: ExtractionData,
        downcast_dtypes: bool = True,
        gc_every_n_users: int = 5,
        log_every_n_users: int = 1,
    ) -> ExtractionData:
        """
        Split the dataset into train/test sets.

        :param extraction_data: the list of datasets to be split
        :param downcast_dtypes: if True, casts float64->float32 / int64->smaller int
            on the cached data before building the splits. Big memory win, tiny
            precision cost. Set False if you need float64 downstream.
        :param gc_every_n_users: run gc.collect() every N processed users, not just
            at the start/end, to release freed temporaries promptly during long runs.
        :param log_every_n_users: log per-user progress every N users (set higher
            to reduce log volume on datasets with many users).
        :return: The list of train/test sets. user.training_sessions and
            user.testing_sessions are left EMPTY after this runs -- the data
            lives on disk under self.output_dir (see src.utils.split_cache).
            Classifiers read it back lazily, one user at a time.
        """
        t_start = time.monotonic()
        users = extraction_data.users
        n_users = len(users)
        n_support = n_users - 1

        logger.info(
            "[HalfSplitter] START | users=%d n_support=%d downcast_dtypes=%s output_dir=%s",
            n_users, n_support, downcast_dtypes, self.output_dir,
        )
        _log_mem("before cache build")

        if n_support <= 0:
            logger.error(
                "[HalfSplitter] ABORT | need at least 2 users to build impostor data, got %d",
                n_users,
            )
            raise ValueError("HalfSplitter requires at least 2 users")

        training_cache: dict[str, pd.DataFrame] = {}
        testing_cache: dict[str, pd.DataFrame] = {}

        for idx, user in enumerate(users, start=1):
            try:
                logger.debug("[HalfSplitter] caching user=%s (%d/%d)", user.id, idx, n_users)
                train_df = user.merged_sessions(EnumTypeOfSession.TRAINING)
                test_df = user.merged_sessions(EnumTypeOfSession.TESTING)

                logger.debug(
                    "[HalfSplitter] user=%s raw sizes | train=%d rows (%.1f MB) test=%d rows (%.1f MB)",
                    user.id, len(train_df), _df_mb(train_df), len(test_df), _df_mb(test_df),
                )

                if downcast_dtypes:
                    train_df = _downcast_numeric(train_df)
                    test_df = _downcast_numeric(test_df)
                    logger.debug(
                        "[HalfSplitter] user=%s after downcast | train=%.1f MB test=%.1f MB",
                        user.id, _df_mb(train_df), _df_mb(test_df),
                    )

                training_cache[user.id] = train_df
                testing_cache[user.id] = test_df

                user.training_sessions = {}
                user.testing_sessions = {}

            except Exception as e:
                logger.exception(
                    "[HalfSplitter] FAILED while caching user=%s (%d/%d) -- likely OOM or bad data here",
                    getattr(user, "id", "?"), idx, n_users,
                )
                
                print("ERRO: ")
                print(e)
                
                raise

        gc.collect()
        cache_total_mb = sum(_df_mb(df) for df in training_cache.values()) + sum(
            _df_mb(df) for df in testing_cache.values()
        )
        logger.info(
            "[HalfSplitter] cache built | %d users cached | total cache size ~%.1f MB", n_users, cache_total_mb
        )
        _log_mem("after cache build")

        for processed, user in enumerate(users, start=1):
            user_t_start = time.monotonic()
            try:
                true_user_training_df = training_cache[user.id]
                true_user_test_df = testing_cache[user.id]
                has_impostors_in_test = (true_user_test_df["authentic"] == 0).any()

                authentic_training_df_size = len(true_user_training_df)
                training_per_support_size = authentic_training_df_size // n_support

                all_training_dfs = [true_user_training_df]

                authentic_test_df_size = len(true_user_test_df)
                test_per_support_size = authentic_test_df_size // n_support

                all_testing_dfs = [true_user_test_df]

                if training_per_support_size == 0 or test_per_support_size == 0:
                    logger.warning(
                        "[HalfSplitter] user=%s has very little data (train=%d, test=%d) -- "
                        "per-support sample size rounded to 0, impostor set may be empty/tiny",
                        user.id, authentic_training_df_size, authentic_test_df_size,
                    )

                for support_user in users:
                    if support_user.id == user.id:
                        continue

                    seed = int(user.id) * 1000 + int(support_user.id) + self.seed_number

                    try:
                        if len(training_cache[support_user.id]) <= training_per_support_size:
                            # Copy is required here: without it we'd mutate the shared
                            # cache entry below (support_training_df["authentic"] = 0),
                            # silently corrupting that user's data for later iterations.
                            support_training_df = training_cache[support_user.id].copy()
                        else:
                            support_training_df = training_cache[support_user.id].sample(
                                training_per_support_size,
                                random_state=seed,
                            ).copy()

                        support_training_df["authentic"] = 0
                        all_training_dfs.append(support_training_df)

                        if not has_impostors_in_test:
                            support_test_df = testing_cache[support_user.id].sample(
                                test_per_support_size,
                                random_state=seed,
                            ).copy()

                            support_test_df["authentic"] = 0
                            all_testing_dfs.append(support_test_df)

                    except Exception:
                        logger.exception(
                            "[HalfSplitter] FAILED building support data | user=%s support_user=%s "
                            "training_per_support_size=%d test_per_support_size=%d",
                            user.id, support_user.id, training_per_support_size, test_per_support_size,
                        )
                        raise

                final_training_df = pd.concat(all_training_dfs, ignore_index=True, copy=False)
                final_testing_df = pd.concat(all_testing_dfs, ignore_index=True, copy=False)

                del all_training_dfs, all_testing_dfs

                # Persist to the classifier hand-off cache.
                write_merged(
                    split_dir(self.output_dir, user.id, "training", self.seed_number),
                    final_training_df,
                )
                write_merged(
                    split_dir(self.output_dir, user.id, "testing", self.seed_number),
                    final_testing_df,
                )

                if self.is_debug:
                    # Debug export uses the project's own log_dataframe_sessions
                    # mechanism, unrelated to (and non-colliding with) the cache above.
                    user.training_sessions = {"_merged": final_training_df}
                    user.testing_sessions = {"_merged": final_testing_df}
                    self._write_debug_file(user)

                # Data is safely on disk now -- drop it from memory so RAM stays
                # flat regardless of how many users are left to process.
                user.training_sessions = {}
                user.testing_sessions = {}
                del final_training_df, final_testing_df

            except Exception:
                logger.exception(
                    "[HalfSplitter] FAILED while processing user=%s (%d/%d) -- pipeline stopped here",
                    getattr(user, "id", "?"), processed, n_users,
                )
                raise

            if processed % log_every_n_users == 0 or processed == n_users:
                logger.info(
                    "[HalfSplitter] processed+persisted user=%s (%d/%d) in %.2fs",
                    user.id, processed, n_users, time.monotonic() - user_t_start,
                )
                _log_mem(f"after user {user.id} ({processed}/{n_users})")

            if gc_every_n_users and processed % gc_every_n_users == 0:
                gc.collect()
                logger.debug("[HalfSplitter] gc.collect() ran after %d users", processed)

        training_cache.clear()
        testing_cache.clear()
        gc.collect()

        _log_mem("after final cleanup")
        logger.info(
            "[HalfSplitter] DONE | %d users processed and persisted to %s in %.2fs",
            n_users, self.output_dir, time.monotonic() - t_start,
        )

        return extraction_data
