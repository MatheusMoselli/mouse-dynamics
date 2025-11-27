"""
Feature extractors for mouse dynamics.
Contains the main MouseDynamicsExtractor and batch processing utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


# Import the main extractor (from your previous implementation)
# This is the MouseDynamicsExtractor class you created earlier


class BatchFeatureExtractor:
    """Extract features from multiple trajectories in parallel."""

    def __init__(self,
                 extractor: MouseDynamicsExtractor = None,
                 n_jobs: int = -1):
        """
        Initialize batch feature extractor.

        Args:
            extractor: MouseDynamicsExtractor instance (creates default if None)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.extractor = extractor or MouseDynamicsExtractor()
        self.n_jobs = n_jobs

    def extract_user_features(self,
                              user_data: Dict[str, pd.DataFrame],
                              segment_trajectories: bool = False) -> pd.DataFrame:
        """
        Extract features from all users in the dataset.

        Args:
            user_data: Dictionary mapping user_id to trajectory DataFrame
            segment_trajectories: Whether to segment long trajectories

        Returns:
            DataFrame with features for each user/segment
        """
        results = []

        # Prepare tasks
        tasks = []
        for user_id, df in user_data.items():
            if segment_trajectories:
                from ..data.preprocessors import TrajectorySegmenter
                segmenter = TrajectorySegmenter(method='pause', min_points=10)
                segments = segmenter.segment(df)

                for seg_idx, segment in enumerate(segments):
                    tasks.append((user_id, seg_idx, segment))
            else:
                tasks.append((user_id, 0, df))

        # Extract features in parallel
        if self.verbose:
            print(f"Extracting features from {len(tasks)} trajectories...")

        features_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_single)(user_id, seg_idx, df)
            for user_id, seg_idx, df in tqdm(tasks, disable=not self.verbose)
        )

        # Combine results
        for features in features_list:
            if features is not None:
                results.append(features)

        if not results:
            logger.warning("No features extracted!")
            return pd.DataFrame()

        # Create DataFrame
        features_df = pd.DataFrame(results)

        logger.info(f"Extracted {len(features_df)} feature vectors with "
                    f"{len(features_df.columns) - 2} features each")

        return features_df

    def _extract_single(self, user_id: str, seg_idx: int,
                        df: pd.DataFrame) -> Optional[Dict]:
        """Extract features from a single trajectory."""
        try:
            features = self.extractor.extract_features(df)
            features['user_id'] = user_id
            features['segment_id'] = seg_idx
            return features
        except Exception as e:
            logger.error(f"Failed to extract features for {user_id}, segment {seg_idx}: {e}")
            return None

    def save_features(self, features_df: pd.DataFrame,
                      output_path: Union[str, Path]):
        """Save extracted features to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.csv':
            features_df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.pkl', '.pickle']:
            features_df.to_pickle(output_path)
        elif output_path.suffix == '.parquet':
            features_df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        logger.info(f"Features saved to {output_path}")