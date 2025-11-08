"""Self-Learning episodic memory with optional FAISS acceleration."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Optional dependency management
# ---------------------------------------------------------------------------

# Allow libomp/libiomp to coexist when FAISS and PyTorch are both installed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_FAISS_IMPORT_ERROR: Optional[BaseException] = None
_FAISS_MODULE = None


def _load_faiss():
    """Load FAISS lazily to avoid hard dependency conflicts."""

    global _FAISS_MODULE, _FAISS_IMPORT_ERROR

    if _FAISS_MODULE is not None or _FAISS_IMPORT_ERROR is not None:
        return _FAISS_MODULE

    spec = importlib.util.find_spec("faiss")
    if spec is None:
        _FAISS_IMPORT_ERROR = ModuleNotFoundError("faiss module not found")
        return None

    try:
        _FAISS_MODULE = importlib.import_module("faiss")
    except BaseException as exc:  # pragma: no cover - defensive guard
        _FAISS_IMPORT_ERROR = exc
        _FAISS_MODULE = None

    return _FAISS_MODULE

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Memory episode"""
    episode_id: int
    timestamp: float
    state: Dict
    action: Dict
    reward: float
    next_state: Dict
    metadata: Dict
    
    def to_dict(self):
        return asdict(self)


class StateEmbedder:
    """Embed states into vector space for similarity search"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()
        self.fitted = False
    
    def extract_features(self, state: Dict) -> np.ndarray:
        """Extract numerical features from state"""
        features = []
        
        # Machine features (averaged across machines)
        machines = state.get('machines', [])
        if machines:
            avg_utilization = np.mean([m['utilization'] for m in machines])
            avg_next_available = np.mean([m['next_available'] for m in machines])
            total_reconfigs = np.sum([m['reconfigurations'] for m in machines])
            features.extend([avg_utilization, avg_next_available, total_reconfigs])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Job queue features
        job_queue = state.get('job_queue', [])
        if job_queue:
            avg_remaining_ops = np.mean([j['remaining_operations'] for j in job_queue])
            avg_waiting_time = np.mean([j['waiting_time'] for j in job_queue])
            avg_slack = np.mean([j['slack'] for j in job_queue])
            num_urgent = sum(1 for j in job_queue if j['priority'] >= 8)
            features.extend([len(job_queue), avg_remaining_ops, avg_waiting_time, 
                           avg_slack, num_urgent])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Performance metrics
        metrics = state.get('metrics', {})
        features.extend([
            metrics.get('makespan', 0.0),
            metrics.get('avg_tardiness', 0.0),
            metrics.get('utilization', 0.0),
            metrics.get('energy_consumption', 0.0),
            metrics.get('reconfigurations', 0.0)
        ])
        
        # System state
        features.extend([
            state.get('current_time', 0.0),
            state.get('pending_jobs', 0),
            state.get('completed_jobs', 0)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def fit_transform(self, states: List[Dict]) -> np.ndarray:
        """Fit scaler and transform states"""
        features = np.array([self.extract_features(s) for s in states])
        
        # Handle NaN and inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.scaler.fit(features)
        self.fitted = True
        
        scaled = self.scaler.transform(features)
        return scaled.astype(np.float32)
    
    def transform(self, state: Dict) -> np.ndarray:
        """Transform single state"""
        features = self.extract_features(state).reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if self.fitted:
            scaled = self.scaler.transform(features)
        else:
            scaled = features
        
        return scaled.astype(np.float32)[0]


class MemorySystem:
    """Self-learning memory system with episodic storage"""

    def __init__(self,
                 max_episodes: int = 10000,
                 embedding_dim: int = 128,
                 save_dir: str = "data/memory"):
        self.max_episodes = max_episodes
        self.embedding_dim = embedding_dim
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Episode storage
        self.episodes: List[Episode] = []
        self.episode_counter = 0
        
        # State embeddings for similarity search
        self.embedder = StateEmbedder(embedding_dim)
        self.index = None  # Optional FAISS index
        self._embedding_matrix: Optional[np.ndarray] = None

        # Performance tracking
        self.config_performance = {}  # config_id -> List[performance]
        self.strategy_performance = {}  # strategy -> List[performance]
        self.assignment_history = []  # List of assignments
        
        # Statistics
        self.stats = {
            'total_episodes': 0,
            'avg_reward': 0.0,
            'best_reward': float('-inf'),
            'worst_reward': float('inf')
        }
    
    def store_episode(self, state: Dict, action: Dict, reward: float, 
                     next_state: Dict = None, metadata: Dict = None):
        """Store new episode in memory"""
        episode = Episode(
            episode_id=self.episode_counter,
            timestamp=state.get('current_time', 0.0),
            state=state,
            action=action,
            reward=reward,
            next_state=next_state or state,
            metadata=metadata or {}
        )
        
        self.episodes.append(episode)
        self.episode_counter += 1
        
        # Update statistics
        self.stats['total_episodes'] += 1
        self.stats['avg_reward'] = (
            (self.stats['avg_reward'] * (self.stats['total_episodes'] - 1) + reward) 
            / self.stats['total_episodes']
        )
        self.stats['best_reward'] = max(self.stats['best_reward'], reward)
        self.stats['worst_reward'] = min(self.stats['worst_reward'], reward)
        
        # Prune if exceeding max
        if len(self.episodes) > self.max_episodes:
            # Keep recent episodes and high-reward episodes
            sorted_episodes = sorted(self.episodes, key=lambda e: e.reward, reverse=True)
            high_reward = sorted_episodes[:self.max_episodes // 4]
            recent = self.episodes[-3 * self.max_episodes // 4:]
            self.episodes = list(set(high_reward + recent))
        
        logger.debug(f"Stored episode {episode.episode_id} with reward {reward:.2f}")

    def build_index(self):
        """Build FAISS index for similarity search"""
        if len(self.episodes) < 10:
            logger.warning("Too few episodes to build index")
            return

        # Extract states and fit embedder
        states = [ep.state for ep in self.episodes]
        embeddings = self.embedder.fit_transform(states)

        self._embedding_matrix = embeddings

        faiss_module = _load_faiss()
        if faiss_module is None:
            if _FAISS_IMPORT_ERROR is not None:
                logger.warning(
                    "FAISS unavailable (%s); falling back to numpy similarity search",
                    _FAISS_IMPORT_ERROR,
                )
            else:  # pragma: no cover - defensive fallback
                logger.warning("FAISS unavailable; falling back to numpy similarity search")
            self.index = None
            return

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss_module.IndexFlatL2(dimension)
        self.index.add(embeddings)

        logger.info(f"Built FAISS index with {len(self.episodes)} episodes")

    def find_similar_states(self, current_state: Dict, k: int = 5) -> List[Dict]:
        """Find k most similar past states"""
        if len(self.episodes) == 0 or self._embedding_matrix is None:
            return []

        # Embed current state
        query_embedding = self.embedder.transform(current_state).reshape(1, -1)

        # Search
        k_actual = min(k, len(self.episodes))
        if self.index is not None:
            distances, indices = self.index.search(query_embedding, k_actual)
            indices_iter = zip(indices[0], distances[0])
        else:
            # Brute-force fallback using numpy distances
            diff = self._embedding_matrix[: len(self.episodes)] - query_embedding
            distances_np = np.linalg.norm(diff, axis=1)
            nearest_idx = np.argsort(distances_np)[:k_actual]
            indices_iter = ((idx, distances_np[idx]) for idx in nearest_idx)

        # Retrieve episodes
        similar_episodes = []
        for idx, dist in indices_iter:
            if idx < len(self.episodes):
                ep = self.episodes[idx]
                similar_episodes.append({
                    'episode_id': ep.episode_id,
                    'state': ep.state,
                    'action': ep.action,
                    'reward': ep.reward,
                    'distance': float(dist),
                    'metadata': ep.metadata
                })
        
        return similar_episodes
    
    def record_assignment(self, job_id: int, op_id: int, 
                         machine_id: int, result: Dict):
        """Record operation assignment"""
        self.assignment_history.append({
            'job_id': job_id,
            'op_id': op_id,
            'machine_id': machine_id,
            'success': result.get('success', False),
            'processing_time': result.get('processing_time', 0.0),
            'timestamp': result.get('actual_start_time', 0.0)
        })
    
    def query_config_performance(self, operation_types: set) -> Dict:
        """Query historical performance of configurations"""
        # Return stored configuration performance
        return self.config_performance.copy()
    
    def get_config_history(self, config_id: str) -> List[Dict]:
        """Get historical performance for specific configuration"""
        return self.config_performance.get(config_id, [])
    
    def update_config_performance(self, config_id: str, performance: Dict):
        """Update configuration performance statistics"""
        if config_id not in self.config_performance:
            self.config_performance[config_id] = []
        
        self.config_performance[config_id].append(performance)
        
        # Keep only recent records
        if len(self.config_performance[config_id]) > 100:
            self.config_performance[config_id] = self.config_performance[config_id][-100:]
    
    def get_strategy_performance(self, strategy: str) -> Dict:
        """Get performance statistics for scheduling strategy"""
        if strategy not in self.strategy_performance:
            return {'count': 0, 'avg_makespan': float('inf'), 'success_rate': 0.0}
        
        records = self.strategy_performance[strategy]
        
        return {
            'count': len(records),
            'avg_makespan': np.mean([r['makespan'] for r in records]),
            'avg_tardiness': np.mean([r['tardiness'] for r in records]),
            'success_rate': np.mean([r['success'] for r in records])
        }
    
    def update_strategy_performance(self, strategy: str, performance: Dict):
        """Update strategy performance"""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        
        self.strategy_performance[strategy].append(performance)
        
        # Keep only recent records
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]
    
    def save(self, filename: str = "memory.pkl"):
        """Save memory to disk"""
        save_path = self.save_dir / filename
        
        data = {
            'episodes': self.episodes,
            'episode_counter': self.episode_counter,
            'config_performance': self.config_performance,
            'strategy_performance': self.strategy_performance,
            'assignment_history': self.assignment_history,
            'stats': self.stats
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Memory saved to {save_path}")
    
    def load(self, filename: str = "memory.pkl"):
        """Load memory from disk"""
        load_path = self.save_dir / filename
        
        if not load_path.exists():
            logger.warning(f"Memory file not found: {load_path}")
            return
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.episodes = data['episodes']
        self.episode_counter = data['episode_counter']
        self.config_performance = data['config_performance']
        self.strategy_performance = data['strategy_performance']
        self.assignment_history = data['assignment_history']
        self.stats = data['stats']
        
        # Rebuild index
        self.build_index()
        
        logger.info(f"Memory loaded from {load_path} ({len(self.episodes)} episodes)")
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            **self.stats,
            'num_episodes': len(self.episodes),
            'num_configs_tracked': len(self.config_performance),
            'num_strategies_tracked': len(self.strategy_performance),
            'num_assignments': len(self.assignment_history)
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export episodes to pandas DataFrame"""
        records = []
        for ep in self.episodes:
            record = {
                'episode_id': ep.episode_id,
                'timestamp': ep.timestamp,
                'reward': ep.reward,
                **{f'meta_{k}': v for k, v in ep.metadata.items()}
            }
            records.append(record)
        
        return pd.DataFrame(records)