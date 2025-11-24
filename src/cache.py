import os
import pickle
import tempfile
import shutil
import hashlib


class FileCache:
    """Flexible cache supporting in-memory, file-based, or no caching"""
    
    def __init__(self, cache_mode='memory', cache_dir='cache', dataset_name=None):
        """
        Args:
            cache_mode: 'memory' (default), 'file', or 'none'
            cache_dir: directory for file cache (default: 'cache')
            dataset_name: name of the dataset for cache file naming
        """
        self.cache_mode = cache_mode
        self.dataset_name = dataset_name
        
        if cache_mode == 'memory':
            # Use in-memory cache
            self.cache_data = {}
            self.cache_dir = None
        elif cache_mode == 'file':
            # Use file-based cache
            if dataset_name is None:
                raise ValueError("dataset_name cannot be None when cache_mode is 'file'")
            self.cache_data = None
            if cache_dir is None:
                self.cache_dir = tempfile.mkdtemp(prefix='dataset_cache_')
            else:
                self.cache_dir = cache_dir
                os.makedirs(self.cache_dir, exist_ok=True)
        else:  # cache_mode == 'none'
            # No caching
            self.cache_data = None
            self.cache_dir = None
    
    def _get_cache_path(self, item_id):
        """Get cache file path for given item_id with consistent naming"""
        # Use dataset name in filename to allow cache reuse across training runs
        filename = f'{self.dataset_name}_{item_id}.pkl'
        return os.path.join(self.cache_dir, filename)
    
    def get(self, item_id):
        """Load data from cache"""
        if self.cache_mode == 'none':
            return None
        
        if self.cache_mode == 'memory':
            # Load from memory cache
            return self.cache_data.get(item_id)
        else:  # cache_mode == 'file'
            # Load from file cache
            cache_path = self._get_cache_path(item_id)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except (pickle.PickleError, EOFError, FileNotFoundError, ModuleNotFoundError, ImportError):
                    # Remove corrupted cache file (may be due to version incompatibility)
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
            return None
    
    def set(self, item_id, data):
        """Save data to cache"""
        if self.cache_mode == 'none':
            return
        
        if self.cache_mode == 'memory':
            # Save to memory cache
            self.cache_data[item_id] = data
        else:  # cache_mode == 'file'
            # Save to file cache
            cache_path = self._get_cache_path(item_id)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            except (OSError, IOError) as e:
                print(f"Warning: Failed to save cache for item {item_id}: {e}")
    
    def cleanup(self):
        """Clean up cache"""
        if self.cache_mode == 'memory':
            # Clear memory cache
            self.cache_data.clear()
        elif self.cache_mode == 'file':
            # Clean up file cache directory
            if self.cache_dir and os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
        # No cleanup needed for 'none' mode
    
    def __del__(self):
        """Cleanup cache directory when object is destroyed"""
        # Note: This might not always be called due to Python's garbage collection
        # It's better to call cleanup() explicitly when done
        pass
