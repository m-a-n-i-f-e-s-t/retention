import shelve
import pickle
import os
from typing import Dict, Any, List, Tuple, Callable
from collections import OrderedDict


def canonicalize_dict(d: Dict[str, Any]) -> Any:
    """Recursively convert a dictionary to a canonically ordered representation."""
    if isinstance(d, dict):
        return OrderedDict(sorted(
            ((k, canonicalize_dict(v)) for k, v in d.items()),
            key=lambda x: str(x[0])
        ))
    elif isinstance(d, list):
        return [canonicalize_dict(x) for x in d]
    elif isinstance(d, tuple):
        return tuple(canonicalize_dict(x) for x in d)
    else:
        return d


def dict_to_key(key: Dict[str, Any]) -> str:
    """Convert a dictionary to a string for use as a shelve key.
    
    Ensures that the ordering of dictionary keys is canonical to avoid
    duplicate entries for equivalent dictionaries.
    """
    canonical_key = canonicalize_dict(key)
    return pickle.dumps(canonical_key).hex()


def key_to_dict(key_str: str) -> Dict[str, Any]:
    """Convert a string key back to a dictionary."""
    ordered_dict = pickle.loads(bytes.fromhex(key_str))
    # Convert OrderedDict back to regular dict if needed
    if isinstance(ordered_dict, OrderedDict):
        return dict(ordered_dict)
    return ordered_dict


def compare_dict(A: Dict[str, Any], B: Dict[str, Any]) -> bool:
    """Compare two dictionaries for equality by converting them to canonical form.
    
    Args:
        A: First dictionary to compare
        B: Second dictionary to compare
        
    Returns:
        True if the dictionaries are equivalent (same keys and values), False otherwise
    """
    return canonicalize_dict(A) == canonicalize_dict(B)


class KVDB:
    """A file-based key-value database for storing kernel benchmarking results."""
    
    def __init__(self, db_path: str):
        """Initialize the database with the given path.
        
        Args:
            db_path: Path to the database file (without extension)
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def __len__(self) -> int:
        """Return the number of entries in the database."""
        with shelve.open(self.db_path) as db:
            return len(db)
    
    def put(self, key: Dict[str, Any], val: Any) -> None:
        """Store a benchmark result.
        
        Args:
            key: Dictionary with benchmark parameters
            val: Benchmark result (any pickleable Python object)
        """
        key_str = dict_to_key(key)
        with shelve.open(self.db_path) as db:
            db[key_str] = val

    def post(self, table: str, key: Dict[str, Any], val: Any) -> None:
        """Store a benchmark result in a specific table.
        
        Args:
            table: Name of the table
            key: Dictionary with benchmark parameters
            val: Benchmark result (any pickleable Python object)
        """
        key = key | {'table': table}
        self.put(key, val)

    def get(self, key_fn: Callable[[Dict[str, Any]], bool] | None = None, 
              value_fn: Callable[[Any], Any] | None = None) -> List[Any]:
        """Query the database using filter functions. No order guaranteed on the results.
        
        Args:
            key_fn: Function that takes a key dict and returns True if it should be included
            value_fn: Optional function that processes or filters values. If it returns None,
                      the entry will be excluded from results.
                      
        Returns:
            List of (key, processed_value) pairs that match the filters
        """
        default_key_fn = lambda key: True
        results = []
        with shelve.open(self.db_path) as db:
            for key_str in db:
                key = key_to_dict(key_str)
                if (key_fn or default_key_fn)(key):
                    value = db[key_str]
                    
                    if value_fn is not None:
                        processed_value = value_fn(value)
                        if processed_value is not None:
                            results.append((key, processed_value))
                    else:
                        results.append((key, value))
        return results
    
    def query(self, table: str, key_fn: Callable[[Dict[str, Any]], bool] | None = None, 
              value_fn: Callable[[Any], Any] | None = None) -> List[Any]:
        """Query the database using filter functions.
        
        Args:
            table: Name of the table
            key_fn: Function that takes a key dict and returns True if it should be included
            value_fn: Optional function that processes or filters values. If it returns None,
                      the entry will be excluded from results.
                      
        Returns:
            List of (key, processed_value) pairs that match the filters
        """
        default_key_fn = lambda key: True
        return self.get(lambda key: key.get('table') == table and (key_fn or default_key_fn)(key), value_fn)

    def clear(self, fn: Callable[[Dict[str, Any]], bool] | None = None) -> int:
        """Clear entries from the database, optionally based on a filter function.
        
        Args:
            fn: Optional function that takes a key dict and returns True if it should be deleted.
                If None, all entries are cleared.
                
        Returns:
            Number of entries deleted
        """
        deleted_count = 0
        with shelve.open(self.db_path) as db:
            if fn is None:
                deleted_count = len(db)
                db.clear()
            else:
                keys_to_delete = []
                for key_str in db:
                    key = key_to_dict(key_str)
                    if fn(key):
                        keys_to_delete.append(key_str)
                
                for key_str in keys_to_delete:
                    del db[key_str]
                    deleted_count += 1
                    
        return deleted_count
    
    def delete(self, table: str, key_fn: Callable[[Dict[str, Any]], bool] | None = None) -> int:
        """Delete entries from the database, optionally based on a filter function.
        
        Args:
            table: Name of the table
            key_fn: Function that takes a key dict and returns True if it should be deleted.
        """
        default_key_fn = lambda key: True
        return self.clear(lambda key: key.get('table') == table and (key_fn or default_key_fn)(key))

    def __contains__(self, key: Dict[str, Any]) -> bool:
        """Check if a key is in the database."""
        with shelve.open(self.db_path) as db:
            key_str = dict_to_key(key)
            return key_str in db


if __name__ == "__main__":
    # Create a database
    db = KVDB("/tmp/test.db")
    db.clear()

    # Store benchmark results
    db.put({"kernel": "sympow", "power": 2, "d": 64, "b": 16}, 10.5)
    db.put({"power": 3, "d": 64, "kernel": "sympow", "b": 16}, 15.2)
    db.put({"power": 4, "d": 64, "kernel": "sympow", "layout": (2, (2, 3))}, 15.3)
    db.put({"power": 5, "d": 64, "kernel": "sympow", "layout": (2, (2, 1))}, dict(a=1, b=2))
    db.post('runtime', {"power": 5, "d": 64, "kernel": "sympow"}, (10.5, 0.1))
    db.post('runtime', {"power": 5, "d": 32, "kernel": "sympow"}, (10.6, 0.2))

    # Query results
    # Find all benchmarks for power=2
    results = db.get(lambda key: key.get("power") == 2)
    assert len(results) == 1
    assert results[0][1] == 10.5

    # Find all benchmarks with dimension 64 and batch size 16
    results = db.get(lambda key: key.get("d") == 64 and key.get("b") == 16)
    assert len(results) == 2
    assert results[0][1] == 10.5
    assert results[1][1] == 15.2

    # Find all benchmarks with layout (2, (2, 1))
    results = db.get(lambda key: key.get("layout") == (2, (2, 3)))
    assert len(results) == 1
    assert results[0][1] == 15.3

    # Test dict value
    results = db.get(lambda key: key.get("layout") == (2, (2, 1)) and key['power'] == 5)
    assert len(results) == 1
    assert results[0][1] == dict(a=1, b=2)

    # Test dict value with value_fn
    results = db.get(lambda key: key.get("layout") == (2, (2, 1)) and key['power'] == 5, value_fn=lambda x: x['a'])
    assert len(results) == 1
    assert results[0][1] == 1

    # Test table query
    results = db.query('runtime', lambda key: key.get("power") == 5)
    assert len(results) == 2

    # Test compare_dict function
    # Test basic dictionary comparison
    assert compare_dict({"a": 1, "b": 2}, {"b": 2, "a": 1})  # Different order
    assert not compare_dict({"a": 1, "b": 2}, {"a": 1, "b": 3})  # Different values
    assert not compare_dict({"a": 1, "b": 2}, {"a": 1})  # Different keys

    # Test nested dictionaries
    assert compare_dict(
        {"a": {"x": 1, "y": 2}, "b": 3},
        {"b": 3, "a": {"y": 2, "x": 1}}
    )
    assert not compare_dict(
        {"a": {"x": 1, "y": 2}, "b": 3},
        {"b": 3, "a": {"x": 1, "y": 3}}
    )

    # Test with lists and tuples
    assert compare_dict(
        {"a": [1, 2, 3], "b": (4, 5)},
        {"b": (4, 5), "a": [1, 2, 3]}
    )
    assert not compare_dict(
        {"a": [1, 2, 3], "b": (4, 5)},
        {"a": [1, 3, 2], "b": (4, 5)}
    )

    # Test with complex nested structures
    assert compare_dict(
        {"a": {"x": [1, 2], "y": (3, 4)}, "b": {"z": 5}},
        {"b": {"z": 5}, "a": {"y": (3, 4), "x": [1, 2]}}
    )
    assert not compare_dict(
        {"a": {"x": [1, 2], "y": (3, 4)}, "b": {"z": 5}},
        {"b": {"z": 5}, "a": {"y": (3, 4), "x": [2, 1]}}
    )

    print("All tests passed!")