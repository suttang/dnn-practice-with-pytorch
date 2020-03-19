import os

import unittest

from datasets.utils import get_cache_dir


class TestGetCacheDir(unittest.TestCase):
    def test_cache_dir(self):
        expected = os.path.join(os.path.dirname(__file__), ".cache")
        self.assertTrue(get_cache_dir(), expected)


if __name__ == "__main__":
    unittest.main()
