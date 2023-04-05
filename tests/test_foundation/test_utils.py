import unittest
from src.foundation.utils import *


class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_convert_to_string(self):
        """Test convert_to_string function.
        """
        state = [1, 2, 3]
        state_str = convert_to_string(state)
        assert state_str == "1,2,3"

    def test_convert_to_list(self):
        """Test convert_to_list function.
        """
        state_str = "1,2,3"
        state = convert_to_list(state_str)
        assert state == [1, 2, 3]

    def test_decay_param(self):
        """Test decay_param function.
        """
        param = 1
        decay = 0.1
        min_param = 0.1
        param = decay_param(param, decay, min_param)
        assert param == 0.9

if __name__ == '__main__':
    unittest.main()