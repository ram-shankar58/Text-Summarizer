from utils.preprocessing import preprocess

def test_preprocess():
    text = "This is a test."
    expected_output = ["This", "test", "."]  # This is just an example. Replace with your expected output.

    assert preprocess(text) == expected_output
