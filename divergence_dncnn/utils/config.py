
import argparse

# ---- analyze the parse arguments -----
def analyze_parse(default_sigma):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="DnCNN", help='DnCNN')
    parser.add_argument("--sigma", type=int, default=default_sigma, help="Noise level for the denoising model")
    parser.add_argument("--verbose", type=int, default=1, help="Whether printing the info out")
    opt = parser.parse_args()
    return opt

