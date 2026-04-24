import argparse
from aftab import Aftab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Aftab running seed"
    )
    parser.add_argument(
        "--environment",
        type=str,
        required=True,
        help="Environment name"
    )
    args = parser.parse_args()
    agent = Aftab()
    agent.train(environment=args.environment, seed=args.seed)
    agent.log(directory="results")

if __name__ == "__main__":
    main()