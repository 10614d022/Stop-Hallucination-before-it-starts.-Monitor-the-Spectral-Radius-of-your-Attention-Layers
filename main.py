import argparse
import os
from experiments.run_ab import run_experiment
from analysis.summary import summarize
from analysis.plot_survival import plot_survival


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="Paris is the capital of")
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = run_experiment(
        prompt=args.prompt,
        model_name=args.model,
        runs=args.runs,
        max_steps=args.max_steps,
    )

    csv_path = os.path.join(args.out_dir, "runs.csv")
    fig_path = os.path.join(args.out_dir, "survival.png")
    df.to_csv(csv_path, index=False)
    summarize(df)
    plot_survival(df, fig_path)

    print(f"\nSaved runs to: {csv_path}")
    print(f"Saved survival plot to: {fig_path}")


if __name__ == "__main__":
    main()
