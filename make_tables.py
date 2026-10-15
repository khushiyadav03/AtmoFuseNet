import re
import sys
import os

def parse_results(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return {}

    with open(filepath, 'r') as f:
        content = f.read()

    results = {}
    pattern = re.compile(
        r'Evaluation seq (\d+):\s*\n'
        r'MAE \| mean: ([\d\.]+) std: ([\d\.]+)\s*\n'
        r'MAPE \| mean: ([\d\.]+) std: ([\d\.]+)\s*\n'
        r'RMSE \| mean: ([\d\.]+) std: ([\d\.]+)',
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        seq, mae, mae_std, mape, mape_std, rmse, rmse_std = match.groups()
        results[int(seq)] = {
            'mae': float(mae), 'mae_std': float(mae_std),
            'mape': float(mape), 'mape_std': float(mape_std),
            'rmse': float(rmse), 'rmse_std': float(rmse_std)
        }
    return results

def format_latex_rows(results, model_name="AtmoFuseNet"):
    print(f"% --- LaTeX rows for {model_name} by Horizon ---")
    print(r"\hline")
    print(r"Horizon (Hours) & MAE ($\mu \text{g/m}^3$) & MAPE & RMSE ($\mu \text{g/m}^3$) \\")
    print(r"\hline")
    for seq in sorted(results.keys()):
        r = results[seq]
        print(f"Horizon {seq} & {r['mae']:.2f} $\\pm$ {r['mae_std']:.2f} & {r['mape']:.4f} $\\pm$ {r['mape_std']:.4f} & {r['rmse']:.2f} $\\pm$ {r['rmse_std']:.2f} \\\\")
    print(r"\hline")

    print(f"\n% --- LaTeX baseline comparison row for {model_name} (MAE across horizons) ---")
    mae_str = " & ".join([f"{results[s]['mae']:.2f}" if s in results else "\\textcolor{red}{[TODO]}" for s in [2, 4, 8, 16, 24]])
    print(f"{model_name} (Ours) & {mae_str} \\\\")

if __name__ == '__main__':
    target_file = sys.argv[1] if len(sys.argv) > 1 else 'logs/atmofusenet_results.txt'
    res = parse_results(target_file)
    if res:
        format_latex_rows(res)
    else:
        print("No evaluation blocks parsed.")
