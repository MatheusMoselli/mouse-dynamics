#!/usr/bin/env python3
"""
Consolida os resultados de execução (JSONs) do pipeline de classificação
de dinâmica de mouse.

Cada arquivo JSON segue o padrão de nome:
    {id}_{classificador}_{window_size}_{seed}.json

e vive em outputs/<dataset>/. Para cada combinação
(dataset, classificador, window_size), existem (idealmente) 5 arquivos,
um por seed (001..005). Este script agrupa esses arquivos e calcula,
para cada métrica de interesse, a média, o desvio padrão e outras
estatísticas ao longo das seeds.

Uso:
    python consolidar_resultados.py --root /caminho/para/outputs --out /caminho/para/saida

Se --root não for informado, assume "outputs" no diretório atual.
Se --out não for informado, assume "consolidado" no diretório atual.

Saídas geradas em --out:
    resumo_por_config.csv   -> uma linha por (dataset, classificador, window_size)
                                com média/std/min/max/mediana das métricas
                                agregadas por run (mean_score, mean_f1_macro, etc.)
    resumo_por_config.json  -> mesma informação, em JSON
    resumo_por_usuario.csv  -> uma linha por (dataset, classificador, window_size, user_id)
                                com média/std das métricas por usuário ao longo das seeds
                                (útil para ver variância individual, comum em biometria
                                comportamental)
    log_inconsistencias.txt -> avisos sobre seeds faltando, duplicadas ou arquivos
                                com nome fora do padrão
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FILENAME_RE = re.compile(
    r"^(?P<run_id>[^_]+)_(?P<classifier>.+)_wind(?P<window>\d+)_(?P<seed>\d+)\.json$"
)

# Métricas agregadas de cada run (já vêm prontas no JSON, calculadas sobre
# os usuários daquela execução). É sobre essas que fazemos média/std entre
# as seeds.
RUN_METRICS = [
    "mean_score",
    "mean_balanced_score",
    "mean_f1_macro",
    "mean_f1_weighted",
    "mean_precision_macro",
    "mean_recall_macro",
]

# Métricas por usuário, presentes em cada item de "user_results".
USER_METRICS = [
    "score",
    "balanced_score",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
]

EXPECTED_SEEDS = {"001", "002", "003", "004", "005"}


def stats_dict(values, prefix):
    """Recebe uma lista de floats e devolve um dicionário de estatísticas
    com o prefixo dado (ex.: prefix='mean_score' -> mean_score_mean, ...)."""
    n = len(values)
    out = {
        f"{prefix}_n": n,
        f"{prefix}_mean": statistics.fmean(values) if n else float("nan"),
        f"{prefix}_std": statistics.stdev(values) if n > 1 else 0.0,
        f"{prefix}_min": min(values) if n else float("nan"),
        f"{prefix}_max": max(values) if n else float("nan"),
        f"{prefix}_median": statistics.median(values) if n else float("nan"),
    }
    if out[f"{prefix}_mean"]:
        cv = (out[f"{prefix}_std"] / out[f"{prefix}_mean"]) if out[f"{prefix}_mean"] != 0 else float("nan")
        out[f"{prefix}_cv"] = cv  # coeficiente de variação (std/mean)
    return out


def collect_runs(root: Path, log_lines: list):
    """Percorre root/<dataset>/*.json e retorna um dict:
    (dataset, classifier, window) -> {seed: run_dict}
    """
    groups = defaultdict(dict)

    dataset_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dataset_dirs:
        log_lines.append(f"[AVISO] Nenhuma subpasta de dataset encontrada em {root}")

    for dataset_dir in sorted(dataset_dirs):
        dataset = dataset_dir.name
        for fp in sorted(dataset_dir.glob("*.json")):
            m = FILENAME_RE.match(fp.name)
            if not m:
                log_lines.append(f"[AVISO] Nome fora do padrão, ignorado: {fp}")
                continue

            seed = m.group("seed")
            window = m.group("window")

            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                log_lines.append(f"[ERRO] Falha ao ler {fp}: {e}")
                continue

            classifier = data.get("classifier", m.group("classifier"))
            key = (dataset, classifier, window)

            if seed in groups[key]:
                prev = groups[key][seed]
                prev_ts = prev.get("timestamp", "")
                new_ts = data.get("timestamp", "")
                # Mantém o run mais recente por timestamp; avisa sobre o descarte.
                if new_ts >= prev_ts:
                    log_lines.append(
                        f"[AVISO] Seed duplicada para {key} seed={seed}: "
                        f"mantendo {fp.name} (timestamp {new_ts}), "
                        f"descartando run anterior (timestamp {prev_ts})"
                    )
                    groups[key][seed] = data
                else:
                    log_lines.append(
                        f"[AVISO] Seed duplicada para {key} seed={seed}: "
                        f"mantendo run anterior (timestamp {prev_ts}), "
                        f"descartando {fp.name} (timestamp {new_ts})"
                    )
            else:
                groups[key][seed] = data

    return groups


def check_missing_seeds(groups, log_lines):
    for key, seeds_dict in sorted(groups.items()):
        found = set(seeds_dict.keys())
        missing = EXPECTED_SEEDS - found
        extra = found - EXPECTED_SEEDS
        if missing:
            log_lines.append(f"[AVISO] {key}: faltando seeds {sorted(missing)} (encontradas: {sorted(found)})")
        if extra:
            log_lines.append(f"[AVISO] {key}: seeds inesperadas {sorted(extra)}")


def build_config_summary(groups):
    rows = []
    for (dataset, classifier, window), seeds_dict in sorted(groups.items()):
        runs = list(seeds_dict.values())
        row = {
            "dataset": dataset,
            "classifier": classifier,
            "window_size": int(window),
            "n_seeds": len(runs),
            "seeds_usadas": ",".join(sorted(seeds_dict.keys())),
        }

        # n_users e n_users_skipped: reporta média/consistência
        n_users_vals = [r.get("n_users") for r in runs if r.get("n_users") is not None]
        n_skip_vals = [r.get("n_users_skipped") for r in runs if r.get("n_users_skipped") is not None]
        if n_users_vals:
            row["n_users_mean"] = statistics.fmean(n_users_vals)
            row["n_users_consistente"] = len(set(n_users_vals)) == 1
        if n_skip_vals:
            row["n_users_skipped_mean"] = statistics.fmean(n_skip_vals)

        for metric in RUN_METRICS:
            values = [r[metric] for r in runs if metric in r and r[metric] is not None]
            row.update(stats_dict(values, metric))

        rows.append(row)
    return rows


def build_user_summary(groups):
    rows = []
    for (dataset, classifier, window), seeds_dict in sorted(groups.items()):
        per_user = defaultdict(lambda: defaultdict(list))
        for seed, run in sorted(seeds_dict.items()):
            for user_result in run.get("user_results", []):
                uid = user_result.get("user_id")
                for metric in USER_METRICS:
                    if metric in user_result and user_result[metric] is not None:
                        per_user[uid][metric].append(user_result[metric])

        for uid, metric_values in sorted(per_user.items(), key=lambda kv: str(kv[0])):
            row = {
                "dataset": dataset,
                "classifier": classifier,
                "window_size": int(window),
                "user_id": uid,
            }
            for metric in USER_METRICS:
                values = metric_values.get(metric, [])
                row.update(stats_dict(values, metric))
            rows.append(row)
    return rows


def write_csv(rows, path: Path):
    if not rows:
        path.write_text("")
        return
    # União ordenada das chaves (algumas linhas podem ter colunas extras/faltando)
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Consolida resultados de múltiplas seeds em estatísticas agregadas.")
    parser.add_argument("--root", type=str, default="outputs", help="Pasta raiz contendo uma subpasta por dataset (default: ./outputs)")
    parser.add_argument("--out", type=str, default="consolidado", help="Pasta de saída para os arquivos consolidados (default: ./consolidado)")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"[ERRO] Pasta raiz não encontrada: {root}", file=sys.stderr)
        sys.exit(1)

    log_lines = []

    groups = collect_runs(root, log_lines)
    check_missing_seeds(groups, log_lines)

    config_rows = build_config_summary(groups)
    user_rows = build_user_summary(groups)

    write_csv(config_rows, out_dir / "resumo_por_config.csv")
    write_csv(user_rows, out_dir / "resumo_por_usuario.csv")

    with open(out_dir / "resumo_por_config.json", "w", encoding="utf-8") as f:
        json.dump(config_rows, f, indent=2, ensure_ascii=False)

    with open(out_dir / "log_inconsistencias.txt", "w", encoding="utf-8") as f:
        if log_lines:
            f.write("\n".join(log_lines) + "\n")
        else:
            f.write("Nenhuma inconsistência encontrada.\n")

    print(f"Combinações (dataset x classificador x window_size) processadas: {len(groups)}")
    print(f"Linhas em resumo_por_config.csv: {len(config_rows)}")
    print(f"Linhas em resumo_por_usuario.csv: {len(user_rows)}")
    if log_lines:
        print(f"Avisos encontrados: {len(log_lines)} (ver {out_dir / 'log_inconsistencias.txt'})")
    print(f"Saída escrita em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()