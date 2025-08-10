import os
import io
import json
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import (
    load_csv, synthesize, merge_datasets, compute_baseline_predictions,
    REQUIRED_COLUMNS, TARGET_DEFAULT
)
from src.evaluate import make_splits, grid_from_user
from src.models import build_mlp, compile_and_train, evaluate_model, permutation_importance, set_all_seeds
from src.constants import RANGES_DEFAULT

st.set_page_config(page_title="Modelagem POAs via RNA (2,4-D)", layout="wide")

st.title("Modelagem do tratamento de efluentes por POAs (TiO₂/UV) com RNAs")
st.caption("Ferramenta interativa para geração de dados sintéticos, treino e comparação de arquiteturas de redes neurais (Keras/TensorFlow).")

with st.expander("ℹ️ Sobre", expanded=False):
    st.markdown("""
Este aplicativo segue a metodologia descrita na **qualificação**: regressão múltipla para gerar **dados sintéticos** e, em seguida, avaliação de diversas **arquiteturas de RNAs** com funções de ativação ReLU/Tanh/Sigmoid, uma ou duas camadas ocultas, e divisões **60/20/20**, **70/15/15** e **80/10/10**.
    
**Entradas** esperadas: `pH`, `TiO2_gL` (g/L), `tempo_min` (min). **Saída** (alvo): `degradacao` (%).
    """)

# =============== SIDEBAR CONFIG ===============
st.sidebar.header("1) Dados")
mode = st.sidebar.selectbox("Fonte dos dados", ["Carregar CSV", "Gerar sintético", "CSV + Sintético"])

uploaded = None
df_loaded = None
if mode in ["Carregar CSV", "CSV + Sintético"]:
    uploaded = st.sidebar.file_uploader("Envie um CSV com colunas: pH, TiO2_gL, tempo_min, degradacao (opcional)", type=["csv"])
    if uploaded:
        try:
            bundle = load_csv(uploaded)
            df_loaded = bundle.df.copy()
            st.sidebar.success(f"CSV carregado com {len(df_loaded)} linhas.")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar CSV: {e}")

st.sidebar.subheader("Configuração dos dados sintéticos")
n_samples = st.sidebar.number_input("Amostras sintéticas (n)", min_value=100, max_value=100000, value=1000, step=100)
noise_pct = st.sidebar.slider("Ruído gaussiano (± % do valor)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

c1, c2 = st.sidebar.columns(2)
pH_min = c1.number_input("pH min", value=float(RANGES_DEFAULT["pH"][0]))
pH_max = c2.number_input("pH max", value=float(RANGES_DEFAULT["pH"][1]))
c3, c4 = st.sidebar.columns(2)
Ti_min = c3.number_input("TiO₂ (g/L) min", value=float(RANGES_DEFAULT["TiO2_gL"][0]), format="%.2f")
Ti_max = c4.number_input("TiO₂ (g/L) max", value=float(RANGES_DEFAULT["TiO2_gL"][1]), format="%.2f")
c5, c6 = st.sidebar.columns(2)
t_min = c5.number_input("Tempo (min) min", value=float(RANGES_DEFAULT["tempo_min"][0]))
t_max = c6.number_input("Tempo (min) max", value=float(RANGES_DEFAULT["tempo_min"][1]))

st.sidebar.header("2) Pré-processamento")
split_choice = st.sidebar.selectbox("Divisão treino/val/teste", ["60/20/20", "70/15/15", "80/10/10"])
scaler_choice = st.sidebar.selectbox("Escalonamento", ["minmax", "standard"])

st.sidebar.header("3) Configuração das RNAs")
n_layers = st.sidebar.radio("Camadas ocultas", [1, 2], index=0, horizontal=True)
neurons_input = st.sidebar.text_input("Opções de neurônios (separadas por vírgula)", value="10,30,60,120")
activations = st.sidebar.multiselect("Funções de ativação", ["relu", "tanh", "sigmoid"], default=["relu", "tanh", "sigmoid"])
epochs = st.sidebar.number_input("Épocas (máx.)", min_value=10, max_value=5000, value=600, step=10)
batch_size = st.sidebar.number_input("Batch size", min_value=8, max_value=1024, value=32, step=8)
patience = st.sidebar.number_input("Early stopping (paciência)", min_value=5, max_value=200, value=50, step=5)
learning_rate = float(st.sidebar.text_input("Learning rate", value="0.001"))
n_runs = st.sidebar.number_input("Repetições por configuração (seeds)", min_value=1, max_value=10, value=2, step=1)
seed_base = st.sidebar.number_input("Seed base", min_value=0, max_value=1000000, value=42, step=1)

run_button = st.sidebar.button("🚀 Executar Experimentos", use_container_width=True)

# =============== DATASET CREATION ===============
ranges = dict(pH=(pH_min, pH_max), TiO2_gL=(Ti_min, Ti_max), tempo_min=(t_min, t_max))

df_synth = None
if mode in ["Gerar sintético", "CSV + Sintético"]:
    df_synth = synthesize(n_samples=n_samples, ranges=ranges, noise_pct=noise_pct, clip_01=True, seed=int(seed_base))
    st.success(f"Geradas {len(df_synth)} linhas sintéticas.")

df_final = None
if mode == "Carregar CSV" and df_loaded is not None:
    df_final = df_loaded.copy()
elif mode == "Gerar sintético":
    df_final = df_synth.copy()
elif mode == "CSV + Sintético" and (df_loaded is not None) and (df_synth is not None):
    df_final = merge_datasets(df_loaded, df_synth)

if df_final is not None:
    st.subheader("📦 Conjunto de dados (amostra)")
    st.dataframe(df_final.head(20), use_container_width=True)
    with st.expander("Estatísticas descritivas"):
        st.write(df_final.describe().T)

# =============== EXPERIMENTS ===============
if run_button:
    if df_final is None:
        st.error("Carregue ou gere dados primeiro.")
        st.stop()

    if "degradacao" not in df_final.columns:
        st.error("A coluna alvo 'degradacao' não está presente. Gere dados sintéticos ou forneça o alvo no CSV.")
        st.stop()

    # Prepare splits
    X = df_final[["pH", "TiO2_gL", "tempo_min"]].values.astype("float32")
    y = df_final["degradacao"].values.astype("float32")
    split_map = {"60/20/20": (0.6, 0.2, 0.2), "70/15/15": (0.7, 0.15, 0.15), "80/10/10": (0.8, 0.1, 0.1)}
    splits = make_splits(X, y, split_tuple=split_map[split_choice], scaler_type=scaler_choice, seed=int(seed_base))

    neurons = [int(x.strip()) for x in neurons_input.split(",") if x.strip().isdigit()]
    grid = grid_from_user(neurons, activations, n_hidden_layers=int(n_layers))

    results_rows = []
    best_key = None
    best_rmse = float("inf")
    best_artifacts: Dict[str, Any] = {}

    progress = st.progress(0.0, text="Iniciando...")
    total_iters = max(1, len(grid) * n_runs)
    done = 0

    for cfg in grid:
        for run_idx in range(n_runs):
            seed = int(seed_base) + run_idx
            set_all_seeds(seed)
            model = build_mlp(input_dim=X.shape[1], hidden_layers=cfg["hidden_layers"], activation=cfg["activation"])
            train_info = compile_and_train(
                model,
                splits.X_train, splits.y_train,
                splits.X_val, splits.y_val,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                patience=int(patience),
                verbose=0
            )
            metrics_test = evaluate_model(model, splits.X_test, splits.y_test)

            row = {
                "run": run_idx,
                "layers": len(cfg["hidden_layers"]),
                "hidden": "-".join(map(str, cfg["hidden_layers"])),
                "activation": cfg["activation"],
                "epochs_trained": len(train_info["history"].history["loss"]),
                "MAE": metrics_test["MAE"],
                "MSE": metrics_test["MSE"],
                "RMSE": metrics_test["RMSE"],
                "R2": metrics_test["R2"],
                "R2_adj": metrics_test["R2_adj"],
                "F": metrics_test["F"],
            }
            results_rows.append(row)

            if metrics_test["RMSE"] < best_rmse:
                best_rmse = metrics_test["RMSE"]
                best_key = (cfg["hidden_layers"], cfg["activation"], run_idx)
                # Save artifacts for the best model
                y_pred_test = model.predict(splits.X_test, verbose=0).reshape(-1)
                best_artifacts = dict(
                    model=model,
                    history=train_info["history"].history,
                    y_test=splits.y_test.copy(),
                    y_pred=y_pred_test.copy(),
                    X_test=splits.X_test.copy(),
                    scaler=splits.scaler
                )

            done += 1
            progress.progress(done / total_iters, text=f"Rodando... ({done}/{total_iters})")

    results_df = pd.DataFrame(results_rows).sort_values(by=["RMSE", "MSE"]).reset_index(drop=True)
    st.subheader("📊 Resultados por configuração")
    st.dataframe(results_df, use_container_width=True)

    # Baseline: regressão completa
    y_base = compute_baseline_predictions(pd.DataFrame(splits.X_test, columns=["pH","TiO2_gL","tempo_min"]))
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae_b = mean_absolute_error(splits.y_test, y_base)
    mse_b = mean_squared_error(splits.y_test, y_base)
    rmse_b = np.sqrt(mse_b)
    r2_b = r2_score(splits.y_test, y_base)
    r2a_b = 1 - (1 - r2_b) * ((len(splits.y_test) - 1) / (len(splits.y_test) - X.shape[1] - 1))

    st.markdown("### 📌 Comparação com baseline (equação de regressão completa)")
    st.write(pd.DataFrame([{"Modelo":"Baseline (Regressão)","MAE":mae_b,"MSE":mse_b,"RMSE":rmse_b,"R2":r2_b,"R2_adj":r2a_b}]))

    # Best model plots
    if best_artifacts:
        st.markdown("### 🏆 Melhor RNA (menor RMSE no teste)")
        st.write(f"Configuração: hidden={best_key[0]}, activation='{best_key[1]}', run={best_key[2]}")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.scatter(best_artifacts["y_test"], best_artifacts["y_pred"], alpha=0.6)
            lims = [min(best_artifacts["y_test"].min(), best_artifacts["y_pred"].min()),
                    max(best_artifacts["y_test"].max(), best_artifacts["y_pred"].max())]
            ax1.plot(lims, lims, "--")
            ax1.set_xlabel("Observado (%)")
            ax1.set_ylabel("Previsto (%)")
            ax1.set_title("Dispersão: Observado vs. Previsto (teste)")
            st.pyplot(fig1, use_container_width=True)
        with col2:
            fig2, ax2 = plt.subplots()
            residuals = best_artifacts["y_test"] - best_artifacts["y_pred"]
            ax2.hist(residuals, bins=30)
            ax2.set_title("Distribuição dos resíduos (teste)")
            st.pyplot(fig2, use_container_width=True)

        # Learning curves
        hist = best_artifacts["history"]
        fig3, ax3 = plt.subplots()
        ax3.plot(hist["loss"], label="treino")
        if "val_loss" in hist:
            ax3.plot(hist["val_loss"], label="val")
        ax3.set_title("Curvas de aprendizado (MSE)")
        ax3.set_xlabel("Época")
        ax3.set_ylabel("MSE")
        ax3.legend()
        st.pyplot(fig3, use_container_width=True)

        # Permutation importance (on test set)
        st.markdown("#### Importância (permutação) nas entradas (teste)")
        imps, order = permutation_importance(best_artifacts["model"], best_artifacts["X_test"], best_artifacts["y_test"], n_repeats=10, random_state=int(seed_base))
        feat_names = ["pH","TiO2_gL","tempo_min"]
        imp_df = pd.DataFrame({"feature": feat_names, "importance_R2_drop": imps})
        imp_df = imp_df.sort_values(by="importance_R2_drop", ascending=False).reset_index(drop=True)
        st.dataframe(imp_df)

        # Save artifacts
        export_dir = "artifacts"
        os.makedirs(export_dir, exist_ok=True)
        # Save model
        best_artifacts["model"].save(os.path.join(export_dir, "best_model.h5"))
        # Save scaler
        import joblib
        joblib.dump(best_artifacts["scaler"], os.path.join(export_dir, "scaler.joblib"))
        # Save results table
        results_df.to_csv(os.path.join(export_dir, "resultados_experimentos.csv"), index=False)
        st.success("Artefatos salvos em ./artifacts (modelo, scaler, resultados). Baixe estes arquivos ao subir ao GitHub.")

    # Download results
    st.download_button(
        "⬇️ Baixar tabela de resultados (CSV)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="resultados_rna.csv",
        mime="text/csv",
        use_container_width=True,
    )
