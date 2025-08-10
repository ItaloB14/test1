# Modelagem do tratamento de efluentes por POAs (TiO₂/UV) via RNAs

Aplicativo **Streamlit** para geração de **dados sintéticos** e avaliação de **redes neurais artificiais (RNAs)** no contexto da degradação fotocatalítica do herbicida **2,4-D** por **TiO₂/UV**.

> **Base metodológica**: dados e parâmetros (faixas de pH, TiO₂, tempo; fórmula do modelo de regressão "completo"; uso de ReLU/Tanh/Sigmoid; estratégias de *early stopping*; divisões 60/20/20, 70/15/15, 80/10/10) foram extraídos da qualificação do projeto. Vide referências.

## ✨ Recursos

- Upload de CSV (`pH`, `TiO2_gL`, `tempo_min`, `degradacao`) **ou** geração de **dados sintéticos** por regressão múltipla com **ruído gaussiano controlado**.
- Configuração de faixas para pH, TiO₂ (g/L) e tempo (min).
- Divisão treino/val/teste configurável (60/20/20, 70/15/15, 80/10/10) e escolha de escalonamento (MinMax ou Standard).
- Varredura de arquiteturas **Feedforward (Keras/TensorFlow)** com **1 ou 2 camadas ocultas**, tamanhos de neurônios configuráveis e funções de ativação **ReLU / Tanh / Sigmoid**.
- *Early stopping*, *learning rate*, *batch size* e repetições por *seed* configuráveis.
- Métricas: **MAE, MSE, RMSE, R², R² ajustado, F-statistic (aprox.)** no conjunto de teste.
- **Comparação com baseline** (equação de regressão "completa").
- Gráficos: **observado vs previsto**, **resíduos**, **curvas de aprendizado** e **importância por permutação** das entradas.
- Exporta **melhor modelo** (`artifacts/best_model.h5`), **scaler** (`scaler.joblib`) e **tabela de resultados** (`resultados_experimentos.csv`).

## 🗂 Estrutura

```
.
├── app.py
├── src/
│   ├── constants.py
│   ├── data.py
│   ├── evaluate.py
│   └── models.py
├── sample_data/
│   └── experimentos_2_4D.csv
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

## ▶️ Como executar localmente

1. **Crie e ative um ambiente** (recomendado):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
   ```
2. **Instale dependências**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute o app**:
   ```bash
   streamlit run app.py
   ```
4. Acesse o endereço indicado (ex.: `http://localhost:8501`).

## 🚀 Deploy no Streamlit Cloud

- Faça *push* deste repositório ao GitHub.
- Em **streamlit.io**, crie um novo app apontando para `app.py` na *main branch*.
- Não é necessário configurar Secrets. Tempo de build ~ 5–10 min (por conta do TensorFlow).

## 📄 Formato esperado do CSV

- **Colunas**: `pH`, `TiO2_gL`, `tempo_min` e opcionalmente `degradacao`.
- Se `degradacao` não estiver presente, **gere sintético** no app para obter a variável alvo.

Veja um **exemplo** em `sample_data/experimentos_2_4D.csv`.

## 🧪 Dados sintéticos (regressão "completa")

A predição **D(%)** é calculada por:

```
D(%) = 141.1222 − 10,2937·pH + 17,0625·TiO₂ − 1,1964·tempo
       − 2,5625·(pH·TiO₂) + 0,1314·(pH·tempo) + 2,3141·(TiO₂·tempo)
       − 0,2641·(pH·TiO₂·tempo)
```

Em seguida, é adicionado **ruído gaussiano** (desvio padrão = ±`noise_pct`% do valor esperado), com **pós-processamento** (clip 0–100 e remoção de *outliers* pelo IQR).

## 📚 Referências (qualificação)

- **Faixas experimentais** (pH 5–9; TiO₂ 0,10–0,50 g/L; tempo 40–80 min) — Tabela de níveis do delineamento fatorial (p. 30).
- **Equação do modelo de regressão "completo"** — Equação (8) (p. 43).
- **Resultados estatísticos dos modelos** e escolha do "completo" — Tabelas e discussão (pp. 41–42).
- **Geração de dados sintéticos** (distribuição homogênea e ruído ±5%) — Seção 5.4 (pp. 35–36).
- **Arquiteturas e parâmetros das RNAs** (ReLU/Tanh/Sigmoid; 1–2 camadas; 10–120 neurônios; *early stopping*; divisões 60/20/20, 70/15/15, 80/10/10) — Seções 5.5–5.6 e Tabela 7 (pp. 37–40).

> Estes pontos são os utilizados pelo app para reproduzir e expandir os experimentos.

## ⚠️ Observações

- O app usa **TensorFlow/Keras (CPU)** e pode levar alguns minutos para treinar arquiteturas maiores.
- Resultados dependem dos dados fornecidos e das escolhas de *hyperparameters*.
- A importância por permutação é uma **aproximação** baseada na queda do R² ao embaralhar cada entrada.

## 📜 Licença

MIT — sinta-se livre para adaptar para sua dissertação, citando a fonte dos dados experimentais quando utilizar dados reais.

---

**Autor:** Seu Nome
