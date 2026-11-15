# 4μ — Automated Statistical Reporting

An exploratory analytics application that turns a CSV dataset and a plain-language
question into statistical summaries, visualizations, and a downloadable report.
It also lets users ask questions about PDF reports through a retrieval-augmented
generation workflow.

The project connects three parts of an analyst's work: preparing and investigating
data, interpreting the findings in context, and communicating them to a reader.
Python computes the statistics; language models help map the user's question to
the data and turn the resulting evidence into a narrative.

[Analysis engine](main.py) · [Web interface](app/website.py) ·
[Example generated report](app/reports%20generated/revenue_analysis.pdf) ·
[Repository overview](../README.md)

## What the application does

A user uploads a dataset, describes an objective, and explores the results through
a Gradio interface. For example, a retail analyst might ask which customer profiles
account for different purchasing patterns. The application produces descriptive
statistics and cluster profiles, applies the available domain-specific analyses,
and generates a written interpretation with supporting charts.

The interface exposes numerical and categorical summaries, distributions,
correlations, cluster profiles, categorical association tests, and domain results.
The report can be exported to PDF. A separate report-questioning interface accepts
PDFs and retrieves relevant excerpts before requesting an answer from the LLM.

## Statistical methods and their interpretation

| Component | Technique | What the result helps a reader understand |
|---|---|---|
| Data preparation | Variable-type inference, missing-value rules, iterative numeric imputation, median/mode filling | Which variables can be analyzed and how incomplete observations are treated |
| Exploratory analysis | Means, standard deviations, quantiles, category frequencies, date ranges, Pearson correlations | Typical values, dispersion, imbalance, and relationships in the selected data |
| Numerical segmentation | StandardScaler followed by K-Means; cluster count compared with silhouette scores | Groups of observations with similar numerical profiles, summarized by feature means and group size |
| Encoded categorical/ordinal segmentation | Custom K-Medoids with Manhattan distance and silhouette-based model selection | Representative observed profiles, rather than synthetic centroids |
| Categorical associations | Pairwise chi-square tests of independence | Category combinations that may merit further investigation |
| Outlier diagnostics | Regression residual diagnostics and Cook's distance; IQR fallback with one numeric variable | Observations that are unusual relative to a fitted relationship or influential on that relationship |
| Domain analysis | Dedicated modules, including retail, e-commerce, HR, and insurance | Business-oriented summaries and analyses when the required variables are available |

The distinction between an unusual observation and an influential one is useful:
a point can be far from the fitted relationship without substantially changing the
fit, or it can strongly affect the estimated relationship. The workflow keeps both
the cleaned sample and flagged observations and can generate a separate analysis
when the flagged subset contains more than 30 rows.

Clusters describe similarity, not automatically meaningful business categories.
Categorical encoding affects Manhattan distances, so encoded category order needs
review. Chi-square results are exploratory associations, with no implemented
multiple-testing adjustment. Correlations, clusters, and automatically generated
interpretations do not establish causal effects.

## How the pipeline is organized

```text
CSV + analysis objective
        ↓
Data preparation and variable identification
        ↓
Outlier/influence diagnostics and descriptive statistics
        ↓
Segmentation, association tests, and available domain analysis
        ↓
LLM interpretation + charts
        ↓
Interactive results and PDF report
```

`main.py` coordinates the analysis, while statistical functions and domain modules
remain separate from the Gradio presentation. This organization makes it possible
to inspect the computations behind a written report instead of treating the
narrative as the only output.

The LLM layer uses Groq-hosted models to identify the dataset's domain, infer column
roles, interpret the analysis objective and its language, and write report text.
The narrative is conditioned on computed statistics and domain outputs. Column
mapping and interpretation remain model-generated decisions that require review.

## Asking questions about reports: RAG

The PDF workflow in [LLM/RAG.py](LLM/RAG.py) combines document processing, semantic
retrieval, and answer generation:

1. Extract text from PDF pages with `pdfplumber`.
2. Split the text into overlapping chunks: 800 characters with 150-character overlap.
3. Store chunks and source metadata in an in-memory ChromaDB collection, using its
   default embedding function and cosine similarity.
4. Retrieve the six most relevant passages for a question by default.
5. Supply the passages and their source filenames to the Groq model, with
   instructions to answer from the retrieved context and acknowledge missing information.

This supports questions such as “What did the report identify as the main source
of variation?” or comparisons between uploaded reports. Retrieval makes the context
inspectable, but it does not independently verify a report's statistics or guarantee
that the generated answer is correct. Text extraction also depends on the PDF
containing extractable text; no OCR stage is implemented.

## Run the prototype

From the root of this repository, enter the project directory and install the
packages used by the application:

```sh
cd auto_reporting
python3 -m pip install numpy pandas scipy scikit-learn matplotlib seaborn gradio reportlab requests chromadb pdfplumber diffprivlib pillow
```

These are the source-level dependencies; the project does not currently provide a
pinned environment or lockfile. The optional partial-order experiment uses the
included `pyLattice.py` module and is separate from the default reporting workflow.

Set a Groq API key in the terminal environment, then launch the web interface:

```sh
export GROQ_API_KEY="your_groq_api_key"
python3 app/website.py
```

Open the Gradio address printed in the terminal. Upload a CSV, enter an objective,
click **Analyze Dataset**, and inspect the result sections. Use **Generate and
Download Report** for the PDF. The **Report Intelligence** section accepts PDFs for
retrieval-based questions.

Run from the `auto_reporting` directory because some resources, including
`knowledge_base.txt`, use relative paths. [`.env.example`](.env.example) documents
the key name, but the application reads the environment directly; creating a `.env`
file alone does not load it. Keep the real key outside version control.

The current launcher calls `iface.launch(share=True)`, which requests a public
Gradio share link. Set `share=False` in `app/website.py` for local-only use. The LLM
workflow sends selected statistics, dataset samples, or retrieved report excerpts
to Groq, so this prototype is best demonstrated with public or synthetic inputs.
The embedding model may also require an initial download.

## Current scope and limitations

This is a research prototype with components at different stages of
maturity. Its scope is important when interpreting a demonstration:

- **Dataset selection:** `run_analysis` currently selects at most the first 500
  rows and columns at zero-based positions 2 through 7. Results describe that
  subset, not necessarily the whole uploaded file.
- **Data preparation:** missingness rules are heuristic. They should not be read
  as a validated identification of the MCAR/MAR mechanism. Some downstream paths
  assume that usable numerical variables remain after selection and cleaning.
- **Domain coverage:** the project contains both developed analysis modules and
  placeholders. The presence of a domain name does not guarantee a complete
  analysis for every dataset in that domain.
- **Statistical interpretation:** outlier removal depends on a fitted relationship
  and can remove meaningful rare cases. Comparing cleaned and flagged samples is
  part of interpreting the result.
- **Experimental privacy:** a `diffprivlib` Laplace mechanism perturbs selected
  numerical domain results. The current application does not establish an
  end-to-end differential privacy guarantee; other summaries and samples also
  enter the LLM workflow.
- **Execution and persistence:** analysis state, generated plots, and PDF filenames
  are shared within the prototype. It is not designed as an isolated multi-user
  reporting service, and repeated runs can replace generated files.

## What this project demonstrates

The main contribution is assembling an analytical workflow whose numerical
outputs and written interpretation can be inspected together. It brings together
statistical preprocessing, unsupervised learning, influence diagnostics, semantic
column mapping, document retrieval, and report generation in one application.

For data science work, the interesting decisions are how to choose analyses from
a heterogeneous table, preserve the context of flagged observations, and distinguish
computed evidence from model-generated interpretation. For applied ML work, the
RAG component shows the path from document extraction and chunking to retrieval
and a contextual answer. The interface and PDF export make those components
accessible beyond a notebook or terminal session.

## Code map

| Path | Responsibility |
|---|---|
| [`main.py`](main.py) | Orchestrate preparation, diagnostics, statistical analysis, and narrative generation |
| [`app/website.py`](app/website.py) | Gradio interface, result views, PDF formatting, and report-questioning controls |
| [`data_manipulation/`](data_manipulation/) | CSV loading, missing-data treatment, type inference, experimental privacy |
| [`analystics/`](analystics/) | Descriptive statistics and domain-specific analysis modules |
| [`statistics_functions/`](statistics_functions/) | Clustering, association tests, outlier diagnostics, and plots |
| [`LLM/llm_output.py`](LLM/llm_output.py) | Domain recognition, objective interpretation, and semantic column mapping |
| [`LLM/report.py`](LLM/report.py) | Generate the analytical report narrative |
| [`LLM/RAG.py`](LLM/RAG.py) | Extract, index, retrieve, and query PDF report text |
| [`data/`](data/) | Example tabular datasets for exploration |
| [`app/reports generated/`](app/reports%20generated/) | Checked-in example reports illustrating the presentation layer |
| [`testing_codes/`](testing_codes/) | Exploratory checks for selected components |

The scripts in `testing_codes/` are component experiments rather than a comprehensive
automated test suite for the full reporting application.
