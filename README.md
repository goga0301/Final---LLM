# Multi-LLM Collaborative Debate System

A sophisticated problem-solving system where multiple Large Language Models (LLMs) collaborate through structured debate to produce high-quality answers. The system leverages diverse AI perspectives and adversarial review to combat hallucination and improve accuracy.

## 🎯 Overview

This system implements a multi-stage debate workflow:

1. **Role Assignment (Stage 0/0.5)**: Four LLMs self-assess their capabilities and are algorithmically assigned roles
2. **Independent Solutions (Stage 1)**: Three Solvers generate solutions independently with detailed reasoning
3. **Peer Review (Stage 2)**: Each Solver critically evaluates the other two solutions
4. **Refinement (Stage 3)**: Solvers refine their solutions based on peer feedback
5. **Final Judgment (Stage 4)**: A Judge LLM evaluates all refined solutions and selects the best answer

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Problem Input                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 0: Role Self-Assessment                       │
│         GPT │ Claude │ Gemini │ Grok                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 0.5: Algorithmic Role Assignment              │
│         → 3 Solvers + 1 Judge                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 1: Independent Solutions                      │
│         Solver 1 │ Solver 2 │ Solver 3                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 2: Peer Review                                │
│         Each solver reviews 2 peer solutions                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 3: Refinement                                 │
│         Address critiques → Improved solutions                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 4: Final Judgment                             │
│         Judge selects best answer                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Final Answer                             │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Final/
├── config/
│   ├── __init__.py
│   └── config.py              # API keys and model settings
├── data/
│   └── problems.json          # 25-problem dataset
├── src/
│   ├── __init__.py
│   ├── llm_clients/           # LLM API client implementations
│   │   ├── base_client.py     # Abstract base class
│   │   ├── openai_client.py   # GPT
│   │   ├── anthropic_client.py # Claude
│   │   ├── google_client.py   # Gemini
│   │   └── xai_client.py      # Grok
│   ├── stages/                # Debate stage implementations
│   │   ├── role_assignment.py # Stage 0 & 0.5
│   │   ├── solver.py          # Stage 1
│   │   ├── peer_review.py     # Stage 2
│   │   ├── refinement.py      # Stage 3
│   │   └── judge.py           # Stage 4
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── evaluation/
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── baselines.py       # Baseline comparisons
│   └── orchestrator.py        # Main workflow coordinator
├── visualization/
│   └── plots.py               # Matplotlib/Seaborn visualizations
├── results/                   # Output directory
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Final
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
XAI_API_KEY=your_xai_api_key
```

## 📊 Usage

### Run Full Pipeline
```bash
python main.py --full
```
This runs the complete pipeline: debate system, baselines, evaluation, and visualization.

### Run Individual Components
```bash
# Run debate system only
python main.py --run-debate

# Run baseline comparisons
python main.py --run-baselines

# Evaluate existing results
python main.py --evaluate

# Generate plots from existing results
python main.py --generate-plots

# Check API key configuration
python main.py --check-keys
```

### Additional Options
```bash
# Limit number of problems (useful for testing)
python main.py --full --limit 5

# Use custom problems file
python main.py --full --problems-file path/to/problems.json
```

## 📈 Evaluation Metrics

The system tracks the following metrics:

- **Overall Accuracy**: Percentage of problems solved correctly
- **Improvement Rate**: Problems where refinement improved initial answers
- **Consensus Rate**: Problems where all 3 Solvers agreed
- **Judge Accuracy**: Correct selections when Solvers disagreed
- **Per-Category Accuracy**: Breakdown by problem type
- **Model Performance**: Individual model statistics by role

### Baseline Comparisons

- **Single-LLM Baseline**: Each model asked once independently
- **Simple Voting Baseline**: 3 models vote, majority wins
- **Full Debate System**: Complete multi-stage workflow

## 📋 Problem Dataset

The dataset includes 25 challenging problems across 4 categories:

| Category | Count | Description |
|----------|-------|-------------|
| Mathematical/Logical | 7 | Combinatorics, probability, number theory |
| Physics/Scientific | 6 | Multi-step physics, counterintuitive scenarios |
| Logic Puzzles | 6 | Knights/knaves, constraint satisfaction |
| Game Theory | 6 | Auctions, Nash equilibria, backward induction |

## 📊 Generated Visualizations

The system generates the following plots in `results/plots/`:

1. **accuracy_by_category.png**: Bar chart of accuracy by problem category
2. **system_vs_baselines.png**: Comparison chart vs. baseline methods
3. **model_performance_heatmap.png**: Heatmap of model performance by role
4. **improvement_through_stages.png**: Line chart showing accuracy improvement
5. **judge_confusion_matrix.png**: Judge decision analysis
6. **consensus_analysis.png**: Consensus vs. disagreement outcomes

## 🔧 Configuration

### Model Settings (config/config.py)

```python
# Adjust model parameters
OPENAI_CONFIG = ModelConfig(
    name="GPT",
    model_id="GPT-turbo-preview",
    max_tokens=4096,
    temperature=0.7
)
```

### System Settings

```python
SYSTEM_CONFIG = SystemConfig(
    api_timeout=120,      # API call timeout
    max_retries=3,        # Retry attempts
    retry_delay=1.0,      # Delay between retries
    results_dir="results" # Output directory
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is for educational purposes as part of a course final project.

## 👥 Authors

Multi-LLM Debate Team

## 🙏 Acknowledgments

- OpenAI for GPT
- Anthropic for Claude
- Google for Gemini
- xAI for Grok
