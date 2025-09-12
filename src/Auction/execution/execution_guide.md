# Combined Execution Script Usage Guide

This script (`generation_execution.py`) consolidates all the auction experiment functionality. It supports both Claude and GPT models with various configurations for both instruction-based and imitation experiments.

## Features

- **Unified LLM Support**: Both Claude and GPT models
- **Multiple Experiment Types**: Instruction-based and imitation experiments
- **Configurable Risk Preferences**: Risk-averse, risk-seeking, or neutral (instruction only)
- **Imitation Modes**: Direct, shuffle, reverse, mask, and regionshuffle
- **Flexible Model Selection**: Choose specific model versions
- **Threading Support**: Parallel execution for faster results (instruction only)
- **Command-line Interface**: Parameter configuration

## Prerequisites

1. **API Keys**: Set up API keys in environment variables:
   ```bash
   export ANTHROPIC_API_KEY="your_claude_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

2. **Required Files**: Ensure these files exist in the execution directory:
   - `auction_human_data.csv`
   - `profile_generation/umich_undergraduate_profiles.xlsx`
   - `experiment_instructions.txt`

## Usage Examples

### Instruction-Based Experiments (Default)

```bash
# Run with Claude (default)
python generation_execution.py

# Run with GPT
python generation_execution.py --llm gpt

# Run with specific model
python generation_execution.py --llm claude --model claude-3-5-sonnet-latest
python generation_execution.py --llm gpt --model gpt-4o
```

### Imitation Experiments

```bash
# Run direct imitation with GPT (default for imitation)
python generation_execution.py --experiment imitation

# Run with different imitation modes
python generation_execution.py --experiment imitation --mode direct
python generation_execution.py --experiment imitation --mode shuffle
python generation_execution.py --experiment imitation --mode reverse
python generation_execution.py --experiment imitation --mode mask
python generation_execution.py --experiment imitation --mode regionshuffle

# Run with Claude
python generation_execution.py --experiment imitation --llm claude --mode direct
```

### Risk Preference Configurations (Instruction Only)

```bash
# Risk-averse agent
python generation_execution.py --llm claude --risk averse

# Risk-seeking agent  
python generation_execution.py --llm gpt --risk seeking
```

### Imitation-Specific Configurations

```bash
# Custom context length (default is 30 rounds)
python generation_execution.py --experiment imitation --context-num 15

# Process specific bidder groups
python generation_execution.py --experiment imitation --bidder-groups S.1 S.2 S.3

# Custom output file
python generation_execution.py --experiment imitation --output custom_imitation_results.csv
```


### Range and Output Control (Instruction Only)

The `--start` and `--end` parameters control which participant profiles and bidder groups to run:
- Each index `i` corresponds to profile `B{i}` paired with bidder group `S.{i}`
- Default range is 1-40 (profiles B1-B40 with bidder groups S.1-S.40)
- End index is exclusive

```bash
# Run specific range of experiments (profiles B1-B9 with bidder groups S.1-S.9)
python generation_execution.py --start 1 --end 10 --output "test_results.csv"

# Run subset with custom settings (profiles B5-B14 with bidder groups S.5-S.14)
python generation_execution.py --llm gpt --risk averse --start 5 --end 15

# Test with just a few profiles (B1-B2 with S.1-S.2)
python generation_execution.py --start 1 --end 3

# Resume from where you left off (B20-B40 with S.20-S.40)
python generation_execution.py --start 20 --end 41
```

### Parallel Execution (Instruction Only)

```bash
# Use threading for faster execution
python generation_execution.py --threading --workers 5

# High-throughput example
python generation_execution.py --llm gpt --threading --workers 10 --start 1 --end 41
```

## Command-line Arguments

### Common Arguments

| Argument | Choices/Type | Default | Description |
|----------|--------------|---------|-------------|
| `--experiment` | instruction, imitation | instruction | Type of experiment to run |
| `--llm` | claude, gpt | claude | LLM type to use |
| `--model` | string | (auto) | Specific model version |
| `--output` | string | (auto) | Output CSV file name |

### Instruction-Only Arguments

| Argument | Choices/Type | Default | Description |
|----------|--------------|---------|-------------|
| `--risk` | averse, seeking | (none) | Risk preference |
| `--start` | integer | 1 | Start index for experiments |
| `--end` | integer | 41 | End index for experiments |
| `--workers` | integer | 1 | Number of threads |
| `--threading` | flag | False | Enable parallel execution |

### Imitation-Only Arguments

| Argument | Choices/Type | Default | Description |
|----------|--------------|---------|-------------|
| `--mode` | direct, shuffle, reverse, mask, regionshuffle | direct | Imitation mode |
| `--context-num` | integer | 30 | Number of context rounds |
| `--bidder-groups` | list | (all) | Specific bidder groups to process |

## Configuration Equivalents

### Original Instruction Experiments

#### Original `run_claude.py`
```bash
python generation_execution.py --llm claude --model claude-3-7-sonnet-latest --start 1 --end 2
```

#### Original `run_gpt.py`
```bash
python generation_execution.py --llm gpt --start 1 --end 41
```

#### Original `run_risk_claude.py`
```bash
python generation_execution.py --llm claude --model claude-3-5-sonnet-latest --risk seeking --start 3 --end 41
```

#### Original `run_risk_gpt.py`
```bash
python generation_execution.py --llm gpt --risk averse --threading --workers 10 --start 4 --end 41
```

### Original `auction_imitation.ipynb` Equivalents

#### Run all imitation modes
```bash
# Direct mode
python generation_execution.py --experiment imitation --mode direct --llm gpt

# Shuffle mode
python generation_execution.py --experiment imitation --mode shuffle --llm gpt

# Reverse mode
python generation_execution.py --experiment imitation --mode reverse --llm gpt

# Mask mode
python generation_execution.py --experiment imitation --mode mask --llm gpt

# Region shuffle mode
python generation_execution.py --experiment imitation --mode regionshuffle --llm gpt
```

## Programmatic Usage

You can also import and use the functions directly:

### Instruction Experiments

```python
from generation_execution import run_experiments

# Run instruction experiments
run_experiments(
    llm_type="claude",
    model="claude-3-5-sonnet-latest", 
    risk_preference="averse",
    start_index=1,
    end_index=10,
    output_file="my_results.csv",
    use_threading=True,
    max_workers=5
)
```

### Imitation Experiments

```python
from generation_execution import run_imitation_experiments

# Run imitation experiments
run_imitation_experiments(
    mode="direct",
    context_num=30,
    llm_type="gpt",
    model="gpt-4o",
    bidder_groups=["S.1", "S.2", "S.3"],
    output_file="imitation_results.csv"
)
```

## Output

### Instruction Experiment Output

Results are saved to CSV files with columns:
- `Bidder Group`: The bidder group identifier
- `Profile ID`: The participant profile ID  
- `reserve_price_llm`: The LLM's chosen reserve price
- `profit_llm`: The resulting profit

### Imitation Experiment Output

Results are saved to CSV files with columns:
- `bidder_group`: The bidder group identifier
- `round`: Round number (31-60)
- `ai_reserve_price`: LLM's predicted reserve price
- `human_reserve_price`: Actual human reserve price
- `num_bidder`: Number of bidders in the round
- `bid_prices`: Bid prices for the round
- `mode`: Imitation mode used
- `context_num`: Number of context rounds provided


## Performance Notes

- **API Rate Limits**: When using threading, be mindful of API rate limits
- **Memory**: Large experiments may require significant memory for result storage
- **Interruption**: Use Ctrl+C to stop and save partial results