# AudioCraft Music Generation System

A powerful layered music generation system built on Meta's AudioCraft/MusicGen models. This system enables complex multi-layer music composition with conditioning, effects, and advanced audio processing.

## Features

- **Layered Composition**: Build complex musical pieces with multiple synchronized layers
- **Conditional Generation**: Each layer can be conditioned on previous layers for coherent composition
- **Advanced Audio Processing**: Includes crossfade looping, fade effects, and intelligent normalization
- **Caching System**: Smart caching for faster iterative music generation
- **YAML Configuration**: Simple, declarative configuration for defining musical structure

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install audiocraft torch torchaudio pyyaml
```

## Quick Start

```bash
# Generate music from a configuration file
python main.py music/edm_002.yml

# Run with stem export
python main.py music/edm_002.yml --export-stems
```

## Project Structure

```
audiocraft/
├── main.py                 # Main music generation script
├── BUG_FIXES.md           # Documentation of fixed bugs
├── music/                 # YAML configuration files
│   ├── edm_001.yml       # Complex EDM track example
│   ├── edm_002.yml       # Simplified, coherent EDM track
│   ├── music.yml         # Basic example configuration
│   └── test_fixes.yml    # Test configuration for bug fixes
├── tests/                 # Unit and integration tests
│   ├── test_main.py      # Main functionality tests
│   ├── test_coverage.py  # Coverage completion tests
│   ├── test_corner_cases.py # Edge case tests
│   └── run_tests.py      # Test runner script
├── cache/                 # Generated cache directory
└── output/               # Generated music output

```

## Configuration Format

The YAML configuration defines your musical composition:

```yaml
song:
  name: "My Song"
  output_file: "output/my_song.wav"
  sample_rate: 32000
  master_volume: 1.0

sections:
  intro:
    start: 0
    duration: 16
  main:
    start: 16
    duration: 32

layers:
  - name: "drums"
    model: "facebook/musicgen-small"
    sections:
      - section: "intro"
        prompt: "Soft drum pattern at 120 bpm"
        volume: 0.8
      - section: "main"
        prompt: "Energetic drums at 120 bpm"
        volume: 1.0

  - name: "bass"
    model: "facebook/musicgen-small"
    condition_on: ["drums"]  # Generate bass conditioned on drums
    sections:
      - section: "main"
        prompt: "Deep bass line following drums"
        volume: 0.9
        loop: true  # Loop if shorter than section

processing:
  normalization_strategy: "loudness"  # or "peak", "manual"
  normalize_db: -14
  prevent_clipping: true

advanced:
  use_cache: true
  use_full_conditioning: true  # Use more context for generation
  generation_params:
    temperature: 1.0
    top_k: 250
    cfg_coef: 3.0
```

## Key Features Explained

### Layered Generation

Layers are generated sequentially, allowing each layer to be conditioned on previous ones for musical coherence.

### Conditioning System

The `condition_on` parameter allows layers to use previous layers as context, creating harmonically and rhythmically aligned music.

### Audio Effects

- **Looping with Crossfade**: Seamlessly loop short generations to fill longer sections
- **Fade Effects**: Apply fade in/out to audio sections
- **Volume Control**: Per-section volume adjustment

### Smart Caching

The system caches generated sections, speeding up iterative composition and experimentation.

## Advanced Usage

### Using Different Models

```yaml
layers:
  - name: "melody"
    model: "facebook/musicgen-melody"  # Melody-specific model
```

### Conditional Generation Chains

```yaml
layers:
  - name: "drums"
    sections: [...]
  - name: "bass"
    condition_on: ["drums"]
  - name: "lead"
    condition_on: ["drums", "bass"]  # Condition on multiple layers
```

### Manual Normalization Control

```yaml
processing:
  normalization_strategy: "manual"
  normalize_db: -12  # Target peak level in dB
```

## Testing

Run the test suite:

```bash
python run_tests.py
```

Or run specific test files:

```bash
cd tests
python test_main.py
python test_corner_cases.py
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- audiocraft
- pyyaml

## GPU Support

The system automatically uses CUDA if available. Generation is significantly faster on GPU.

## Tips for Best Results

1. **Keep prompts consistent**: Use similar BPM and key across layers
2. **Use conditioning**: Condition melodic layers on rhythmic ones
3. **Start simple**: Begin with drums/bass, then add complexity
4. **Use caching**: Enable caching when iterating on compositions
5. **Adjust temperature**: Lower values (0.8-0.9) for more predictable output

## License

This project uses Meta's AudioCraft models. Please refer to the AudioCraft license for model usage terms.

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting PRs.

## Troubleshooting

### Out of Memory Errors

- Reduce section durations
- Use smaller models (musicgen-small)
- Ensure cache directory has sufficient space

### Poor Audio Quality

- Check your prompts for consistency
- Enable full conditioning
- Adjust generation parameters (temperature, cfg_coef)

### Generation Takes Too Long

- Enable caching
- Use GPU if available
- Generate shorter sections
