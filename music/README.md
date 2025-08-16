# Music Configuration Files

This directory contains YAML configuration files for generating music with the AudioCraft system. Each file defines a complete musical composition with layers, sections, and processing parameters.

## Available Configurations

### edm_001.yml
**Complex EDM Track - "Ascension Anthem"**
- Full EDM arrangement with 7 sections
- 4 layers: drums, bass, lead synth, fx/risers
- 150 seconds total duration
- Features builds, drops, and breakdown
- Advanced production techniques

### edm_002.yml
**Simplified EDM Track - "Smooth Flow"**
- Streamlined 3-section structure
- Focus on coherence and smooth transitions
- 96 seconds duration
- 4 layers with consistent presence
- Optimized for pleasant, continuous listening

### music.yml
**Basic Example Configuration**
- Simple structure for learning
- Minimal layer setup
- Good starting point for new compositions

### test_fixes.yml
**Bug Fix Test Configuration**
- Demonstrates all bug fixes
- Tests start_offset, looping, conditioning
- Validates normalization strategies
- Technical testing configuration

## Configuration Structure

### Basic Structure
```yaml
song:
  name: "Song Name"
  output_file: "output/filename.wav"
  sample_rate: 32000
  master_volume: 1.0

sections:
  section_name:
    start: 0        # Start time in seconds
    duration: 16    # Duration in seconds

layers:
  - name: "layer_name"
    model: "facebook/musicgen-small"
    condition_on: ["other_layer"]  # Optional
    sections:
      - section: "section_name"
        prompt: "Musical description"
        volume: 1.0
        loop: false           # Optional
        fade_out: false       # Optional
        start_offset: 0.0     # Optional
```

## Musical Guidelines

### Tempo Consistency
Maintain consistent BPM across all prompts in a song:
- Slow: 60-90 BPM (ballads, ambient)
- Medium: 90-120 BPM (pop, rock)
- Fast: 120-140 BPM (EDM, dance)
- Very Fast: 140+ BPM (drum & bass, hardcore)

### Prompt Writing Tips

#### Drums
```yaml
prompt: "Four-on-the-floor kick drum at 128 bpm, crisp hi-hats"
prompt: "Breakbeat drums at 140 bpm, syncopated snare"
prompt: "Trap drums at 70 bpm, rolling hi-hats, deep 808 kick"
```

#### Bass
```yaml
prompt: "Deep sub bass following kick pattern, warm and round"
prompt: "Funky slap bass, syncopated groove"
prompt: "Acid bass line, filter sweeps, resonant"
```

#### Melody/Lead
```yaml
prompt: "Euphoric trance lead, major key, soaring melody"
prompt: "Dark techno stabs, minor key, metallic"
prompt: "Gentle piano melody, emotional, with reverb"
```

#### Atmosphere/FX
```yaml
prompt: "Ambient pad, warm texture, slow evolution"
prompt: "White noise riser, building tension"
prompt: "Vinyl crackle, vintage atmosphere"
```

## Section Design

### Common EDM Structure
```yaml
sections:
  intro:
    start: 0
    duration: 16    # 8 bars at 128 BPM
  
  buildup:
    start: 16
    duration: 16    # 8 bars
  
  drop:
    start: 32
    duration: 32    # 16 bars
  
  breakdown:
    start: 64
    duration: 32    # 16 bars
  
  buildup2:
    start: 96
    duration: 16    # 8 bars
  
  drop2:
    start: 112
    duration: 32    # 16 bars
  
  outro:
    start: 144
    duration: 16    # 8 bars
```

### Bar to Second Conversion
At 128 BPM:
- 1 bar = 1.875 seconds
- 4 bars = 7.5 seconds
- 8 bars = 15 seconds
- 16 bars = 30 seconds
- 32 bars = 60 seconds

## Layer Strategies

### Foundation First
1. Start with drums (rhythmic foundation)
2. Add bass (harmonic foundation)
3. Add lead/melody (main musical content)
4. Add atmosphere/fx (polish and texture)

### Conditioning Chain
```yaml
layers:
  - name: "drums"
    # No conditioning - base layer
    
  - name: "bass"
    condition_on: ["drums"]
    # Bass follows drum rhythm
    
  - name: "melody"
    condition_on: ["drums", "bass"]
    # Melody aware of rhythm and harmony
    
  - name: "atmosphere"
    condition_on: ["drums", "bass", "melody"]
    # Atmosphere complements everything
```

## Advanced Techniques

### Dynamic Sections
Use volume and effects to create dynamics:
```yaml
sections:
  - section: "intro"
    prompt: "Soft drums"
    volume: 0.5
    
  - section: "drop"
    prompt: "Powerful drums"
    volume: 1.0
    
  - section: "outro"
    prompt: "Fading drums"
    volume: 0.3
    fade_out: true
```

### Looping Short Generations
For consistent patterns:
```yaml
sections:
  - section: "main"
    prompt: "Simple bass pattern"
    loop: true  # Loop if shorter than section
```

### Delayed Entry
Use start_offset for dramatic effect:
```yaml
sections:
  - section: "intro"
    prompt: "Bass drop"
    start_offset: 4.0  # Enter 4 seconds late
```

## Processing Options

### Normalization Strategies
```yaml
processing:
  normalization_strategy: "loudness"  # Best for streaming
  # OR
  normalization_strategy: "peak"      # Preserve dynamics
  # OR
  normalization_strategy: "manual"    # Custom control
  normalize_db: -14  # Target level
```

### Generation Parameters
```yaml
advanced:
  generation_params:
    temperature: 1.0    # Creativity (0.5-1.5)
    top_k: 250         # Vocabulary limit
    top_p: 0.0         # Nucleus sampling
    cfg_coef: 3.0      # Prompt adherence (1-5)
```

## Creating New Configurations

### Step 1: Define Structure
Decide on:
- Total duration
- Number of sections
- Section transitions

### Step 2: Design Layers
Plan:
- Which instruments/sounds
- How they interact
- When they enter/exit

### Step 3: Write Prompts
- Be specific but not overly complex
- Include tempo/BPM
- Describe tone/mood

### Step 4: Set Processing
- Choose normalization strategy
- Set appropriate volume levels
- Enable caching for iteration

### Step 5: Test and Refine
- Generate with caching enabled
- Listen and adjust prompts
- Fine-tune volumes and effects

## Tips for Success

### Coherence
- Use consistent BPM throughout
- Keep similar tonal/key centers
- Use conditioning for harmonic alignment

### Quality
- Longer sections generate better than many short ones
- Simple, clear prompts work better than complex ones
- Use appropriate models (melody model for melodic content)

### Performance
- Enable caching during development
- Start with shorter test durations
- Use smaller models for drafts

## Common Issues and Solutions

### Abrupt Transitions
- Use fade effects
- Overlap sections slightly
- Condition new layers on existing ones

### Inconsistent Rhythm
- Always specify BPM in prompts
- Use conditioning on drum layer
- Keep rhythmic descriptors consistent

### Poor Mix Balance
- Adjust layer volumes
- Use normalization strategies
- Leave headroom (master_volume < 1.0)

### Generation Takes Too Long
- Enable caching
- Reduce section durations for testing
- Use smaller models initially

## Example Workflows

### Quick Test
1. Copy `edm_002.yml`
2. Modify prompts for your genre
3. Reduce durations for testing
4. Generate with cache enabled

### From Scratch
1. Start with `music.yml` template
2. Add sections incrementally
3. Test each layer individually
4. Combine and adjust volumes

### Remix/Variation
1. Take existing configuration
2. Keep structure, change prompts
3. Adjust tempos and keys
4. Modify layer relationships