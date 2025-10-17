# 🌌 XDMUSIC

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

**Quantum-Inspired Avant-Garde Music Generation**

*Powered by XDUST - Where quantum computing meets sonic chaos*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Examples](#-examples) • [Documentation](#-documentation)

</div>

---

## 🎸 What is XDMUSIC?

XDMUSIC is a revolutionary Python library that fuses **quantum computing concepts** with **advanced audio processing** to generate truly unique, experimental music. Using binary qubits, superposition, and entanglement, it creates compositions that exist at the intersection of mathematics, physics, and sound design.

### 🧬 Key Concepts

- **Quantum-Inspired Algorithms**: Real quantum computing principles (qubits, gates, superposition) applied to music generation
- **Fractal Granulation**: Recursive audio processing with pitch shifting and time stretching
- **Musical DNA**: Analyze and extract the essence of audio to generate intelligent expansions
- **Chaos Engineering**: Controlled randomness through entropy, decoherence, and attention matrices

---

## ✨ Features

🎵 **Generative Music Engine**
- Multiple moods: `glitch`, `rage`, `dream`
- Dynamic tempo and meter changes (4/4, 7/8)
- Pattern breaks and quantum rifts

🔊 **Advanced Audio Processing**
- Fractal granular synthesis with recursive transformations
- Quantum mixer for entangled track blending
- Real-time pitch shifting and time stretching
- Professional mastering chain (compression, limiting)

📊 **Production-Ready Output**
- Multi-track stems (bass, lead, synth, anarchy)
- MIDI export for DAW integration
- JSON metadata with chaos metrics
- Real-time visualization (waveforms, spectrograms, 3D attention surfaces)

🚀 **Track Expansion**
- Transform 15-second seeds into full 3+ minute compositions
- Automatic structure generation (intro, verse, drop, outro)
- Musical DNA analysis for intelligent evolution
- Seamless transitions between sections

---

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/0penAGI/XDMUSIC.git
cd XDMUSIC
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install the library
```bash
pip install .
```

### Requirements
- Python 3.8+
- NumPy, SciPy, Librosa
- Matplotlib, scikit-learn
- midiutil, mido

---

## 🚀 Quick Start

### Generate a Seed Track (15 seconds)

```python
from xdust import quantum_punk_avantgarde

audio, main_q, bass_q, synth_q, anarchists, chaos, ai_chaos, tempo, stems = quantum_punk_avantgarde(
    n_qubits=3,
    length=15,
    filename="my_seed.wav",
    seed_mood="glitch",
    chaos_level=2.0,
    visualize=True
)

print(f"✅ Generated: my_seed.wav")
print(f"🧬 Chaos entropy: {max(chaos):.3f}")
```

### Expand to Full Track (3+ minutes)

```python
from xdust import create_full_track_from_seed

expanded_audio, structure, dna, stems = create_full_track_from_seed(
    seed_filename="my_seed.wav",
    target_length=180,  # 3 minutes
    output_filename="my_full_track.wav"
)

print(f"✅ Generated: my_full_track.wav")
print(f"📐 Structure: {list(structure.keys())}")
print(f"🎼 Musical DNA: {dna['mood']}")
```

### Output Files

After generation, you'll find:
- 🎵 `my_full_track.wav` - Main audio
- 🎛️ `stem_bass_*.wav`, `stem_lead_*.wav`, etc. - Individual stems
- 🎹 `my_full_track.mid` - MIDI data
- 📊 `my_full_track_metadata.json` - Complete metadata

---

## 🎨 Examples

### 1. Glitchy Experimental (High Chaos)

```python
from xdust import quantum_punk_avantgarde

audio, *_ = quantum_punk_avantgarde(
    n_qubits=4,                    # More complexity
    length=30,
    filename="glitch_chaos.wav",
    chaos_level=3.5,               # Maximum chaos
    seed_mood="glitch",
    pattern_break_chance=0.6,      # Frequent breaks
    granular_prob=0.7,             # Heavy granulation
    grain_size_ms=80,
    flux=0.8,                      # High randomness
    fractal_iterations=4,
    waveform_types=["saw", "noise", "triangle"]
)
```

**Result**: Chaotic, fragmented track with unpredictable glitchy textures and quantum rifts.

---

### 2. Rage-Driven Industrial

```python
audio, *_ = quantum_punk_avantgarde(
    n_qubits=3,
    length=25,
    filename="industrial_rage.wav",
    chaos_level=2.5,
    bass_heavy=True,               # Heavy bass emphasis
    seed_mood="rage",
    pattern_break_chance=0.3,
    granular_prob=0.2,             # Punchy, less granular
    waveform_types=["saw", "square", "noise"]
)
```

**Result**: Aggressive, bass-heavy industrial track with sharp, distorted rhythms.

---

### 3. Dreamy Ambient Soundscape

```python
audio, *_ = quantum_punk_avantgarde(
    n_qubits=3,
    length=40,
    filename="ambient_dream.wav",
    chaos_level=1.0,               # Low chaos for smoothness
    bass_heavy=False,
    seed_mood="dream",
    pattern_break_chance=0.1,      # Minimal interruptions
    granular_prob=0.5,
    grain_size_ms=120,             # Large ambient grains
    flux=0.3,
    waveform_types=["sin", "triangle"]
)
```

**Result**: Ethereal, flowing ambient composition with soft, meditative qualities.

---

### 4. Full Track Expansion with Custom Structure

```python
from xdust import create_full_track_from_seed

# Generate and expand in one go
expanded, structure, dna, stems = create_full_track_from_seed(
    seed_filename="glitch_chaos.wav",
    target_length=240,              # 4-minute epic
    output_filename="epic_journey.wav"
)

# Inspect the generated structure
print("🎵 Track Structure:")
for section, (start, end) in structure.items():
    print(f"  {section}: {start}s - {end}s ({end-start}s)")

print(f"\n🧬 Musical DNA:")
print(f"  Energy: {dna['energy_profile']:.2f}")
print(f"  Brightness: {dna['spectral_brightness']:.0f}Hz")
print(f"  Mood: {dna['mood']}")
```

**Result**: Full composition with intelligent sections (intro, build, drop, chaos, outro) based on seed analysis.

---

## 🧪 Advanced Parameters

### Quantum Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_qubits` | int | 3 | Number of qubits (2^n quantum states) |
| `chaos_level` | float | 2.0 | Randomness intensity (0.5-5.0) |
| `seed_mood` | str | "glitch" | Initial mood: "glitch", "rage", "dream" |

### Audio Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bass_heavy` | bool | True | Emphasize low frequencies |
| `pattern_break_chance` | float | 0.4 | Probability of quantum rifts (0-1) |
| `granular_prob` | float | 0.3 | Granular synthesis chance (0-1) |
| `grain_size_ms` | int | 50 | Granule duration in milliseconds |
| `flux` | float | 0.5 | Temporal randomness (0-1) |
| `fractal_iterations` | int | 3 | Recursive granulation depth |

### Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | "quantum_punk.wav" | Output filename |
| `sample_rate` | int | 44100 | Audio sample rate |
| `visualize` | bool | True | Show real-time visualization |

---

## 🎛️ How It Works

### 1. Quantum State Initialization

```python
# Binary qubits represent quantum states
q = BinaryQubit(n_qubits=3)  # 2^3 = 8 possible states

# Apply quantum gates
q.apply_gate(hadamard(3))        # Superposition
q.apply_gate(cnot(3, 0, 1))      # Entanglement
q.apply_gate(phase_shift(3, π/4)) # Phase rotation
```

### 2. Superposition & Measurement

```python
# Apply kernel-based superposition
q.apply_superposition(hamming_kernel, tau=0.5)

# Measure to collapse into musical note
state = q.measure()
note = sum(state * (2 ** np.arange(n_qubits)[::-1]))
```

### 3. Fractal Granulation

```python
# Recursive audio transformation
grain = audio_segment[start:end]
grain = pitch_shift(grain, ratio=1.2)
grain = time_stretch(grain, factor=0.9)
grain = recursive_granule(grain, depth-1)  # Recurse!
```

### 4. Musical DNA Extraction

```python
# Analyze seed audio
analysis = analyze_seed_sample(audio)
dna = extract_musical_dna(analysis)

# DNA contains: energy, brightness, tempo, mood, harmonics
# Used to intelligently generate full track structure
```

---

## 📊 Visualization

XDMUSIC provides real-time visualization:

- 🌊 **Waveform**: Live audio amplitude over time
- 🔥 **Spectrogram**: Frequency spectrum analysis
- 🧬 **Entropy Curve**: Chaos evolution during generation
- 🧠 **3D Attention Surface**: Quantum state interactions

Enable with `visualize=True` parameter.

---

## 🗂️ Project Structure

```
XDMUSIC/
├── xdust/
│   ├── __init__.py
│   ├── quantum_core.py      # BinaryQubit, quantum gates
│   ├── audio_engine.py      # Audio synthesis & effects
│   ├── expansion.py         # TrackExpansionEngine
│   └── utils.py             # Helper functions
├── examples/
│   ├── example_generate_track.py
│   ├── example_expansion.py
│   └── example_genres.py
├── tests/
│   └── test_quantum_core.py
├── README.md
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## 🛣️ Roadmap

- [ ] **VST/AU Plugin** - Real-time performance mode
- [ ] **MIDI Input** - Use external controllers
- [ ] **Style Transfer** - Blend multiple seed tracks
- [ ] **Web Interface** - Browser-based generator
- [ ] **Euclidean Rhythms** - Additional rhythm patterns
- [ ] **GPU Acceleration** - CUDA/OpenCL support
- [ ] **Preset Library** - Community-contributed templates

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

**XDMUSIC** is powered by **XDUST**, a quantum-inspired computing framework for creative applications.

**Creator**: [0penAGI](https://github.com/0penAGI)

### Inspiration & Technologies

- Quantum Computing: Superposition, entanglement, measurement
- Digital Signal Processing: Librosa, granular synthesis
- Generative Music: Algorithmic composition, chaos theory
- Avant-Garde: IDM, glitch, breakcore, experimental electronic

---

## 📬 Contact & Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Share your creations and ideas


---

## 🎵 Example Tracks

Want to hear what XDMUSIC creates? Check out example outputs:

- 🎧 [Glitch Chaos - SoundCloud](#) *(coming soon)*
- 🎧 [Industrial Rage - YouTube](#) *(coming soon)*
- 🎧 [Ambient Dream - Bandcamp](#) *(coming soon)*

---

<div align="center">

**🌌 Built with quantum chaos and sonic alchemy 🌌**

*XDMUSIC - Where mathematics becomes music*

⭐ Star this repo if you enjoy experimental audio! ⭐

</div>
