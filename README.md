XDMUSIC: Quantum-Inspired Avant-Garde Music Generation

XDMUSIC (powered by XDUST) is a Python library for generating experimental, quantum-inspired music tracks. It combines quantum computing concepts (qubits, superposition, entanglement) with advanced audio processing techniques like fractal granulation, MIDI generation, and real-time visualization to create unique, avant-garde compositions.
Features

Generate music tracks with quantum-inspired algorithms.
Support for different moods: "glitch", "rage", "dream".
Fractal granulation and quantum mixing for complex sound design.
Real-time visualization of waveforms, spectrograms, and chaos metrics.
Expand short seed audio into full-length tracks with musical DNA analysis.
Export audio, stems, MIDI, and metadata.

Installation

Clone the repository:
git clone https://github.com/0penAGI/XDMUSIC.git
cd XDMUSIC


Install dependencies:
pip install -r requirements.txt


Install the library:
pip install .



Quick Start Guide
Get started with XDMUSIC in just a few steps:

Generate a seed track:
from xdust import quantum_punk_avantgarde

audio, _, _, _, _, _, _, _, _ = quantum_punk_avantgarde(
    n_qubits=3,
    length=15,
    filename="quick_start_seed.wav",
    seed_mood="glitch",
    visualize=False
)


Expand it to a full track:
from xdust import create_full_track_from_seed

expanded_audio, structure, dna, stems = create_full_track_from_seed(
    seed_filename="quick_start_seed.wav",
    target_length=180,
    output_filename="quick_start_full_track.wav"
)


Check the output:

Find the generated WAV files (quick_start_seed.wav, quick_start_full_track.wav).
Explore stems (stem_bass_*.wav, etc.) and MIDI (quick_start_full_track.mid).
View metadata in quick_start_full_track_metadata.json.



Run the example in examples/example_generate_track.py for a full demo.
Usage
Generating a Quantum-Inspired Track
Use the quantum_punk_avantgarde function to create a track with specific parameters.
from xdust import quantum_punk_avantgarde

# Generate a 20-second track with glitch mood
audio, main_q, bass_q, synth_q, anarchists, chaos, ai_chaos, tempo, stems = quantum_punk_avantgarde(
    n_qubits=3,
    length=20,
    filename="my_quantum_track.wav",
    seed_mood="glitch",
    visualize=True
)

Expanding a Seed Track
Extend a short seed audio into a full-length composition with create_full_track_from_seed.
from xdust import create_full_track_from_seed

# Expand a seed track to 180 seconds
expanded_audio, structure, dna, stems = create_full_track_from_seed(
    seed_filename="my_quantum_track.wav",
    target_length=180,
    output_filename="my_full_track.wav"
)

Examples for Different Genres
XDMUSIC supports various musical moods through the seed_mood parameter and other settings. Below are example configurations for different genres:
1. Glitchy Experimental (High Chaos)
from xdust import quantum_punk_avantgarde

audio, _, _, _, _, _, _, _, _ = quantum_punk_avantgarde(
    n_qubits=4,                   # More qubits for complexity
    length=30,
    filename="glitch_track.wav",
    chaos_level=3.0,              # High randomness
    seed_mood="glitch",
    pattern_break_chance=0.5,     # Frequent pattern breaks
    granular_prob=0.6,            # Heavy granular synthesis
    grain_size_ms=80,
    flux=0.7,
    fractal_iterations=4,
    visualize=True
)

Result: A chaotic, fragmented track with glitchy textures and unpredictable shifts.
2. Rage-Driven Industrial
from xdust import quantum_punk_avantgarde

audio, _, _, _, _, _, _, _, _ = quantum_punk_avantgarde(
    n_qubits=3,
    length=25,
    filename="rage_track.wav",
    chaos_level=2.5,
    bass_heavy=True,              # Emphasize bass
    seed_mood="rage",
    pattern_break_chance=0.3,
    granular_prob=0.2,            # Less granulation for punchy sound
    grain_size_ms=50,
    flux=0.4,
    fractal_iterations=2,
    waveform_types=["saw", "square", "noise"]
)

Result: Aggressive, bass-heavy track with industrial vibes and sharp rhythms.
3. Dreamy Ambient
from xdust import quantum_punk_avantgarde

audio, _, _, _, _, _, _, _, _ = quantum_punk_avantgarde(
    n_qubits=3,
    length=40,
    filename="dream_track.wav",
    chaos_level=1.0,              # Low randomness for smooth sound
    bass_heavy=False,
    seed_mood="dream",
    pattern_break_chance=0.1,     # Minimal pattern breaks
    granular_prob=0.4,
    grain_size_ms=100,            # Larger grains for ambient texture
    flux=0.3,
    fractal_iterations=3,
    waveform_types=["sin", "triangle"]
)

Result: Ethereal, flowing track with soft, ambient soundscapes.
4. Expanding a Seed to a Full Track
from xdust import create_full_track_from_seed

expanded_audio, structure, dna, stems = create_full_track_from_seed(
    seed_filename="glitch_track.wav",
    target_length=240,            # 4-minute track
    output_filename="full_glitch_track.wav"
)
print(f"Structure: {list(structure.keys())}")
print(f"Musical DNA: {dna}")

Result: A full-length track with sections (e.g., intro, verse, chorus) based on the seed's musical DNA.
Check the examples/ directory for more detailed examples.
Requirements

Python 3.8+
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.9.0
mido>=1.2.0
matplotlib>=3.4.0
scikit-learn>=1.0.0

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.
Contact
For questions or feedback, reach out at thedubsty@gmail.com.
