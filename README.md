# XDMUSIC

XDUST: Quantum-Inspired Avant-Garde Music Generation
XDUST is a Python library for generating experimental, quantum-inspired music tracks. It combines concepts from quantum computing (qubits, superposition, entanglement) with audio processing techniques like fractal granulation, MIDI generation, and real-time visualization to create unique, avant-garde compositions.
Features

Generate music tracks with quantum-inspired algorithms.
Support for different moods ("glitch", "rage", "dream").
Fractal granulation and quantum mixing for complex sound design.
Real-time visualization of waveforms, spectrograms, and chaos metrics.
Expand short seed audio into full-length tracks with musical DNA analysis.
Export audio, stems, MIDI, and metadata.

Installation

Clone the repository:
git clone https://github.com/yourusername/xdust.git
cd xdust


Install dependencies:
pip install -r requirements.txt


Install the library:
pip install .



Usage
Generate a quantum-inspired track:
from xdust import quantum_punk_avantgarde

# Generate a 20-second track with glitch mood
audio, main_q, bass_q, synth_q, anarchists, chaos, ai_chaos, tempo, stems = quantum_punk_avantgarde(
    n_qubits=3,
    length=20,
    filename="my_quantum_track.wav",
    seed_mood="glitch",
    visualize=True
)

Expand a seed track to a full-length composition:
from xdust import create_full_track_from_seed

# Expand a seed track to 180 seconds
expanded_audio, structure, dna, stems = create_full_track_from_seed(
    seed_filename="my_quantum_track.wav",
    target_length=180,
    output_filename="my_full_track.wav"
)

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
