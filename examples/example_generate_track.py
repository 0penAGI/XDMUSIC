from xdust import quantum_punk_avantgarde, create_full_track_from_seed

def main():
    # Generate a 15-second seed track
    audio, _, _, _, _, _, _, _, _ = quantum_punk_avantgarde(
        n_qubits=3,
        length=15,
        filename="example_seed.wav",
        chaos_level=2.0,
        bass_heavy=True,
        visualize=True,
        seed_mood="glitch"
    )

    # Expand the seed to a 180-second track
    expanded_audio, structure, dna, stems = create_full_track_from_seed(
        seed_filename="example_seed.wav",
        target_length=180,
        output_filename="example_full_track.wav"
    )

    print(f"Generated full track with structure: {list(structure.keys())}")
    print(f"Musical DNA: {dna}")

if __name__ == "__main__":
    main()
