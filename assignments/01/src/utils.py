def is_sorted(sequence):
    return all(sequence[i] >= sequence[i + 1] for i in range(len(sequence) - 1))
