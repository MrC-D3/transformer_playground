#!/usr/bin/env python3
"""Create small train/test datasets from a Morph-it! word list.

Morph-it! contains one form per row, followed by lemma and grammatical tag.
This project currently accepts ASCII words only, so accented forms and symbols
are intentionally skipped.
"""

import argparse
import random
from pathlib import Path


def load_ascii_forms(path: Path) -> list[str]:
    forms: set[str] = set()
    with path.open("r", encoding="latin-1") as source:
        for line in source:
            fields = line.rstrip("\n").split("\t")
            if not fields:
                continue
            form = fields[0].strip().lower()
            if form and all("a" <= character <= "z" for character in form):
                forms.add(form)
    return sorted(forms)


def mutate_word(word: str, max_changes: int, rng: random.Random) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    candidate = word
    for _ in range(rng.randint(1, max_changes)):
        if rng.random() < 0.5:
            position = rng.randrange(len(candidate))
            replacements = alphabet.replace(candidate[position], "")
            candidate = (candidate[:position] + rng.choice(replacements) +
                         candidate[position + 1:])
        else:
            position = rng.randrange(len(candidate) + 1)
            candidate = candidate[:position] + rng.choice(alphabet) + candidate[position:]
    return candidate


def make_negative(
    word: str,
    known_words: set[str],
    used: set[str],
    max_changes: int,
    rng: random.Random,
) -> str:
    while True:
        candidate = mutate_word(word, max_changes, rng)
        if candidate != word and candidate not in known_words and candidate not in used:
            return candidate


def write_dataset(path: Path, samples: list[tuple[int, str]]) -> None:
    with path.open("w", encoding="ascii") as output:
        output.write(f"{len(samples)}\n")
        for label, word in samples:
            output.write(f"{label} {word}\n")


def write_balanced_split(
    path: Path,
    pairs: list[tuple[str, str]],
    rng: random.Random,
) -> None:
    samples: list[tuple[int, str]] = []
    for positive, negative in pairs:
        samples.append((1, positive))
        samples.append((0, negative))
    rng.shuffle(samples)
    write_dataset(path, samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("morphit", type=Path, help="percorso del file morph-it_048.txt")
    parser.add_argument("--train-positive", type=int, default=4000)
    parser.add_argument("--test-positive", type=int, default=1000)
    parser.add_argument("--train-output", type=Path, default=Path("train_data.txt"))
    parser.add_argument("--test-output", type=Path, default=Path("test_data.txt"))
    parser.add_argument(
        "--cleaned-output",
        type=Path,
        help="salva tutte le forme ASCII come esempi positivi con label 1",
    )
    parser.add_argument(
        "--negative-output",
        type=Path,
        help="salva una forma inventata per ogni forma ASCII con label 0",
    )
    parser.add_argument(
        "--split-training-output",
        type=Path,
        help="salva l'80%% delle coppie positive/negative in un dataset bilanciato",
    )
    parser.add_argument(
        "--split-validation-output",
        type=Path,
        help="salva il 10%% delle coppie positive/negative in un dataset bilanciato",
    )
    parser.add_argument(
        "--split-test-output",
        type=Path,
        help="salva il restante 10%% delle coppie positive/negative in un dataset bilanciato",
    )
    parser.add_argument(
        "--max-changes",
        type=int,
        default=3,
        help="numero massimo di caratteri cambiati o aggiunti nelle parole negative",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.train_positive <= 0 or args.test_positive <= 0:
        parser.error("il numero di esempi positivi deve essere maggiore di zero")
    if args.max_changes <= 0:
        parser.error("--max-changes deve essere maggiore di zero")

    forms = load_ascii_forms(args.morphit)
    required = args.train_positive + args.test_positive
    if len(forms) < required:
        parser.error(f"servono {required} forme, ma ne sono state trovate {len(forms)}")

    rng = random.Random(args.seed)
    rng.shuffle(forms)
    train_words = forms[: args.train_positive]
    test_words = forms[args.train_positive:required]
    known_words = set(forms)
    used_negative_words: set[str] = set()

    train_samples = [(1, word) for word in train_words]
    train_samples.extend(
        (0, make_negative(word, known_words, used_negative_words, args.max_changes, rng))
        for word in train_words
    )
    test_samples = [(1, word) for word in test_words]
    test_samples.extend(
        (0, make_negative(word, known_words, used_negative_words, args.max_changes, rng))
        for word in test_words
    )
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    write_dataset(args.train_output, train_samples)
    write_dataset(args.test_output, test_samples)
    if args.cleaned_output is not None:
        write_dataset(
            args.cleaned_output,
            [(1, word) for word in forms],
        )
    split_outputs = (
        args.split_training_output,
        args.split_validation_output,
        args.split_test_output,
    )
    if any(output is not None for output in split_outputs) and not all(
        output is not None for output in split_outputs
    ):
        parser.error("i tre output degli split devono essere indicati insieme")

    negative_words: list[str] = []
    if args.negative_output is not None or all(
        output is not None for output in split_outputs
    ):
        used_negative_words: set[str] = set()
        for word in forms:
            negative = make_negative(
                word, known_words, used_negative_words, args.max_changes, rng
            )
            negative_words.append(negative)
            used_negative_words.add(negative)
        if args.negative_output is not None:
            write_dataset(
                args.negative_output,
                [(0, word) for word in negative_words],
            )
        if all(output is not None for output in split_outputs):
            pairs = list(zip(forms, negative_words))
            rng.shuffle(pairs)
            training_count = len(pairs) * 8 // 10
            validation_count = len(pairs) // 10
            training_pairs = pairs[:training_count]
            validation_pairs = pairs[training_count:training_count + validation_count]
            test_pairs = pairs[training_count + validation_count:]
            write_balanced_split(args.split_training_output, training_pairs, rng)
            write_balanced_split(args.split_validation_output, validation_pairs, rng)
            write_balanced_split(args.split_test_output, test_pairs, rng)
    print(f"Forme ASCII disponibili: {len(forms)}")
    print(f"Training: {len(train_samples)} esempi in {args.train_output}")
    print(f"Test: {len(test_samples)} esempi in {args.test_output}")
    if args.cleaned_output is not None:
        print(f"Dataset pulito: {len(forms)} esempi in {args.cleaned_output}")
    if args.negative_output is not None:
        print(f"Dataset negativo: {len(forms)} esempi in {args.negative_output}")
    if all(output is not None for output in split_outputs):
        print(f"Training split: {len(training_pairs) * 2} esempi in {args.split_training_output}")
        print(f"Validation split: {len(validation_pairs) * 2} esempi in {args.split_validation_output}")
        print(f"Test split: {len(test_pairs) * 2} esempi in {args.split_test_output}")


if __name__ == "__main__":
    main()
