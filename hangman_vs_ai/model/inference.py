import torch
from collections import Counter
import re
import string
import random
import math
from model.model_definition import HangmanTransformer
import torch.nn.functional as F

random.seed(42)

# Initialise global variables

device = 'cpu'
max_lives = 6
max_len = 10
inference=True
alphabet = list(string.ascii_lowercase)
guess_freq = {
    'e': 0.1175169493434241,
    's': 0.09242216065650082,
    'i': 0.0817352523819177,
    'a': 0.07717338258591046,
    'r': 0.07275266346825236,
    'n': 0.06424307948735275,
    't': 0.06347726197716062,
    'o': 0.061044665180079734,
    'l': 0.053326425959711994,
    'd': 0.04023845455023237,
    'c': 0.03906870584348792,
    'u': 0.03410891126277301,
    'g': 0.031182287091470143,
    'p': 0.028631063660457538,
    'm': 0.027730101883760915,
    'h': 0.0234640478711024,
    'b': 0.02069809521664377,
    'y': 0.01571877979743376,
    'f': 0.014406378809379013,
    'k': 0.01061783453836971,
    'w': 0.00980847054230391,
    'v': 0.00963578620177039,
    'z': 0.00413241134911518,
    'x': 0.003072279658535487,
    'j': 0.00197911270281025,
    'q': 0.0018154379800436966
 }
guess_order = sorted(guess_freq, key=guess_freq.get, reverse=True)
ngram_list = [
    'in',
    'er',
    'es',
    'ed',
    'ng',
    'te',
    're',
    'st',
    'le',
    'at',
    'ti',
    'an',
    'en',
    'ri',
    'ar',
    'on',
    'li',
    'ra',
    'al',
    'or',
    'ing',
    'ers',
    'ate',
    'ter',
    'ies',
    'est',
    'tin',
    'ine',
    'lin',
    'ent',
    'nes',
    'ess',
    'ted',
    'ion',
    'ati'
]
# Map: MASK -> 0, PAD -> 1, 'a' -> 2, ..., 'z' -> 27
special_tokens = ['[PAD]', '[MASK]']
alphabet = list(string.ascii_lowercase)
all_tokens = special_tokens + alphabet
vocab = {letter: index for index, letter in enumerate(all_tokens)}

model = None  # Global model object

def load_model(model_path: str, device: str = "cpu"):
    global model

    # Match these hyperparameters to training
    model = HangmanTransformer(
        vocab_size=28,
        max_len=10,
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        ngram_dim=35,
        aux_dim=19
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

def pad_sequences(sequences, maxlen=None, padding='pre', truncating='pre', value=0):
    
    if not maxlen:
        maxlen = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:  # 'post'
                seq = seq[:maxlen]
        elif len(seq) < maxlen:
            pad_width = maxlen - len(seq)
            pad = [value] * pad_width
            if padding == 'pre':
                seq = pad + seq
            else:  # 'post'
                seq = seq + pad
        padded.append(seq)

    return padded

def generate_features(masked_samples):

    global max_lives, inference

    all_features = []

    # Define vowels and consonants
    vowels = set('aeiou')
    consonants = set(string.ascii_lowercase) - vowels

    # If only one sample is given as a dict, wrap it in a list
    if isinstance(masked_samples, dict):
        masked_samples = [masked_samples]

    # Iterate over each sample
    for sample in masked_samples:

        if inference:
            assert 'guess_history' in sample
        else:
            assert 'next_guesses' in sample and 'original_word' in sample

        current_state = sample['current_state']
        state_set = set(current_state)

        if inference:
            guess_history = sample['guess_history']
            correct_guesses = [letter for letter in guess_history if letter in state_set]
            incorrect_guesses = [letter for letter in guess_history if letter not in correct_guesses]
        else:
            next_guesses = sample['next_guesses']
            original_word = sample['original_word']
            # If not inference, we need to 'simulate' a guess history
            correct_guesses = state_set - {'_'}
            # If word is fully masked, pretend like you have guessed some so you still get incorrect guesses sometimes
            if random.random() < 0.8:
                min_guesses = random.randint(1, 5)
            else:
                min_guesses = 0
            # Construct a plausible guess history until all correct letters are guessed
            guess_history = []
            for letter in guess_order:
                if set(guess_history) >= correct_guesses and len(guess_history) >= min_guesses:
                    break
                guess_history.append(letter)
            # Identify incorrect guesses based on the guess history
            possible_incorrects = [letter for letter in guess_history if letter not in set(original_word)]
            # Randomly sample incorrect guesses to simulate guess limit
            sample_size = min(len(possible_incorrects), random.randint(0, max_lives))
            incorrect_guesses = random.sample(possible_incorrects, k=sample_size)
            # Calculate in-word letter frequencies to add to the labels
            letter_counts = Counter(original_word)
            total_count = sum(letter_counts[letter] for letter in next_guesses)
            label_weights = {letter: letter_counts[letter] / total_count for letter in next_guesses}

        total_correct = len(correct_guesses)
        total_incorrect = len(incorrect_guesses)
        lives_remaining = max(max_lives - len(incorrect_guesses), 0)

        # Basic word structure
        word_length = len(current_state)
        num_masked = current_state.count('_')

        # Get the location of the first and last masked letter
        first_masked_idx = current_state.find('_')
        last_masked_idx = current_state.rfind('_')

        # Get the number of contiguous masked sequences and their average length
        mask_blocks = re.findall(r'_+', current_state)
        num_masked_blocks = len(mask_blocks)
        avg_masked_block_len = sum(len(block) for block in mask_blocks) / num_masked_blocks if num_masked_blocks > 0 else 0

        # Get the average frequency of the remaining unguessed letters
        unguessed = set(string.ascii_lowercase) - set(guess_history)
        remaining_freqs = [guess_freq[letter] for letter in unguessed]
        avg_remaining_freq = sum(remaining_freqs) / len(remaining_freqs) if remaining_freqs else 0
        avg_remaining_freq /= max(guess_freq.values())

        # Binary list: 1 = masked, 0 = revealed
        masked_idx = [1 if letter == '_' else 0 for letter in current_state]

        # Get mask entropy feature
        total = sum(masked_idx)
        if total == 0 or total == len(masked_idx):
            masked_entropy = 0
        else:
            p = total / len(masked_idx)
            masked_entropy = - (p * math.log2(p) + (1 - p) * math.log2(1 - p))

        # Prefix: how many '_' at the start
        masked_prefix_len = 0
        for letter in current_state:
            if letter == '_':
                masked_prefix_len += 1
            else:
                break

        # Suffix: how many '_' at the end
        masked_suffix_len = 0
        for letter in reversed(current_state):
            if letter == '_':
                masked_suffix_len += 1
            else:
                break

        # Split correct/incorrect guesses by vowels and consonants
        correct_vowels = [letter for letter in correct_guesses if letter in vowels]
        correct_consonants = [letter for letter in correct_guesses if letter in consonants]
        incorrect_vowels = [letter for letter in incorrect_guesses if letter in vowels]
        incorrect_consonants = [letter for letter in incorrect_guesses if letter in consonants]

        num_correct_vowels = len(correct_vowels)
        num_correct_consonants = len(correct_consonants)
        num_incorrect_vowels = len(incorrect_vowels)
        num_incorrect_consonants = len(incorrect_consonants)

        # Repetition and revealed chunks
        revealed_letters = [letter for letter in current_state if letter != '_']
        revealed_counts = Counter(revealed_letters)
        num_repeated_letters = sum(1 for letter, count in revealed_counts.items() if count > 1)
        num_contiguous_letters = len(re.findall(r'[a-zA-Z]+', current_state))

        # Count common n-grams (top 20 bigrams and top 15 trigrams from training dictionary) in current state
        ngram_counts = {}
        for ngram in ngram_list:
            ngram_count = current_state.count(ngram)
            if ngram_count > 0:
                ngram_counts[ngram] = ngram_count

        # Final dictionary of features for this sample
        features = {
            'current_state': current_state,
            'word_length': word_length,
            'num_masked': num_masked,
            'first_masked_idx': first_masked_idx,
            'last_masked_idx': last_masked_idx,
            'num_masked_blocks': num_masked_blocks,
            'avg_masked_block_len': avg_masked_block_len,
            'avg_remaining_freq': avg_remaining_freq,
            'masked_entropy': masked_entropy,
            'masked_idx': masked_idx,
            'masked_prefix_len': masked_prefix_len,
            'masked_suffix_len': masked_suffix_len,
            'total_correct_guesses': total_correct,
            'total_incorrect_guesses': total_incorrect,
            'lives_remaining': lives_remaining,
            'correct_vowels': correct_vowels,
            'correct_consonants': correct_consonants,
            'incorrect_vowels': incorrect_vowels,
            'incorrect_consonants': incorrect_consonants,
            'num_correct_vowels': num_correct_vowels,
            'num_correct_consonants': num_correct_consonants,
            'num_incorrect_vowels': num_incorrect_vowels,
            'num_incorrect_consonants': num_incorrect_consonants,
            'num_repeated_letters': num_repeated_letters,
            'num_contiguous_letters': num_contiguous_letters,
            'ngram_counts': ngram_counts
        }

        if not inference:
            features['next_guesses'] = label_weights

        # Store result
        all_features.append(features)

    return all_features

def encode_features(features):

    global vocab, ngram_list, max_lives, max_len, inference

    encoded = []

    if isinstance(features, dict):
        features = [features]  # Wrap single input into a list

    # Predefine keys to flatten multi-hot features in order
    char_keys = ['correct_vowels', 'correct_consonants', 'incorrect_vowels', 'incorrect_consonants']

    for feature in features:
        # 1. Tokenize and pad current state
        current_state = [
            vocab['[MASK]'] if letter == '_' else vocab[letter]
            for letter in feature['current_state']
        ]
        current_state = pad_sequences([current_state], maxlen=max_len, padding='post', value=vocab['[PAD]'])[0]

        # 2. Masked index vector
        masked_idx = pad_sequences([feature['masked_idx']], maxlen=max_len, padding='post', value=0)[0]

        # 3. Scalar features
        wl = max(1, feature['word_length'])
        nf = feature
        norm_features = [
            wl / max_len,
            1 - nf['num_masked'] / wl,
            nf['first_masked_idx'] / wl,
            nf['last_masked_idx'] / wl,
            nf['num_masked_blocks'] / wl,
            nf['avg_masked_block_len'] / wl,
            nf['avg_remaining_freq'],
            nf['masked_entropy'],
            nf['lives_remaining'] / max_lives,
            nf['masked_prefix_len'] / wl,
            nf['masked_suffix_len'] / wl,
            nf['total_correct_guesses'] / max_lives,
            nf['total_incorrect_guesses'] / max_lives,
            nf['num_correct_vowels'] / 5,
            nf['num_incorrect_vowels'] / 5,
            nf['num_correct_consonants'] / 21,
            nf['num_incorrect_consonants'] / 21,
            nf['num_repeated_letters'] / wl,
            nf['num_contiguous_letters'] / wl
        ]

        # 4. Flattened multi-hot character features
        char_multi_hot = []
        for key in char_keys:
            vec = [0] * 26
            for letter in feature[key]:
                idx = ord(letter) - ord('a')
                vec[idx] = 1
            char_multi_hot.extend(vec)

        # 5. Normalized n-gram frequency vector
        ngram_counts = [feature['ngram_counts'].get(ngram, 0) for ngram in ngram_list]
        total = sum(ngram_counts) or 1
        ngram_vector = [count / total for count in ngram_counts]

        # 6. Build output sample
        output = {
            'input_ids': current_state,
            'masked_idx': masked_idx,
            'norm_features': norm_features,
            'char_multi_hot': char_multi_hot,
            'ngram_vector': ngram_vector
        }

        # 7. Weighted label
        if not inference:
            label = [0.0] * 26
            for letter, weight in feature['next_guesses'].items():
                idx = ord(letter) - ord('a')
                label[idx] = weight
            output['label'] = label

        encoded.append(output)

    return encoded

def predict_next_letter(current_state: str, guessed_letters: list) -> str:
    
    global model, device

    # Step 1: Build sample
    sample = {
        'current_state': current_state,
        'guess_history': guessed_letters
    }

    # Step 2: Feature generation and encoding
    features = generate_features(sample)
    encoded = encode_features(features)[0]

    # Step 3: Move tensors to device
    input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
    masked_idx = torch.tensor(encoded['masked_idx'], dtype=torch.long, device=device).unsqueeze(0)
    norm_features = torch.tensor(encoded['norm_features'], dtype=torch.float32, device=device).unsqueeze(0)
    char_multi_hot = torch.tensor(encoded['char_multi_hot'], dtype=torch.float32, device=device).unsqueeze(0)
    ngram_vector = torch.tensor(encoded['ngram_vector'], dtype=torch.float32, device=device).unsqueeze(0)

    # Step 4: Model inference
    model.eval()

    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            masked_idx=masked_idx,
            norm_features=norm_features,
            char_multi_hot=char_multi_hot,
            ngram_vector=ngram_vector
        )

        probs = torch.sigmoid(logits).squeeze(0)  # shape: [26]

        # Filter out guessed letters
        filtered = [(char, probs[i].item()) for i, char in enumerate(alphabet) if char not in guessed_letters]

        if not filtered:
            return None

        top_char, top_prob = max(filtered, key=lambda x: x[1])

        return top_char