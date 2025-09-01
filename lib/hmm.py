# HMM (Hidden Markov Model) FUNCTIONS

import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import CategoricalHMM
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import train_test_split

ZA_encoding = {'Z': 0, 'A': 1, 'K': 2, 'R': 3}
AA2_encoding = {'A': 0, '02': 1, 'K': 2, 'R': 3}
# ============================================================================
def plot_transition_matrix(transmat, state_mapping, title="State Transition Matrix", ax=None, title_x=-0.5):
    num_states = transmat.shape[0]
    state_labels = [
        f"S{i} = {state_mapping.get(i, f'S{i}')}" for i in range(num_states)
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Heatmap without interpolation or grid lines
    im = ax.imshow(
        transmat,
        cmap='Blues',
        interpolation='none',
        aspect='equal'  # Equal squares
    )

    # Major ticks and labels
    ax.set_xticks(np.arange(num_states))
    ax.set_yticks(np.arange(num_states))
    ax.set_xticklabels(state_labels, rotation=45, ha="right")
    ax.set_yticklabels(state_labels)

    # Remove all grid and tick lines
    ax.grid(False)
    ax.tick_params(which='both', bottom=False, left=False)

    # Hide spines (borders)
    for edge in ["top", "bottom", "left", "right"]:
        ax.spines[edge].set_visible(False)

    # Add text annotations
    for i in range(num_states):
        for j in range(num_states):
            val = transmat[i, j]
            ax.text(j, i, f"{val:.2%}", ha="center", va="center",
                    color="white" if val > 0.5 else "black")

    # Custom axis labels and title
    ax.set_title(title, loc='left', x=title_x)
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")

    # Tight layout without colorbar
    if ax is None:
        plt.tight_layout()
        plt.show()

    
def plot_emission_matrix(emission_probs, state_mapping, obs_mapping):
    num_states, num_obs = emission_probs.shape

    state_labels = [f"S{i} = {state_mapping.get(i, f'S{i}')}" for i in range(num_states)]
    obs_labels = [f"{v} = {k}" for k, v in sorted(obs_mapping.items(), key=lambda x: x[1])]

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        emission_probs,
        cmap='Oranges',
        interpolation='none',
        aspect='equal'
    )

    # --- Minor ticks for tight layout (but turn off visuals) ---
    ax.set_xticks(np.arange(num_obs) + 0.5, minor=True)
    ax.set_yticks(np.arange(num_states) + 0.5, minor=True)
    ax.grid(False)
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- Major ticks and labels ---
    ax.set_xticks(np.arange(num_obs))
    ax.set_yticks(np.arange(num_states))
    ax.set_xticklabels(obs_labels, rotation=45, ha="right")
    ax.set_yticklabels(state_labels)

    # --- Remove plot border spines ---
    for edge in ["top", "bottom", "left", "right"]:
        ax.spines[edge].set_visible(False)

    # --- Add percentage text inside cells ---
    for i in range(num_states):
        for j in range(num_obs):
            val = emission_probs[i, j]
            ax.text(j, i, f"{val:.2%}", ha="center", va="center",
                    color="white" if val > 0.5 else "black")

    ax.set_title("HMM Emission Probability Matrix")
    plt.colorbar(im, ax=ax, label="Emission Probability")
    plt.tight_layout()
    plt.show()
    
import joblib

def save_hmm_results(filepath, results_dict):
    """
    Save HMM result dictionary (with models and titles) to a file.
    
    Parameters:
        filepath (str): Destination path, e.g. 'models/hmm_results.joblib'
        results_dict (dict): Dictionary of results like {'ZA': ZA_best_result, 'AA2': AA2_best_result}
    """
    joblib.dump(results_dict, filepath)
    print(f"✅ HMM results saved to: {filepath}")

def load_hmm_results(filepath):
    """
    Load HMM result dictionary (with models and titles) from a file.
    
    Parameters:
        filepath (str): Path to the joblib file
    Returns:
        dict: The restored results dictionary
    """
    results = joblib.load(filepath)
    print(f"✅ HMM results loaded from: {filepath}")
    return results

def reorder_hmm_matrix_by_emissions(transmat, emissionprob, encoding):
    """
    Reorders HMM transition matrix rows and columns based on most likely emission mapping.

    Parameters:
        transmat (np.ndarray): HMM transition matrix (n_states x n_states)
        emissionprob (np.ndarray): HMM emissionprob_ matrix (n_states x n_symbols)
        encoding (dict): e.g., {'Z': 0, 'A': 1, 'K': 2, 'R': 3}

    Returns:
        np.ndarray: Reordered transition matrix (states sorted by emission symbol)
        List[int]: Original state indices reordered
    """
    n_states = transmat.shape[0]

    # Get reverse encoding: symbol_index -> symbol_label
    reverse_encoding = {v: k for k, v in encoding.items()}

    # Find most likely emitted symbol index per hidden state
    dominant_emissions = emissionprob.argmax(axis=1)  # shape (n_states,)

    # Create mapping from HMM state → meaning based on emission
    state_order = sorted(range(n_states), key=lambda i: dominant_emissions[i])

    # Reorder transition matrix
    reordered_transmat = transmat[np.ix_(state_order, state_order)]

    return reordered_transmat, state_order

def prepare_hmm_data(seqs):
    X = np.concatenate(seqs).reshape(-1, 1)
    lengths = [len(seq) for seq in seqs]
    return X, lengths

def train_and_score(n_hidden, X_train, lengths_train, X_test, lengths_test):
    try:
        model = CategoricalHMM(n_components=n_hidden, n_iter=100, random_state=42, verbose=False)
        model.n_features = 4
        model.fit(X_train, lengths_train)
        train_ll = model.score(X_train, lengths_train)
        test_ll = model.score(X_test, lengths_test)
        return {
            "n_hidden_states": n_hidden,
            "train_ll": train_ll,
            "test_ll": test_ll,
            "train_ll_per_obs": train_ll / len(X_train),
            "test_ll_per_obs": test_ll / len(X_test),
            "model": model,
        }
    except Exception as e:
        return {
            "n_hidden_states": n_hidden,
            "error": str(e)
        }

def run_hmm_pipeline(data, encoding, type, test_size=0.2, hidden_range=range(2, 7)):
    data = data.copy()
    data['obs'] = data['draw_character'].map(encoding)
    # next line should work better but isnt in comparison with "count method, dont understand why"
    # data['obs'] = data['draw_character' if type == "ZA" else "deprecated_draw_character"].map(encoding)
    grouped = data.groupby('id_person')['obs'].apply(list)
    train_ids, test_ids = train_test_split(grouped.index, test_size=test_size, random_state=42)
    train_seqs = grouped.loc[train_ids].values
    test_seqs = grouped.loc[test_ids].values
    
    X_train, lengths_train = prepare_hmm_data(train_seqs)
    X_test, lengths_test = prepare_hmm_data(test_seqs)

    with parallel_backend("loky"):
        results = Parallel(n_jobs=-2)(
            delayed(train_and_score)(n, X_train, lengths_train, X_test, lengths_test)
            for n in hidden_range
        )
    return results

def print_results(results, label=""):
    print(f"\n📊 HMM Evaluation Results: {label}")
    for r in results:
        if "error" in r:
            print(f"⚠️  {r['n_hidden_states']} states: {r['error']}")
        else:
            print(f"✅ {r['n_hidden_states']} states: "
                  f"Train LL/obs = {r['train_ll_per_obs']:.4f}, "
                  f"Test LL/obs = {r['test_ll_per_obs']:.4f}")

# --- Step 1: Prepare ZA process ---

def predict_next_observation(model: CategoricalHMM, obs_seq: list[int]) -> dict:
    X = np.array(obs_seq).reshape(-1, 1)
    hidden_seq = model.predict(X)
    last_state = hidden_seq[-1]

    next_state_probs = model.transmat_[last_state]
    next_obs_probs = next_state_probs @ model.emissionprob_

    return {
        "next_obs_distribution": next_obs_probs,
        "next_obs_predicted": int(np.argmax(next_obs_probs)),
        "next_obs_prob": float(np.max(next_obs_probs)),
    }

def compute_transition_matrix(data, encoding, character_col):
    """
    Compute a transition matrix based on direct observation of state transitions.
    
    Parameters:
        data (pd.DataFrame): Input data with 'id_person', character_col, and 'datetime'.
        encoding (dict): Mapping of character labels to integers.
        character_col (str): Column name with character values (e.g., 'draw_character').
    
    Returns:
        np.ndarray: Normalized transition matrix (n_states x n_states)
    """
    n_states = len(encoding)
    trans_counts = np.zeros((n_states, n_states), dtype=int)

    # Map characters to obs
    data = data.copy()
    data['obs'] = data[character_col].map(encoding)

    # Group and sort per person
    grouped = data.sort_values("datetime").groupby("id_person")['obs'].apply(list)

    for seq in grouped:
        for i in range(1, len(seq)):
            from_state = seq[i - 1]
            to_state = seq[i]
            if from_state is not None and to_state is not None:
                trans_counts[from_state, to_state] += 1

    # Normalize rows to get probabilities
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_probs = np.divide(trans_counts, row_sums, where=row_sums != 0)

    return trans_probs

def predict_next_observation(model: CategoricalHMM, obs_seq: list[int]) -> dict:
    X = np.array(obs_seq).reshape(-1, 1)
    hidden_seq = model.predict(X)
    last_state = hidden_seq[-1]

    next_state_probs = model.transmat_[last_state]
    next_obs_probs = next_state_probs @ model.emissionprob_

    return {
        "next_obs_distribution": next_obs_probs,
        "next_obs_predicted": int(np.argmax(next_obs_probs)),
        "next_obs_prob": float(np.max(next_obs_probs)),
    }
    
def predict_next_draw_char(model, encoding, seq): return {v: k for k, v in encoding.items()}[predict_next_observation(model, [encoding[c] for c in seq])['next_obs_predicted']]


def setup_hmm_models(ZA_results, AA2_results, ZA_encoding, AA2_encoding):
    """
    Setup HMM models with state titles and reordered transition matrices
    
    Args:
        ZA_results: List of ZA HMM results
        AA2_results: List of AA2 HMM results
        ZA_encoding: ZA encoding dictionary
        AA2_encoding: AA2 encoding dictionary
        
    Returns:
        Dictionary containing both models with their configurations
    """
    # Setup ZA model
    ZA_best_result = ZA_results[0]
    ZA_best_model = ZA_best_result['model']
    ZA_n_best = ZA_best_result['n_hidden_states']
    ZA_best_result["state_titles"] = {
        0: "Z (Newcomer)",
        1: "A (Applicant)",
        2: "K (Qualified)",
        3: "K/R (Year Check)",
    }

    reordered_ZA_transmat, ZA_state_order = reorder_hmm_matrix_by_emissions(
        ZA_best_model.transmat_,
        ZA_best_model.emissionprob_,
        ZA_encoding
    )

    # Setup AA2 model
    AA2_best_result = AA2_results[0]
    AA2_best_model = AA2_best_result['model']
    AA2_n_best = AA2_best_result['n_hidden_states']
    AA2_best_result["state_titles"] = {
        0: "A (Newcomer)",
        1: "A2 (Applicant)",
        2: "K (Qualified)",
        3: "R (Year Check)",
    }

    reordered_AA2_transmat, AA2_state_order = reorder_hmm_matrix_by_emissions(
        AA2_best_model.transmat_,
        AA2_best_model.emissionprob_,
        AA2_encoding
    )
    
    return {
        'ZA': {
            'result': ZA_best_result,
            'model': ZA_best_model,
            'n_states': ZA_n_best,
            'reordered_transmat': reordered_ZA_transmat,
            'state_order': ZA_state_order
        },
        'AA2': {
            'result': AA2_best_result,
            'model': AA2_best_model,
            'n_states': AA2_n_best,
            'reordered_transmat': reordered_AA2_transmat,
            'state_order': AA2_state_order
        }
    }

def train(ZA, AA2, ZA_encoding, AA2_encoding):
    # --- Train models ---
    #hidden range is 4+
    ZA_results = run_hmm_pipeline(ZA, ZA_encoding, "ZA", hidden_range=[4])
    AA2_results = run_hmm_pipeline(AA2, AA2_encoding, "AA2", hidden_range=[4])

    # --- Step 4: Print & Compare Results ---
    print_results(ZA_results, label="ZA Process")
    print_results(AA2_results, label="AA2 Process")

    # Store and restore trained models for later use
    hmm_models = setup_hmm_models(ZA_results, AA2_results, ZA_encoding, AA2_encoding)

    save_hmm_results("hmm_consequent_draw_processes_comparison_results.joblib", hmm_models)


if __name__ == "__main__":
    pass