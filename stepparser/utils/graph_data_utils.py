from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from typing import List, Dict, Tuple, Set
from stepparser.utils.sys_utils import load_json
from networkx import DiGraph, all_topological_sorts, from_edgelist
import numpy as np
import random
from tqdm.auto import tqdm
from collections import OrderedDict
from copy import deepcopy
import matplotlib.pyplot as plt

def check_io(func):
    def wrapper(self, G, order_idx, *args, **kwargs):
        # Store original data for comparison
        words_orig = G['words']
        head_indices_orig = G['head_indices'].tolist()
        step_indices_orig = G['step_indices'].tolist()
        
        # Call the original permute_graph function
        G_perm = func(self, G, order_idx, *args, **kwargs)
        
        # Get permuted data
        words_perm = G_perm['words']
        head_indices_perm = G_perm['head_indices'].tolist()
        step_indices_perm = G_perm['step_indices'].tolist()
        
        # Determine the number of non-padding items to display
        valid_items_orig = [i for i, word in enumerate(words_orig) if word != '0']
        valid_items_perm = [i for i, word in enumerate(words_perm) if word != '0']
        max_items = max(len(valid_items_orig), len(valid_items_perm))
        
        # Print header
        print("=" * 160)
        print(f"{'ORIGINAL GRAPH':<80} | {'PERMUTED GRAPH (Order: ' + str(order_idx.tolist()) + ')'}")
        print("=" * 160)
        print(f"{'Index':<6}{'Word':<15}{'Head Index':<12}{'Points To':<15}{'Step':<6} | {'Index':<6}{'Word':<15}{'Head Index':<12}{'Points To':<15}{'Step':<6}")
        print("-" * 160)
        
        # Build rows for both graphs
        orig_rows = []
        for i, (word, head_idx, step_idx) in enumerate(zip(words_orig, head_indices_orig, step_indices_orig)):
            if word != '0':  # Skip padding
                pointing_to = words_orig[head_idx-1] if head_idx > 0 else "ROOT"
                orig_rows.append(f"{i+1:<6}{word:<15}{head_idx:<12}{pointing_to:<15}{step_idx:<6}")
        
        perm_rows = []
        for i, (word, head_idx, step_idx) in enumerate(zip(words_perm, head_indices_perm, step_indices_perm)):
            if word != '0':  # Skip padding
                pointing_to = words_perm[head_idx-1] if head_idx > 0 else "ROOT"
                perm_rows.append(f"{i+1:<6}{word:<15}{head_idx:<12}{pointing_to:<15}{step_idx:<6}")
        
        # Print rows side by side
        for i in range(max_items):
            orig_row = orig_rows[i] if i < len(orig_rows) else " " * 54
            perm_row = perm_rows[i] if i < len(perm_rows) else ""
            print(f"{orig_row} | {perm_row}")
        
        return G_perm
    
    return wrapper

class GraphDataset(Dataset):
    def __init__(self,
                 data: List[Dict[str, str]] = None,
                 tokenizer = None,
                 padding = True,
                 split = None,
                 shuffle = False,
                 ):
        self.ignore_keys = ['edge_index', 'adj_m', 'deg_m', 'lap_m']
        self.tokenizer = tokenizer
        self.padding = padding
        if not self.padding:
            raise NotImplementedError('Padding must be enabled.')
        self.label_index_map = get_mappings(data)

        if shuffle:
            random.shuffle(data)
        self.data = self.preprocess_data(data)
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def from_path(cls, path, **kwargs):
        if isinstance(path, str):
            data = load_json(path)
        elif isinstance(path, list):
            data = []
            for el in path:
                data += load_json(el)
        return cls(data, **kwargs)

    def augment(self, k = 1, keep_og = False):
        augmented_data = []
        for sample in tqdm(self.data, total=len(self.data), desc=f'Augmenting {self.split} dataset...'):
            if keep_og:
                augmented_data.append(sample)
            sample_size = k
            if len(sample['step_graph']) > 0:
                G = from_edgelist(sample['step_graph'], create_using=DiGraph)
                all_topos = np.array(list(all_topological_sorts(G)))
                num_topos = all_topos.shape[0]
                if num_topos > 1:
                    all_topos = all_topos[1:]  # the 0th element is the base order, so we exclude it
                    if sample_size > len(all_topos):
                        sample_size = len(all_topos)
                    perms_mask = np.random.choice(len(all_topos), size = sample_size, replace=False)
                    perms = all_topos[perms_mask].tolist()
                    for i, perm in enumerate(perms):
                        perm_filled = torch.as_tensor(add_isolated_nodes(perm))
                        # NOTE: we +1 the index nodes because the step indices start from 1
                        # but we need edge_index (and hence the step_graph from which it originates)
                        # to start from 0
                        permuted_graph = self.permute_graph(sample, perm_filled + 1)
                        augmented_data.append(permuted_graph)
                    
        print(f'Augmented {self.split} dataset from {len(self.data)} to {len(augmented_data)} samples.')
        # print(f"Augmented value counts: {dict(pd.DataFrame(augmented_data)['permuted'].value_counts())}")
        self.data = augmented_data
    
    def shuffle(self):
        random.shuffle(self.data)

    def only_use_max_step_graph(self, threshold = 0):
        max = 0
        max_step_sample = None
        for sample in self.data:
            if len(sample['step_graph']) > 0:
                G = from_edgelist(sample['step_graph'], create_using=DiGraph)
                all_topos = list(all_topological_sorts(G))
                n_all_topos = len(all_topos)
                if threshold and n_all_topos > threshold:
                    max = n_all_topos
                    max_step_sample = sample
                    break
                if n_all_topos > max:
                    max = n_all_topos
                    max_step_sample = sample
        self.data = [max_step_sample]

    def plot_topological_sorts_histogram(self, savename: str = 'hist.pdf', column_threshold: int = 0):
        """
        Calculates the number of topological sortings for each sample's step graph 
        and plots a histogram. Only bins (columns) with a frequency (number of samples)
        greater than or equal to column_threshold are shown, and they are evenly spaced.
        
        Parameters:
            column_threshold (int): Minimum frequency for a histogram column to be displayed.
        """
        topo_counts = []
        for sample in self.data:
            if sample['step_graph']:
                G = from_edgelist(list(sample['step_graph']), create_using=DiGraph)
                try:
                    n_topos = len(list(all_topological_sorts(G)))
                except Exception as e:
                    print(f"Error computing topological sorts for a sample: {e}")
                    n_topos = 0
                topo_counts.append(n_topos)
        
        if not topo_counts:
            print("No valid step graphs found in the dataset for topological sorting.")
            return

        # Compute frequency counts for each number of topological sorts.
        freq = Counter(topo_counts)
        
        # Filter out bins with frequency below the specified threshold.
        filtered_freq = {k: v for k, v in freq.items() if v >= column_threshold}
        
        if not filtered_freq:
            print(f"No histogram columns have a frequency >= {column_threshold}.")
            return
        
        # Sort the bins for consistent plotting.
        bins = sorted(filtered_freq.keys())
        frequencies = [filtered_freq[k] for k in bins]
        
        # Create evenly spaced positions for the bins.
        x_positions = np.arange(len(bins))
        
        plt.figure(figsize=(10, 6))
        plt.bar(x_positions, frequencies, edgecolor='black', align='center')
        plt.xlabel('Number of Topological Sortings')
        plt.ylabel('Number of Samples')
        plt.title(f'Histogram of Topological Sorting Counts (Columns with frequency >= {column_threshold})')
        # Set x-ticks using the evenly spaced positions, but label them with the actual bin values.
        plt.xticks(x_positions, [str(b) for b in bins], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(savename, format='pdf')


    # @check_io
    def permute_graph(self, G, order_idx):
        step_indices = G['step_indices']
        step_indices_tokens = G['step_indices_tokens']
        attention_mask = G['encoded_input']['attention_mask']
        words_mask_custom = G['encoded_input']['words_mask_custom']

        G_perm = deepcopy(G)
        G_perm.pop('encoded_input')
        bypass_keys = []
        for key in self.ignore_keys:
            bypass_keys.append({key: G_perm.pop(key)})
        
        for key, value in G_perm.items():    
            idx = step_indices_tokens if ('tokens' in key or 'encoded_input' == key) else step_indices
            cutoff = sum(attention_mask) if ('tokens' in key or 'encoded_input' == key) else sum(words_mask_custom)
            if key not in ['head_indices', 'head_indices_tokens']:
                if is_tensorizable(value):
                    G_perm[key] = apply_sub_dicts(value, lambda x: reorder_tensor(x, idx=idx, permutation=order_idx))[:cutoff]
                elif key != 'step_graph':
                    G_perm[key] = reorder_list(value, idx=idx, permutation=order_idx)[:cutoff]
        
        for key, value in G_perm.items():
            idx = step_indices_tokens if ('tokens' in key or 'encoded_input' == key) else step_indices
            if key in ['head_indices', 'head_indices_tokens']:
                src_to_tgt = OrderedDict()
                src_to_tgt.update({k: (int(k + 1), int(v)) for k, v in enumerate(G_perm[key])})
                src_to_tgt = reorder_list(src_to_tgt, idx=idx, permutation=order_idx)
                h_idx_perm = []
                for k1, v1 in enumerate(src_to_tgt):
                    if v1[1] == 0:
                        h_idx_perm.append(0)
                    else:
                        for k2, v2 in enumerate(src_to_tgt):
                            if v1[1] == v2[0]:
                                h_idx_perm.append(k2 + 1)
                G_perm[key] = torch.tensor(h_idx_perm)
        
        encoding = self.tokenizer(G_perm['words'],
                                    is_split_into_words = True,
                                    return_tensors = 'pt',
                                    # padding = 'max_length' if self.padding else False,
                                    )
        word_ids = torch.as_tensor([elem if elem is not None else -100 for elem in encoding.word_ids()])
        words_mask_custom = torch.as_tensor([1 for _ in range(len(G_perm['words']))])
        encoding.update({'words_mask_custom': words_mask_custom, 'word_ids_custom': word_ids})
        G_perm['encoded_input'] = encoding
        G_perm = apply_sub_dicts(G_perm, self.tensorize)
        G_perm = apply_sub_dicts(G_perm, self.pad)

        for ignored_dict in bypass_keys:
            G_perm.update(ignored_dict)
        return G_perm

    def preprocess_data(self, data):
        # turns all fields into appropriate tensors
        processed_data = []
        for sample in data:
            for key in ['sent_indices', 'word_sent_indices']:
                sample.pop(key)
            processed_sample = {}
            processed_sample.update(sample)
            encoding = self.tokenizer(sample['words'],
                                        is_split_into_words = True,
                                        return_tensors = 'pt',
                                        # padding = 'max_length' if self.padding else False,
                                        )
            word_ids = torch.as_tensor([elem if elem is not None else -100 for elem in encoding.word_ids()])
            words_mask_custom = torch.as_tensor([1 for _ in range(len(sample['words']))])
            encoding.update({'words_mask_custom': words_mask_custom, 'word_ids_custom': word_ids})
            processed_sample['encoded_input'] = encoding
            processed_sample['step_indices_tokens'] = self.convert_to_token_indices(sample['step_indices'], word_ids)
            processed_sample['pos_tags'] = torch.tensor([self.label_index_map['tag2class'][el] for el in sample['pos_tags']])
            processed_sample['pos_tags_tokens'] = self.convert_to_token_indices(processed_sample['pos_tags'], word_ids)
            processed_sample['head_tags'] = torch.tensor([self.label_index_map['edgelabel2class'][el] for el in sample['head_tags']])
            processed_sample['head_tags_tokens'] = self.convert_to_token_indices(processed_sample['head_tags'], word_ids)
            processed_sample['head_indices_tokens'] = self.convert_to_token_indices(processed_sample['head_indices'], word_ids)
            # processed_sample['edge_index'] = torch.as_tensor([[head, tail] for head, tail in enumerate(sample['head_indices'])])
            # processed_sample['edge_index_steps'] = torch.as_tensor([[head, tail] for head, tail in enumerate(sample['step_indices'])])
            # processed_sample['edge_index_tokens'] = torch.as_tensor([[head, tail] for head, tail in enumerate(processed_sample['head_indices_tokens'])])
            # processed_sample['edge_index_steps_tokens'] = torch.as_tensor([[head, tail] for head, tail in enumerate(processed_sample['step_indices_tokens'])])
            processed_sample = apply_sub_dicts(processed_sample, self.tensorize)
            processed_sample = apply_sub_dicts(processed_sample, self.pad)
            processed_sample['step_graph'] = self.get_step_graph(sample)
            processed_sample['edge_index'] = graph_to_edge_index(processed_sample['step_graph'])
            processed_sample['adj_m'] = edge_index_to_adj_matrix(processed_sample['edge_index'])
            processed_sample['deg_m'] = get_deg_matrix(processed_sample['adj_m'])
            processed_sample['lap_m'] = get_graph_laplacian(processed_sample['deg_m'], processed_sample['adj_m'])
            processed_data.append(processed_sample)
        return processed_data

    def convert_to_token_indices(self, input: list, word_ids: torch.tensor):
        return torch.tensor([input[el] if el != -100 else 0 for el in word_ids], dtype=torch.long)

    def get_step_graph(self, sample):
        step_indices = torch.as_tensor(sample['step_indices']) - 1
        head_indices = torch.as_tensor(sample['head_indices']) - 1
        target_steps = torch.cat([torch.tensor([0]), step_indices])[head_indices]
        G_loops = torch.vstack([step_indices, target_steps])
        mask_steps = torch.where(G_loops[0] != G_loops[1], True, False) # filter out edges within the same step
        G_masked = G_loops[:, mask_steps]
        mask_zeros = torch.where(G_masked[1] != 0, True, False) # mask again to remove edges going to the R00T
        G_masked = G_masked[:, mask_zeros].T.tolist()
        G = [tuple(sorted([el[0], el[1]])) for el in G_masked]
        G = set(sorted(G, key=lambda x: x[0]))
        return G

    def pad_zeros(self, t):
        if isinstance(t, torch.Tensor):
            t = t.squeeze()
            if len(t.shape) == 1:
                t = t.unsqueeze(1)
                padding_zeros = torch.zeros((self.tokenizer.model_max_length - t.shape[0], t.shape[1]), dtype=t.dtype)
                # use the same type for the 0s so the tensor doesn't change type
                t_padded = torch.cat([t, padding_zeros]).squeeze()
                return t_padded
            else:
                padding_zeros = torch.zeros((self.tokenizer.model_max_length - t.shape[0], t.shape[1]), dtype=t.dtype)
                t_padded = torch.cat([t, padding_zeros], dim = 0)
                return t_padded
        elif isinstance(t, list):
            len_padding = self.tokenizer.model_max_length - len(t)
            padding_zeros = [0] * len_padding
            type_internal = type(t[0])
            return t + [type_internal(el) for el in padding_zeros] 

    def tensorize(self, data):
        if is_tensorizable(data):
            return torch.as_tensor(data)
        else:
            return data

    def pad(self, t):
        if is_paddable(t) and self.padding:
            return self.pad_zeros(t)
        else:
            return t
    
    def get_original_words(self, input_ids, word_ids):
        # Get the token text for each input_id
        word_ids = [el if el != -100 else None for el in word_ids]
        tokens = [self.tokenizer.convert_ids_to_tokens([id])[0] for id in input_ids.squeeze()]
        
        # Find the maximum word_id to determine the number of original words
        max_word_id = max(filter(lambda x: x is not None, word_ids))
        
        # Initialize a list to store the original words
        original_words = ["" for _ in range(max_word_id + 1)]
        
        # Reconstruct each word from its tokens
        for token, word_id in zip(tokens, word_ids):
            if word_id is not None:  # Skip special tokens (CLS, SEP, etc.)
                # If the token starts with ##, it's a subword continuation
                if token.startswith("##"):
                    original_words[word_id] += token[2:]  # Remove ## prefix
                else:
                    original_words[word_id] += token
        
        return original_words

def graph_to_edge_index(graph: Set[Tuple]):
    if not (len(graph)) > 0:
        return torch.empty(0)
    return torch.tensor(list(graph), dtype=torch.long).T

def edge_index_to_adj_matrix(edge_index: torch.Tensor):
    if not edge_index.numel() > 0:
        return edge_index
    edge_index = edge_index
    k = torch.max(edge_index) + 1
    adj = torch.zeros((k, k), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    return adj

def get_deg_matrix(adj_matrix: torch.Tensor):
    N = adj_matrix.shape[0]
    degs = []
    for i in range(N):
        degs.append(sum(adj_matrix[i]))
    deg_matrix = torch.diag(torch.tensor(degs))
    return deg_matrix

def get_graph_laplacian(deg_m: torch.Tensor, adj_m: torch.Tensor):
    L = deg_m - adj_m
    return L

def reorder_tensor(t: torch.Tensor = None, idx = None, permutation = None):
    '''
    Produces a tensor where the elements are permuted based on an idx.
    Example input:
    permutation = torch.tensor([3, 1, 2, 4])
    idx = torch.tensor([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Returns:
        >>> torch.tensor([0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    '''
    idx_reordered = torch.arange(len(idx))
    idx_tensor = []
    for perm in permutation:
        perm_idx_tensor = torch.where(idx == perm)[0]
        idx_tensor += perm_idx_tensor
    left = min(idx_tensor)
    right = max(idx_tensor)
    idx_reordered[left:right+1] = torch.as_tensor(idx_tensor)
    t_reordered = t[idx_reordered]
    return t_reordered

def reorder_list(L: torch.Tensor = None, idx = None, permutation = None):
    '''
    Produces a tensor where the elements are permuted based on an idx.
    Example input:
    permutation = torch.tensor([3, 1, 2, 4])
    idx = torch.tensor([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Returns:
        >>> torch.tensor([0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    '''
    permutation = permutation.tolist()
    idx_original = list(range(len(idx)))
    idx_permuted = []
    for perm in permutation:
        perm_idx_tensor = [i for i, el in enumerate(idx) if el == perm]
        idx_permuted += perm_idx_tensor
    left = min(idx_permuted)
    right = max(idx_permuted)
    idx_original[left:right+1] = idx_permuted
    L_reordered = [L[i] for i in idx_original]
    return L_reordered

def add_isolated_nodes(L):
    min_val = min(L)
    max_val = max(L)

    # Find the missing numbers in the range
    full_range = set(range(min_val, max_val + 1))
    missing_values = list(full_range - set(L))

    # Insert each missing value at a random position
    for value in missing_values:
        random_index = random.randint(0, len(L))
        L.insert(random_index, value)
    return L

def get_mappings(data):
    all_pos_tags = []
    all_head_tags = []

    for line in data:
        all_pos_tags += line['pos_tags']
        all_head_tags += line['head_tags']
    
    # Count frequency of each tag
    pos_tag_counts = {}
    for tag in all_pos_tags:
        pos_tag_counts[tag] = pos_tag_counts.get(tag, 0) + 1
    
    head_tag_counts = {}
    for tag in all_head_tags:
        head_tag_counts[tag] = head_tag_counts.get(tag, 0) + 1
    
    # Sort by frequency (highest to lowest)
    sorted_pos_tags = sorted(pos_tag_counts.items(), key=lambda item: item[1], reverse=True)
    sorted_head_tags = sorted(head_tag_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Create mappings (index 0 is reserved for 'no_label' in POS tags)
    pos_tags_map = {tag: i+1 for i, (tag, _) in enumerate(sorted_pos_tags)}
    pos_tags_map.update({'no_label': 0})
    
    # Head tags start at index 0 (no special reserved index)
    head_tags_map = {tag: i for i, (tag, _) in enumerate(sorted_head_tags)}
    
    # Return in the unified format
    return {'tag2class': pos_tags_map, 'edgelabel2class': head_tags_map}

def apply_sub_dicts(data, func):
    if hasattr(data, 'keys'):
        return {key: apply_sub_dicts(value, func) for key, value in data.items()}
    else:
        return func(data)

def adj_list_2_edge_index(L):
    edge_index = np.array([list(el) for el in L]).T
    return edge_index

def is_paddable(L):
    if isinstance(L, list):
        return True
    elif isinstance(L, torch.Tensor):
        return True
    elif isinstance(L, set):
        return False
    elif hasattr(L, 'keys'):
        return all([is_tensorizable(L[key]) for key in L.keys()])
    else:
        raise NotImplementedError('Not a list, tensor, set, or dict-like.')

def is_tensorizable(L):
    if isinstance(L, list):
        if len(L) == 0 or any([isinstance(el, str) for el in L]) or any([isinstance(el, set) for el in L]):
            return False
        else:
            return True
    elif isinstance(L, torch.Tensor):
        return True
    elif isinstance(L, set):
        return False
    elif hasattr(L, 'keys'):
        return all([is_tensorizable(L[key]) for key in L.keys()])
    else:
        raise NotImplementedError('Not a list, tensor, set, or dict-like.')

class GraphCollator:
    def __init__(self, keys_to_filter = ['words', 'step_graph', 'edge_index', 'adj_m', 'deg_m', 'lap_m'], truncate_to_longest = True):
        self.keys_to_filter = keys_to_filter
        self.truncate_to_longest = truncate_to_longest
    
    def collate(self, input):
        out, filtered = self.filter_keys(input)
        if self.truncate_to_longest:
            max = self.get_trunc_len(input)
            out = self.truncate(out, max)
        out = default_collate(out)
        for key in self.keys_to_filter:
            out[key] = [el[key] for el in filtered]
        return out
    
    def truncate(self, batch, max):
        for i in range(len(batch)):
            for key, value in batch[i].items():
                if isinstance(value, torch.Tensor):
                    batch[i][key] = batch[i][key][:max]
            for key, value in batch[i]['encoded_input'].items():
                batch[i]['encoded_input'][key] = batch[i]['encoded_input'][key][:max]
        return batch


    def get_trunc_len(self, batch):
        max = 0
        for el in batch:
            input_ids = el['encoded_input']['input_ids']
            new = len(input_ids[torch.where(input_ids != 0)])
            if new > max:
                max = new
        return max

    def filter_keys(self, input):
        batch = []
        filtered = []
        for el in input:
            out_el = {}
            filtered_el = {}
            for key, value in el.items():
                if key not in self.keys_to_filter:
                    out_el[key] = value
                else:
                    filtered_el[key] = value
            batch.append(out_el)
            filtered.append(filtered_el)
        return batch, filtered

def transformer_input_filter(input, keys = ['input_ids', 'token_type_ids', 'attention_mask']):
    return {key: input[key] for key in keys}

def build_dataloaders(dataset_dict, collate_fn, batch_size = 1, splits = ['train', 'val', 'test']):
    return {
        split: DataLoader(dataset_dict[split],
                   batch_size=batch_size,
                   collate_fn=collate_fn
                   ) for split in splits
        }
def get_max_steps(data: List):
    max_steps = 0
    for line in data:
        tmp = max(line['step_indices'])
        if tmp > max_steps:
            max_steps = tmp
    return max_steps

def main():
    pass

if __name__ == "__main__":
    main()
