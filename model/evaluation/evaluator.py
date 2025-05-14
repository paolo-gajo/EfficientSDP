"""
    This is a basic evaluation function that'd give evaluation of tagger and
    parser. For parser, it provides labeled and unlabeled evalution.
"""
from typing import List, Optional, Set, Dict

def filter_get_P_R_F1(gts, preds, ignore_list = []):
    """
        This function will remove all to be ignored labels
        and get prec, recall, F1
    """

    ## removing labels which are to be ignored
    preds = [pred for pred in preds if pred.split('-')[-1] not in ignore_list]    
    gts = [gt for gt in gts if gt.split('-')[-1] not in ignore_list]    

    ## calculating P,R,F1
    num_overlap = len([t for t in preds if t in gts])
    precision = num_overlap / len(preds) if len(preds) > 0 else 0.0
    recall = num_overlap / len(gts) if len(gts) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    # Jaccard-style accuracy: |pred ∩ gt| / |pred ∪ gt|
    union = len(gts)
    accuracy = num_overlap / union if union else 0.0

    return round(precision,4), round(recall, 4), round(f1, 4), round(accuracy, 4)

def compute_uas_las(
    model_output: List[Dict],
    ignore_labels: Optional[Set[str]] = None,
    missing_values: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Computes UAS and LAS from model_output.

    Args:
        model_output (List[Dict]): Each dict must contain keys:
            'words', 'head_indices_pred', 'head_tags_pred',
            'head_indices_gt', 'head_tags_gt'
        ignore_labels (Set[str], optional): Dependency labels to ignore (e.g., {"punct"}).
        missing_values (Set[str], optional): Values indicating missing gold labels (e.g., {"_", None}).

    Returns:
        Dict[str, float]: { "uas": ..., "las": ... }
    """
    if ignore_labels is None:
        ignore_labels = set()
    if missing_values is None:
        missing_values = set()

    total = 0
    uas_correct = 0
    las_correct = 0

    for elem in model_output:
        gold_heads = elem['head_indices_gt']
        gold_deps = elem['head_tags_gt']
        pred_heads = elem['head_indices_pred']
        pred_deps = elem['head_tags_pred']

        for head_gt, dep_gt, head_pred, dep_pred in zip(gold_heads, gold_deps, pred_heads, pred_deps):
            if dep_gt in missing_values or dep_gt in ignore_labels:
                continue

            total += 1
            if head_pred == head_gt:
                uas_correct += 1
                if dep_pred == dep_gt:
                    las_correct += 1

    return {
        "uas": uas_correct / total if total > 0 else None,
        "las": las_correct / total if total > 0 else None,
    }

def evaluate_model(model_output, label_to_class_map, ignore_tags=[], ignore_edges = ['0'], ignore_edge_labels=[]):
    """
        model_output: This is output of get_output_as_list_of_dicts() function, which contains
        information about both, ground truth and predictions. Must be a list.
        use_word_level: If True and we're in token mode, evaluation will be done at the word level
                        using the tokenizer's word_ids instead of at the token level.
    """
    
    token_id_field = 'words' if 'words' in model_output[0].keys() else 'input_ids'

    ## get indices of ignored tags/edges
    ignore_tag_indices = [str(label_to_class_map['tag2class'][tag]) for tag in ignore_tags]
    ignore_edge_indices = [str(label_to_class_map['edgelabel2class'][edge]) for edge in ignore_edge_labels]
    ignore_head_index = ignore_edges
    
    # print('Using original evaluation!')
    tagger_preds = [f'{i}-{j}-{word}-{pred_tag}' for i, elem in enumerate(model_output) for j, (word, pred_tag) in enumerate(zip(elem[token_id_field], elem['pos_tags_pred']))]
    tagger_gts = [f'{i}-{j}-{word}-{gt_tag}' for i, elem in enumerate(model_output) for j, (word, gt_tag) in enumerate(zip(elem[token_id_field], elem['pos_tags_gt']))]
    
    parser_labeled_pred = [f'{i}-{j}-{word}-{head_pred}-{edge_pred}' for i, elem in enumerate(model_output) for j, (word, edge_pred, head_pred) in enumerate(zip(elem[token_id_field], elem['head_tags_pred'], elem['head_indices_pred']))]
    parser_labeled_gt = [f'{i}-{j}-{word}-{head_gt}-{edge_gt}' for i, elem in enumerate(model_output) for j, (word, edge_gt, head_gt) in enumerate(zip(elem[token_id_field], elem['head_tags_gt'], elem['head_indices_gt']))]
    
    parser_unlabeled_pred = [f'{i}-{j}-{word}-{head_pred}' for i, elem in enumerate(model_output) for j, (word, head_pred) in enumerate(zip(elem[token_id_field], elem['head_indices_pred']))]
    parser_unlabeled_gt = [f'{i}-{j}-{word}-{head_gt}' for i, elem in enumerate(model_output) for j, (word, head_gt) in enumerate(zip(elem[token_id_field], elem['head_indices_gt']))]

    # Calculate metrics
    tagger_results = {}
    tagger_results['P'], tagger_results['R'], tagger_results['F1'], tagger_results['acc'] = filter_get_P_R_F1(tagger_gts, tagger_preds, ignore_list=ignore_tag_indices)
    
    parser_labeled_results = {}
    parser_labeled_results['P'], parser_labeled_results['R'], parser_labeled_results['F1'], parser_labeled_results['acc'] = filter_get_P_R_F1(parser_labeled_gt, parser_labeled_pred, ignore_list=ignore_edge_indices)
    
    parser_unlabeled_results = {}
    parser_unlabeled_results['P'], parser_unlabeled_results['R'], parser_unlabeled_results['F1'], parser_unlabeled_results['acc'] = filter_get_P_R_F1(parser_unlabeled_gt, parser_unlabeled_pred, ignore_list=ignore_head_index)
    uas_las_results = compute_uas_las(model_output, ignore_labels=[int(el) for el in ignore_edge_indices], missing_values={"_", None})

    return {
        'tagger_results': tagger_results,
        'parser_labeled_results': parser_labeled_results,
        'parser_unlabeled_results': parser_unlabeled_results,
        'uas_las_results': uas_las_results,
    }