"""
    This is a basic evaluation function that'd give evaluation of tagger and
    parser. For parser, it provides labeled and unlabeled evalution.
"""


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

    return round(precision,4), round(recall, 4), round(f1, 4)

# def evaluate_model(model_output, label_to_class_map, ignore_tags = [], ignore_edges = []):
#     """
#         model_output: This is output of get_output_as_list_of_dicts() function, which contains
#         information about both, ground truth and predictions. Must be a list.
#     """

#     token_id_field = 'words' if 'words' in model_output[0].keys() else 'input_ids'

#     ## get indices of ignored tags/edges
#     ignore_tag_indices = [str(label_to_class_map['tag2class'][tag]) for tag in ignore_tags]
#     ignore_edge_indices = [str(label_to_class_map['edgelabel2class'][edge]) for edge in ignore_edges]
#     ignore_head_index = ['0']

#     ## tagger's output
#     tagger_preds = [ f'{i}-{j}-{word}-{pred_tag}' for i, elem in enumerate(model_output) for j, (word, pred_tag) in enumerate(zip(elem[token_id_field], elem['pos_tags_pred']))]
#     tagger_gts = [ f'{i}-{j}-{word}-{gt_tag}' for i, elem in enumerate(model_output) for j, (word, gt_tag) in enumerate(zip(elem[token_id_field], elem['pos_tags_gt'])) ]
#     tagger_results = {}
#     tagger_results['P'], tagger_results['R'], tagger_results['F1'] = filter_get_P_R_F1(tagger_gts, tagger_preds, ignore_list = ignore_tag_indices)
    
#     ## parser labeled output
#     parser_labeled_pred = [ f'{i}-{j}-{word}-{head_pred}-{edge_pred}' for i, elem in enumerate(model_output) for j, (word, edge_pred, head_pred) in enumerate(zip(elem[token_id_field], elem['head_tags_pred'], elem['head_indices_pred']))]
#     parser_labeled_gt = [ f'{i}-{j}-{word}-{head_gt}-{edge_gt}' for i, elem in enumerate(model_output) for j, (word, edge_gt, head_gt) in enumerate(zip(elem[token_id_field], elem['head_tags_gt'], elem['head_indices_gt'])) ]
#     parser_labeled_results = {}
#     parser_labeled_results['P'], parser_labeled_results['R'], parser_labeled_results['F1'] = filter_get_P_R_F1(parser_labeled_gt, parser_labeled_pred, ignore_list = ignore_edge_indices)

#     ## parser unlabeled output
#     parser_unlabeled_pred = [ f'{i}-{j}-{word}-{head_pred}' for i, elem in enumerate(model_output) for j, (word, head_pred) in enumerate(zip(elem[token_id_field], elem['head_indices_pred']))]
#     parser_unlabeled_gt = [ f'{i}-{j}-{word}-{head_gt}' for i, elem in enumerate(model_output) for j, (word, head_gt) in enumerate(zip(elem[token_id_field], elem['head_indices_gt']))]
#     parser_unlabeled_results = {}
#     parser_unlabeled_results['P'], parser_unlabeled_results['R'], parser_unlabeled_results['F1'] = filter_get_P_R_F1(parser_unlabeled_gt, parser_unlabeled_pred, ignore_list = ignore_head_index)

#     return {'tagger_results' :  tagger_results, 'parser_labeled_results': parser_labeled_results, 'parser_unlabeled_results' : parser_unlabeled_results}

def evaluate_model(model_output, label_to_class_map, ignore_tags=[], ignore_edges=[], use_word_level = False):
    """
        model_output: This is output of get_output_as_list_of_dicts() function, which contains
        information about both, ground truth and predictions. Must be a list.
        use_word_level: If True and we're in token mode, evaluation will be done at the word level
                        using the tokenizer's word_ids instead of at the token level.
    """
    
    token_id_field = 'words' if 'words' in model_output[0].keys() else 'input_ids'
    is_token_mode = token_id_field == 'input_ids'
    
    ## get indices of ignored tags/edges
    ignore_tag_indices = [str(label_to_class_map['tag2class'][tag]) for tag in ignore_tags]
    ignore_edge_indices = [str(label_to_class_map['edgelabel2class'][edge]) for edge in ignore_edges]
    ignore_head_index = ['0']
    
    # If we're in token mode and want word-level evaluation, we need to aggregate tokens into words
    if is_token_mode and use_word_level and 'word_ids_custom' in model_output[0]:
        print('Using word-level token majority evaluation!')
        # Process each sample for word-level evaluation
        tagger_gts = []
        tagger_preds = []
        parser_labeled_gt = []
        parser_labeled_pred = []
        parser_unlabeled_gt = []
        parser_unlabeled_pred = []
        
        for i, elem in enumerate(model_output):
            # Group tokens by word_id
            word_groups = {}
            word_ids = [el if el != -100 else None for el in elem['word_ids_custom']]
            for j, word_id in enumerate(word_ids):
                if word_id is not None:  # Skip special tokens (None word_id)
                    if word_id not in word_groups:
                        word_groups[word_id] = []
                    word_groups[word_id].append(j)
            
            # For each word, aggregate token predictions using majority voting
            for word_id, token_indices in word_groups.items():
                # Get the word (use the first token's text as representative)
                word = elem[token_id_field][token_indices[0]]
                
                # Tagger: majority vote for POS tag
                gt_tags = [elem['pos_tags_gt'][j] for j in token_indices]
                pred_tags = [elem['pos_tags_pred'][j] for j in token_indices]
                gt_tag = max(set(gt_tags), key=gt_tags.count)  # Most common tag
                pred_tag = max(set(pred_tags), key=pred_tags.count)  # Most common tag
                
                tagger_gts.append(f'{i}-{word_id}-{word}-{gt_tag}')
                tagger_preds.append(f'{i}-{word_id}-{word}-{pred_tag}')
                
                # Parser: majority vote for head indices and edge labels
                gt_edges = [elem['head_tags_gt'][j] for j in token_indices]
                pred_edges = [elem['head_tags_pred'][j] for j in token_indices]
                gt_heads = [elem['head_indices_gt'][j] for j in token_indices]
                pred_heads = [elem['head_indices_pred'][j] for j in token_indices]
                
                gt_edge = max(set(gt_edges), key=gt_edges.count)
                pred_edge = max(set(pred_edges), key=pred_edges.count)
                gt_head = max(set(gt_heads), key=gt_heads.count)
                pred_head = max(set(pred_heads), key=pred_heads.count)
                
                parser_labeled_gt.append(f'{i}-{word_id}-{word}-{gt_head}-{gt_edge}')
                parser_labeled_pred.append(f'{i}-{word_id}-{word}-{pred_head}-{pred_edge}')
                parser_unlabeled_gt.append(f'{i}-{word_id}-{word}-{gt_head}')
                parser_unlabeled_pred.append(f'{i}-{word_id}-{word}-{pred_head}')
    else:
        print('Using original evaluation!')
        tagger_preds = [f'{i}-{j}-{word}-{pred_tag}' for i, elem in enumerate(model_output) for j, (word, pred_tag) in enumerate(zip(elem[token_id_field], elem['pos_tags_pred']))]
        tagger_gts = [f'{i}-{j}-{word}-{gt_tag}' for i, elem in enumerate(model_output) for j, (word, gt_tag) in enumerate(zip(elem[token_id_field], elem['pos_tags_gt']))]
        
        parser_labeled_pred = [f'{i}-{j}-{word}-{head_pred}-{edge_pred}' for i, elem in enumerate(model_output) for j, (word, edge_pred, head_pred) in enumerate(zip(elem[token_id_field], elem['head_tags_pred'], elem['head_indices_pred']))]
        parser_labeled_gt = [f'{i}-{j}-{word}-{head_gt}-{edge_gt}' for i, elem in enumerate(model_output) for j, (word, edge_gt, head_gt) in enumerate(zip(elem[token_id_field], elem['head_tags_gt'], elem['head_indices_gt']))]
        
        parser_unlabeled_pred = [f'{i}-{j}-{word}-{head_pred}' for i, elem in enumerate(model_output) for j, (word, head_pred) in enumerate(zip(elem[token_id_field], elem['head_indices_pred']))]
        parser_unlabeled_gt = [f'{i}-{j}-{word}-{head_gt}' for i, elem in enumerate(model_output) for j, (word, head_gt) in enumerate(zip(elem[token_id_field], elem['head_indices_gt']))]
    
    # Calculate metrics
    tagger_results = {}
    tagger_results['P'], tagger_results['R'], tagger_results['F1'] = filter_get_P_R_F1(tagger_gts, tagger_preds, ignore_list=ignore_tag_indices)
    
    parser_labeled_results = {}
    parser_labeled_results['P'], parser_labeled_results['R'], parser_labeled_results['F1'] = filter_get_P_R_F1(parser_labeled_gt, parser_labeled_pred, ignore_list=ignore_edge_indices)
    
    parser_unlabeled_results = {}
    parser_unlabeled_results['P'], parser_unlabeled_results['R'], parser_unlabeled_results['F1'] = filter_get_P_R_F1(parser_unlabeled_gt, parser_unlabeled_pred, ignore_list=ignore_head_index)
    
    return {
        'tagger_results': tagger_results,
        'parser_labeled_results': parser_labeled_results,
        'parser_unlabeled_results': parser_unlabeled_results
    }

def evaluate_model_with_all_labels(model_output, label_to_class_map, ignore_tags = [], ignore_edges = []):
    """
        model_output: we evaluate the model with all the labels except "no_label", if it exists. 
        So anything that comes to ignore_edges and ignore_tags variables, will be unused!
    """
    ## get indices of ignored tags/edges
    ignore_tag_indices = [str(label_to_class_map['tag2class'][tag]) for tag in ['no_label']]
    ignore_edge_indices = [str(label_to_class_map['edgelabel2class'][edge]) for edge in ['no_label']]
    ignore_head_index = []

    ## tagger's output
    tagger_preds = [ f'{i}-{j}-{word}-{pred_tag}' for i, elem in enumerate(model_output) for j, (word, pred_tag) in enumerate(zip(elem['words'], elem['pos_tags_pred']))]
    tagger_gts = [ f'{i}-{j}-{word}-{gt_tag}' for i, elem in enumerate(model_output) for j, (word, gt_tag) in enumerate(zip(elem['words'], elem['pos_tags_gt'])) ]
    tagger_results = {}
    tagger_results['P'], tagger_results['R'], tagger_results['F1'] = filter_get_P_R_F1(tagger_gts, tagger_preds, ignore_list = ignore_tag_indices)
    
    ## parser labeled output
    parser_labeled_pred = [ f'{i}-{j}-{word}-{head_pred}-{edge_pred}' for i, elem in enumerate(model_output) for j, (word, edge_pred, head_pred) in enumerate(zip(elem['words'], elem['head_tags_pred'], elem['head_indices_pred']))]
    parser_labeled_gt = [ f'{i}-{j}-{word}-{head_gt}-{edge_gt}' for i, elem in enumerate(model_output) for j, (word, edge_gt, head_gt) in enumerate(zip(elem['words'], elem['head_tags_gt'], elem['head_indices_gt'])) ]
    parser_labeled_results = {}
    parser_labeled_results['P'], parser_labeled_results['R'], parser_labeled_results['F1'] = filter_get_P_R_F1(parser_labeled_gt, parser_labeled_pred, ignore_list = ignore_edge_indices)

    ## parser unlabeled output
    parser_unlabeled_pred = [ f'{i}-{j}-{word}-{head_pred}' for i, elem in enumerate(model_output) for j, (word, head_pred) in enumerate(zip(elem['words'], elem['head_indices_pred']))]
    parser_unlabeled_gt = [ f'{i}-{j}-{word}-{head_gt}' for i, elem in enumerate(model_output) for j, (word, head_gt) in enumerate(zip(elem['words'], elem['head_indices_gt']))]
    parser_unlabeled_results = {}
    parser_unlabeled_results['P'], parser_unlabeled_results['R'], parser_unlabeled_results['F1'] = filter_get_P_R_F1(parser_unlabeled_gt, parser_unlabeled_pred, ignore_list = ignore_head_index)

    return {'tagger_results' :  tagger_results, 'parser_labeled_results': parser_labeled_results, 'parser_unlabeled_results' : parser_unlabeled_results}
