import torch
import os
from model.utils import (save_json, 
                              build_dataloader,  
                              setup_config, 
                              build_model, 
                              get_args, 
                              set_seeds,
                              train_epoch,
                              run_evaluation,
                              load_json,
                              save_python_command,
                              save_reproduce_training_cmd,
                              )
from model.utils.train_utils import get_scheduler
from model.config import default_cfg, custom_config
from model.evaluation import evaluate_model
from model.utils.graph_data_utils import get_mappings
import numpy as np
import sys
import json
from copy import deepcopy
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
def main():

    # torch.set_printoptions(linewidth=10000, threshold=10000)

    # Get the arguments and set up configuration
    args = get_args()
    config = setup_config(default_cfg, args=args, custom_config=custom_config)

    print('Config:\n\n', json.dumps(config, indent=4))
    print('Args:\n\n', json.dumps(args, indent=4))
    
    # Set seeds and show save directory
    set_seeds(config['seed'])
    print(f"Will save to: {config['save_dir']}")
    
    summary = {'config': config}
    save_json(summary, os.path.join(config['save_dir'], 'config.json'))
    cmd_file = os.path.join(config['save_dir'], 'train_command.txt')
    save_python_command(cmd_file, sys.argv)
    reproduce_training_cmd_file = os.path.join(config['save_dir'], 'full_train_reproduce_cmd.txt')
    save_reproduce_training_cmd(sys.argv[0], config, args, reproduce_training_cmd_file)
    
    # Build dataloaders for training, validation, and testing
    train_loader = build_dataloader(config, loader_type='train')
    val_loader = build_dataloader(config, loader_type='val')
    test_loader = build_dataloader(config, loader_type='test')

    # Build label index map and set additional configurations
    all_splits_data = load_json(config['train_file_graphs']) + \
                    load_json(config['val_file_graphs']) + \
                    load_json(config['test_file_graphs'])
    label_index_map = get_mappings(all_splits_data)
    if config['procedural']:
        config['max_steps'] = max(train_loader.dataset.max_steps,
                              val_loader.dataset.max_steps,
                              test_loader.dataset.max_steps,)
    config['n_tags'] = len(label_index_map['tag2class'])
    config['n_edge_labels'] = len(label_index_map['edgelabel2class'])

    # Build model and set up optimizer
    model_start_path = args.get('model_start_path', None)
    model = build_model(config, model_start_path=model_start_path)

    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    scheduler = get_scheduler(optimizer=optimizer,
                            warmup_steps=int(config['training_steps'] * config['warmup_ratio']),
                            training_steps=config['training_steps'],
                            scheduler_type=config['scheduler_type'],
                            use_warmup=config['use_warmup'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Initialize tracking variables for evaluation and early stopping
    val_results_list = []
    test_results_list = []
    best_model_state = None
    best_val_metric = -np.inf
    patience_counter = 0

    # Define evaluation frequency
    eval_steps = config.get('eval_steps', 100)
    test_steps = config.get('test_steps', 100)

    # Choose training mode
    training_mode = config.get('training', 'epochs')  # 'epochs' or 'steps'
    torch.autograd.set_detect_anomaly(True)
    if training_mode == 'steps':
        training_steps = config.get('training_steps', 10000)
        patience = config.get('patience', 0.3)
        current_step = 0
        train_iter = iter(train_loader)  # create an initial iterator

        with tqdm(total=training_steps, desc="Training Steps") as pbar:
            while current_step < training_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # Replenish the iterator when exhausted
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                model.train()
                model.set_mode('train')
                optimizer.zero_grad()
                loss = model(batch)

                if torch.isnan(loss).item():
                    continue

                loss.backward()
                if config['use_clip_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_norm'])
                
                optimizer.step()
                if config['use_warmup']:
                    scheduler.step()

                current_step += 1
                pbar.update(1)
                pbar.set_description(f"Steps: {current_step}, Loss: {loss.item()}")

                if current_step % eval_steps == 0:
                    val_results, _ = run_evaluation(
                        model=model,
                        data_loader=val_loader,
                        eval_function=evaluate_model,
                        config=config,
                        label_index_map=label_index_map,
                        steps=current_step,
                    )
                    print(f'val F1 @ {current_step}:\t', val_results)
                    val_results_list.append(val_results)

                    parser_f1 = val_results['parser_labeled_results']['F1']
                    tagger_f1 = val_results['tagger_results']['F1']
                    labeled_f1 = tagger_f1 if config.get('freeze_parser', False) else parser_f1

                    if config.get('early_stopping', False):
                        if labeled_f1 > best_val_metric:
                            print(f'Best model updated at steps {current_step}: ({labeled_f1} > {best_val_metric})')
                            best_val_metric = labeled_f1
                            best_model_state = deepcopy(model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter * eval_steps >= training_steps * patience:
                            print("Early stopping triggered.")
                            current_step = training_steps  # Force exit of outer loop
                            break
                if current_step % test_steps == 0:
                    test_results, _ = run_evaluation(
                        model=model,
                        data_loader=test_loader,
                        eval_function=evaluate_model,
                        config=config,
                        label_index_map=label_index_map,
                        steps=current_step,
                    )
                    print(f'test F1 @ {current_step}:\t', test_results)
                    tmp_model_name = config['model_path'].replace('.pth', f'_{current_step}.pth')
                    if test_results['parser_labeled_results']['F1'] < 1e-2 and current_step > 1000:
                        tmp_model_name = tmp_model_name.replace('.pth', '_junk.pth')
                        torch.save(deepcopy(model.state_dict()), tmp_model_name)
                        raise RuntimeError('Bricked model!')
                    test_results_list.append(test_results)

    elif training_mode == 'epochs':
        num_epochs = config.get('epochs', 10)

        for epoch in range(1, num_epochs + 1):
            if epoch >= config.get('freeze_until_epoch', np.inf):
                model.encoder.unfreeze_encoder()

            model = train_epoch(model, train_loader, optimizer, epoch)

            val_results, _ = run_evaluation(
                model, val_loader, evaluate_model, config, label_index_map, epoch
            )
            print(val_results)
            val_results_list.append(val_results)

            # Early stopping check
            parser_f1 = val_results['parser_labeled_results']['F1']
            tagger_f1 = val_results['tagger_results']['F1']
            labeled_f1 = tagger_f1 if config.get('freeze_parser', False) else parser_f1

            if config.get('early_stopping', False):
                if labeled_f1 > best_val_metric:
                    print(f'Best model updated at epoch {epoch}: ({labeled_f1} > {best_val_metric})')
                    best_val_metric = labeled_f1
                    best_model_state = deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= num_epochs * 0.3:
                    print("Early stopping triggered.")
                    break

    # Save best model checkpoint if available
    if best_model_state is not None and config.get('save_model', False):
        torch.save(best_model_state, config['model_path'])
        print(f"Model saved at: {config['model_path']}")

    # Save validation results and configuration details
    save_json(val_results_list, os.path.join(config['save_dir'], "val_results_series.json"))
    save_json(test_results_list, os.path.join(config['save_dir'], "test_results_series.json"))
    save_json(train_loader.dataset.label_index_map, os.path.join(config['save_dir'], 'labels.json'))



    # Final evaluation on validation and test sets
    if config.get('save_model', False):
        model.load_state_dict(best_model_state)
        val_results, benchmark_metrics = run_evaluation(
            model, val_loader, evaluate_model, config, label_index_map
        )
        save_json(val_results, os.path.join(config['save_dir'], f"val_results.json"))
        save_json(benchmark_metrics, os.path.join(config['save_dir'], 'val_results_benchmark.json'))
        print('Validation results:', val_results)

        test_results, benchmark_metrics = run_evaluation(
            model, test_loader, evaluate_model, config, label_index_map
        )
        save_json(test_results, os.path.join(config['save_dir'], f"test_results.json"))
        save_json(benchmark_metrics, os.path.join(config['save_dir'], 'test_results_benchmark.json'))
        print('Test results:', test_results)

    summary = {'config': config, 'val_results': val_results, 'test_results': test_results}
    save_json(summary, os.path.join(config['save_dir'], 'config.json'))

if __name__ == "__main__":
    main()