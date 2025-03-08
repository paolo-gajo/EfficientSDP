"""
	Here, we train the MLM model on large corpus of recipes.
	corpus dataset!

	CUDA_VISIBLE_DEVICES=2 python3 tools/train_MLM.py --opts --dataset_name "Recipe1M" --save_dir "saved_models/MLM_models" --model_name "bert-base-uncased"
"""

import os

from stepparser.utils import (write_text,
						get_args,
						download_json_annotation,
						build_train_dev_list,
						is_file,
						ann_file_names,
						ann_url_dict,
						make_dir,
						read_text, 
						load_json,
						get_current_time_string
						)

args = get_args()

## get variables based on args 
dataset_name = args['dataset_name'] if 'dataset_name' in args else 'Recip1M'
save_dir = args['save_dir'] if 'save_dir' in args else 'saved_models/MLM_models'
model_name = args['model_name'] if 'model_name' in args else 'bert-base-uncased'

save_dir = os.path.join(save_dir, f'{dataset_name}_{get_current_time_string()}')
file_name, file_url = ann_file_names[dataset_name], ann_url_dict[dataset_name]
data_dir = os.path.join('data', f'{dataset_name}')
make_dir(data_dir)


## annotation file paths
json_data_path = os.path.join(data_dir, file_name)
train_file, dev_file = os.path.join(data_dir, 'train.txt'), os.path.join(data_dir, 'dev.txt')

## get file annotation and train/test paths
if is_file(json_data_path):
	## since the json file already exists, let's build the dataset
	if is_file(train_file) and is_file(dev_file):
		print("Dataset already exists! Loading them now!")
		train_list, val_list = read_text(train_file), read_text(dev_file)
	else:
		train_list, val_list = build_train_dev_list(load_json(json_data_path), dataset = dataset_name)
		write_text(train_file, '\n'.join(train_list))
		write_text(dev_file, '\n'.join(val_list))

else:
	download_json_annotation(file_url, data_dir)
	train_list, val_list = build_train_dev_list(load_json(json_data_path), dataset = dataset_name)
	write_text(train_file, '\n'.join(train_list))
	write_text(dev_file, '\n'.join(val_list))
	

##############################################################################################################################################################
#############################  Now we will do MLM, since we have train_list and val_list already! ############################################################
### This part of the code is borrowed from https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/MLM/train_mlm.py #######
##############################################################################################################################################################

from transformers import (AutoModelForMaskedLM, 
                        AutoTokenizer, 
                        DataCollatorForWholeWordMask, 
                        Trainer, 
                        TrainingArguments
                        )

per_device_train_batch_size = 256

save_steps = 1000               #Save model every 1k steps
num_train_epochs = 10           #Number of epochs
use_fp16 = False                #Set to True, if your GPU supports FP16 operations
max_length = 100                #Max length for a text input
mlm_prob = 0.15                 #Probability that a word is replaced by a [MASK] token

# Load the model
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Save checkpoints to:", save_dir)

#A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

train_dataset = TokenizedSentencesDataset(train_list, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(val_list, tokenizer, max_length, cache_tokenization=True) if len(val_list) > 0 else None

##### Training arguments
data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

training_args = TrainingArguments(
    output_dir=save_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps" if dev_dataset is not None else "no",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=save_steps,
    save_total_limit=1,
    prediction_loss_only=True,
    fp16=use_fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Save tokenizer to:", save_dir)
tokenizer.save_pretrained(save_dir)

trainer.train()

print("Save model to:", save_dir)
model.save_pretrained(save_dir)

print("Training done")