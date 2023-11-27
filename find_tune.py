import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer

import time
import datetime
from tqdm import tqdm
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig.
from transformers import get_linear_schedule_with_warmup

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

   
# Load the dataset into a pandas dataframe.
df_train = pd.read_csv("~/med264/Dataset1/day1_30mortality_train.csv", index_col=0)
df_val = pd.read_csv("~/med264/Dataset1/day1_30mortality_val.csv", index_col=0)
df_test = pd.read_csv("~/med264/Dataset1/day1_30mortality_test.csv", index_col=0)

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df_train.shape[0]))
print('Number of validation sentences: {:,}\n'.format(df_val.shape[0]))
print('Number of testing sentences: {:,}\n'.format(df_test.shape[0]))

# Get the lists of sentences and their labels.
sentences_train = df_train.TEXT.values
labels_train = df_train.Label.values
sentences_val = df_val.TEXT.values
labels_val = df_val.Label.values
sentences_test = df_test.TEXT.values
labels_test = df_test.Label.values



# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('medicalai/ClinicalBERT', do_lower_case=True)


# Correct the path by expanding the tilde to the user's home directory
file_path_train = os.path.expanduser('~/med264/Dataset1/input_ids_train.pickle')
file_path_valid = os.path.expanduser('~/med264/Dataset1/input_ids_valid.pickle')
file_path_test = os.path.expanduser('~/med264/Dataset1/input_ids_test.pickle')


input_ids_train, input_ids_valid, input_ids_test = [], [], []

if os.path.exists(file_path_train):
    with open(file_path_train, 'rb') as f:
        input_ids_train = pickle.load(f)
    print('Loaded input_ids_train.')
else:
    for sent in tqdm(sentences_train):
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        input_ids_train.append(encoded_sent)
    with open(file_path_train, 'wb') as f:
        pickle.dump(input_ids_train, f)
    print('Saved input_ids_train.')


if os.path.exists(file_path_valid):
    with open(file_path_valid, 'rb') as f:
        input_ids_valid = pickle.load(f)
    print('Loaded input_ids_valid.')
else:
    for sent in tqdm(sentences_val):
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        input_ids_valid.append(encoded_sent)
    with open(file_path_valid, 'wb') as f:
        pickle.dump(input_ids_valid, f)
    print('Saved input_ids_valid.')

if os.path.exists(file_path_test):
    with open(file_path_test, 'rb') as f:
        input_ids_test = pickle.load(f)
    print('Loaded input_ids_test.')
else:
    for sent in tqdm(sentences_test):
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        input_ids_test.append(encoded_sent)
    with open(file_path_test, 'wb') as f:
            pickle.dump(input_ids_test, f)
    print('Saved input_ids_test.')
    

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences_train[0])
print('Token IDs:', input_ids_train[0])


MAX_LEN = 512
print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")
input_ids_valid = pad_sequences(input_ids_valid, maxlen=MAX_LEN, dtype="long",
                            value=0, truncating="post", padding="post")
input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN, dtype="long",
                            value=0, truncating="post", padding="post")

print('\nDone.')


# Create attention masks
attention_masks_train, attention_masks_valid, attention_masks_test = [], [], []

# For each sentence...
for sent in input_ids_train:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks_train.append(att_mask)

for sent in input_ids_valid:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks_valid.append(att_mask)

for sent in input_ids_test:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks_test.append(att_mask)

train_inputs, validation_inputs, test_inpputs, train_labels, validation_labels, test_labels =\
input_ids_train, input_ids_valid, input_ids_test, labels_train, labels_val, labels_test
# Do the same for the masks.
train_masks, validation_masks, test_masks = attention_masks_train, attention_masks_valid, attention_masks_test

# Convert all inputs and labels into torch tensors, the required datatype
# for our model.
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)


# The DataLoader needs to know our batch size for training, so we specify it
# here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.

batch_size = 64

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)



# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "medicalai/ClinicalBERT", num_labels = 2, output_attentions = False, output_hidden_states = False
)

# Tell pytorch to run this model on the GPU.
model.cuda()


optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

epochs = 10
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



import random

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


# Set the seed value all over the place to make this reproducible.
seed_val = 42

model_path = os.path.expanduser('~/med264/models/')
if not os.path.exists(model_path):
    os.makedirs(model_path)

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)
    model.save_pretrained(model_path + str(epoch_i))

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

# Save loss_values
with open(model_path + 'loss_values.pickle', 'wb') as f:
    pickle.dump(loss_values, f)