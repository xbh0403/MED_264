import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm
import os
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# If there's a GPU available...
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
    

print('Max train sentence length: ', max([len(sen) for sen in input_ids_train]))
print('Max valid sentence length: ', max([len(sen) for sen in input_ids_valid]))
print('Max test sentence length: ', max([len(sen) for sen in input_ids_test]))

# We'll borrow the `pad_sequences` utility function to do this.
from keras.preprocessing.sequence import pad_sequences

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

train_inputs, validation_inputs, test_inputs, train_labels, validation_labels, test_labels =\
input_ids_train, input_ids_valid, input_ids_test, labels_train, labels_val, labels_test
# Do the same for the masks.
train_masks, validation_masks, test_masks = attention_masks_train, attention_masks_valid, attention_masks_test

# Convert all inputs and labels into torch tensors, the required datatype
# for our model.
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
test_inputs = torch.tensor(test_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
test_labels = torch.tensor(test_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
test_masks = torch.tensor(test_masks)

batch_size = 64

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Create the DataLoader for our test set.
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model_path = os.path.expanduser('~/med264/models/')
preds_path = os.path.expanduser('~/med264/preds/')

# Check if preds_path exists, if not, create it
if not os.path.exists(preds_path):
    os.makedirs(preds_path)
from tqdm import tqdm

acc_results = {}
preds_results = {}

for i in range(10):
  print('Model ' + str(i) + ':')
  model_path_i = os.path.expanduser('~/med264/models/' + str(i) + '/')
  print('Loading model from ' + model_path_i + '...')
  model = BertForSequenceClassification.from_pretrained(
    model_path_i, num_labels = 2, output_attentions = False, output_hidden_states = False
  )
  model.cuda()
  model.eval()

  pred_path_i = os.path.expanduser('~/med264/preds/' + str(i) + '/')
  # Check if pred_path_i exists, if not, create it
  if not os.path.exists(pred_path_i):
      os.makedirs(pred_path_i)
  # Tracking variables
  predictions_train, predictions_val, predictions_test = [], [], []
  true_labels_train, true_labels_val, true_labels_test = [], [], []
  # Predict
  print('Predicting labels for {:,} train sentences...'.format(len(train_inputs)))
  for batch in tqdm(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
    logits = outputs[0]
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    # Store predictions and true labels
    predictions_train.append(logits)
    true_labels_train.append(label_ids)
  flat_predictions_train = np.concatenate(predictions_train, axis=0)
  flat_predictions_train = np.argmax(flat_predictions_train, axis=1).flatten()
  flat_true_labels_train = np.concatenate(true_labels_train, axis=0)
  # Calculate the accuracy for this batch of test sentences.
  train_accuracy = accuracy_score(flat_true_labels_train, flat_predictions_train)
  print('  Train Accuracy: {0:.2f}'.format(train_accuracy))

  print('Predicting labels for {:,} validation sentences...'.format(len(validation_inputs)))
  for batch in tqdm(validation_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
    logits = outputs[0]
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    # Store predictions and true labels
    predictions_val.append(logits)
    true_labels_val.append(label_ids)
  flat_predictions_val = np.concatenate(predictions_val, axis=0)
  flat_predictions_val = np.argmax(flat_predictions_val, axis=1).flatten()
  flat_true_labels_val = np.concatenate(true_labels_val, axis=0)
  # Calculate the accuracy for this batch of test sentences.
  val_accuracy = accuracy_score(flat_true_labels_val, flat_predictions_val)
  print('  Validation Accuracy: {0:.2f}'.format(val_accuracy))

  print('Predicting labels for {:,} test sentences...'.format(len(test_inputs)))
  for batch in tqdm(test_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
    logits = outputs[0]
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    # Store predictions and true labels
    predictions_test.append(logits)
    true_labels_test.append(label_ids)
  flat_predictions_test = np.concatenate(predictions_test, axis=0)
  flat_predictions_test = np.argmax(flat_predictions_test, axis=1).flatten()
  flat_true_labels_test = np.concatenate(true_labels_test, axis=0)
  # Calculate the accuracy for this batch of test sentences.
  val_accuracy = accuracy_score(flat_true_labels_test, flat_predictions_test)
  print('  Test Accuracy: {0:.2f}'.format(test_accuracy))

  acc_results['model_' + str(i)] = {
    'train_accuracy': train_accuracy,
    'val_accuracy': val_accuracy,
    'test_accuracy': test_accuracy
  }
  
  preds_results['model_' + str(i)] = {
    'train_preds': flat_predictions_train,
    'train_true_labels': flat_true_labels_train,
    'val_preds': flat_predictions_val,
    'val_true_labels': flat_true_labels_val,
    'test_preds': flat_predictions_test,
    'test_true_labels': flat_true_labels_test
  }

  print('Saving predictions...')
  with open(pred_path_i + 'preds.pickle', 'wb') as f:
    pickle.dump(preds_results['model_' + str(i)], f)
  print('Saved predictions.')

print('Saving overall results...')
with open(preds_path + 'acc_results.pickle', 'wb') as f:
  pickle.dump(acc_results, f)
print('Saved overall results.')

print('Done.')