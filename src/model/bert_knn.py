import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from .model import Model
from models.model import BertForSequenceClassification
from utils.dataprocessor import getTestData, saveTestResults, getTrainData
import pickle

class BERTKNN(Model):
    def __init__(self, config):
        np.random.seed(config['general']['seed'])
        torch.manual_seed(config['general']['seed'])
        torch.cuda.manual_seed_all(config['general']['seed'])
        os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpu_ids']
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model_name = config['model']['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = torch.load(config['testing']['model_path'], map_location=self.device)
        model.to(self.device)
        self.model = model
        self.k = config['knnTest']['k']

    """
    Requires GPU. Because of NVIDIA driver issues Brandon is resolving, cannot be run on Patas yet.
    Previously run on Colab: https://colab.research.google.com/drive/1KOnXbKIuKDsjDaGs21QEQI25hbHg3nC0#scrollTo=qeAfrFPEuK7n
    with the code zipped downloaded from https://github.com/czhao028/KNN-EC
    """
    def train(self, training_config):
        np.random.seed(training_config['general']['seed'])
        torch.manual_seed(training_config['general']['seed'])
        torch.cuda.manual_seed_all(training_config['general']['seed'])
        os.environ["CUDA_VISIBLE_DEVICES"] = training_config['training']['gpu_ids']

        model_name = training_config['model'][
            'model_name']  # model_list=['bert-base-uncased','roberta-base','roberta-large']
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        dataset = getTrainData(tokenizer, model_name, training_config['data']['data_path'])
        # dev_data=getDevData(tokenizer,model_name,config['data']['data_path'])

        # train_sampler = RandomSampler(train_data)
        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['training']['train_batch_size'])
        # dev_sampler = SequentialSampler(dev_data)
        # dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=config['training']['dev_batch_size'])
        #

        num_folds = training_config['training']['num_folds']
        batch_size = training_config['training']['train_batch_size']
        num_epochs = training_config['training']['num_train_epochs']

        history_F1 = np.zeros(shape=(num_folds, num_epochs))
        history_trainingAcc = np.zeros(shape=(num_folds, num_epochs))
        history_validationAcc = np.zeros(shape=(num_folds, num_epochs))
        history_loss = np.zeros(shape=(num_folds, num_epochs))
        # train model
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        kfoldValidationPredictLabels = list()
        kfoldValidationTrueLabels = list()
        # Perform cross-validation
        # for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        # print(f"Fold {fold + 1}/{num_folds}")

        # Split data into training and validation sets for this fold
        # train_dataset = [dataset[i] for i in train_index]
        # val_dataset = [dataset[i] for i in val_index]
        train_dataset = dataset
        # Initialize model for each fold
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=training_config['model']['num_classes'],
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True)  # randomize order of training data
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #keep validation data in same order
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=eval(training_config['training']['learning_rate']),
                                      # args.learning_rate - default is 5e-5, our notebook had 2e-5
                                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                      )
        total_steps = num_folds * len(train_dataloader) * training_config['training']['num_train_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=training_config['training']['warmup_prop'],
                                                    # Default value in run_glue.py
                                                    num_training_steps=total_steps
                                                    )

        model.to(self.device)
        for epoch in range(num_epochs):
            # Train model for each epoch within this fold
            model.train()
            total_loss, step = 0, 0
            trainingAccuracyPerBatch = list()
            with tqdm(train_dataloader,
                      desc=f"Epoch {epoch + 1}/{num_epochs} - Training") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}/{training_config['training']['num_train_epochs']}")
                    b_input_ids, b_input_mask, b_labels = batch[0].to(self.device), batch[1].to(self.device), batch[
                        2].long().to(self.device)
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

                    predict = np.argmax(outputs[1].detach().cpu().numpy(), axis=1)
                    step += 1
                    training_acc = accuracy_score(batch[2].flatten(), predict.flatten())
                    trainingAccuracyPerBatch.append(training_acc)
                    tepoch.set_postfix(average_loss=total_loss / step, loss=loss.item(),
                                       f1=f1_score(batch[2].flatten(), predict.flatten(), average='weighted'),
                                       accuracy='{:.3f}'.format(training_acc))
                    time.sleep(0.0001)
            # history_trainingAcc[fold][epoch] = sum(trainingAccuracyPerBatch) / len(trainingAccuracyPerBatch)
            averageLossThisEpoch = total_loss / step
            # history_loss[fold][epoch] = averageLossThisEpoch
            # eval model
            model.eval()
            # validation set accuracy/f1 score
            # true_labels_validation,predict_labels_validation=[],[]
            # for batch in val_dataloader:
            #     batch = tuple(t.to(device) for t in batch)
            #     b_input_ids, b_input_mask, b_labels = batch
            #     with torch.no_grad():
            #         outputs = model(b_input_ids,
            #                         token_type_ids=None,
            #                         attention_mask=b_input_mask)
            #     logits = outputs[0].detach().cpu().numpy()
            #     label_ids = b_labels.to('cpu').numpy()
            #     predict_labels_validation.append(np.argmax(logits, axis=1).flatten())
            #     true_labels_validation.append(label_ids.flatten())
            # true_labels_validation=[y for x in true_labels_validation for y in x]
            # predict_labels_validation=[y for x in predict_labels_validation for y in x]
            # kfoldValidationTrueLabels.extend(true_labels_validation)
            # kfoldValidationPredictLabels.extend(predict_labels_validation)
            # print(classification_report(true_labels,predict_labels,digits=4))
            # f1=f1_score(kfoldValidationTrueLabels,kfoldValidationPredictLabels,average='macro')
            # history_F1[fold][epoch] = f1
            # validationAccuracy = accuracy_score(kfoldValidationTrueLabels, kfoldValidationPredictLabels)
            # history_validationAcc[fold][epoch] = validationAccuracy
            if training_config['training']['save_model'] and num_epochs > 0:
                torch.save(model,
                           "{}/{}_epoch{}.pt".format(training_config['data']['data_path'], model_name.replace("/", "-"),
                                                     epoch))
                # print(f"F1 scores: {history_F1}")
                # print(f"LOSSES: {history_loss}")
                # print(f"TRAINING ACCURACIES: {history_trainingAcc}")
                # print(f"VALIDATION ACCURACIES: {history_validationAcc}")
                # plot_training_history(np.mean(history_loss,axis=0), train_acc=np.mean(history_trainingAcc,axis=0), val_acc=np.mean(history_validationAcc,axis=0), val_f1=np.mean(history_F1,axis=0))

    def load(self, parameters_config):
        batch_size = parameters_config['knnTest']['test_batch_size']
        if "test_data_filename" in parameters_config['data']:
            test_data = getTestData(self.tokenizer, self.model_name, parameters_config['data']['data_path'],
                                    test_file_name=parameters_config['data']["test_data_filename"])
        else:
            test_data = getTestData(self.tokenizer, self.model_name, parameters_config['data']['data_path'])

        self.knn = pickle.load(open(parameters_config['knnTest']['model_file'] + str(parameters_config['knnTest']['k']), 'rb'))
        test_sampler = SequentialSampler(test_data)
        self.test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        self.model.eval()

    def save(self, parameters_config):
        print("Save code is already implemented in train() for KNNBert; skipping save() for bert_knn.py")
        #raise RuntimeError('save is not defined for FewShotGPT')

    def inference(self, inference_config): #Can just feed it an empty inference_config
        all_test_embedds = list()
        # training embeddings
        for batch in self.test_dataloader:
            b_input_ids, b_input_mask = batch[0].to(self.device), batch[1].to(self.device)  # .long().to(device)
            outputs = self.model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask)

            test_embedd = outputs[1].detach().cpu().numpy()
            all_test_embedds.append(test_embedd)

        test_embeds = np.concatenate(all_test_embedds, axis=0).astype("float32")
        predict_labels = self.knn.predict(test_embeds)
        #print(predict_labels)
        if "test_data_filename" in inference_config['data']:
            saveTestResults(inference_config['data']['data_path'], predict_labels,
                            inference_config['knnTest']['prediction_results'],
                            test_file_name=inference_config['data']["test_data_filename"]) #saves output labels
        else:
            saveTestResults(inference_config['data']['data_path'], predict_labels,
                            inference_config['knnTest']['prediction_results'])