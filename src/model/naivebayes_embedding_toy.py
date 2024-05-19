
import collections; import pandas as pd; import numpy as np; import os;
import spacy; import nltk; import ast; import math;
from string import punctuation
from spacy.lang.en import stop_words; # nlp = spacy.load('en_core_web_sm');
# import rich; 
import copy; 
# nltk.download('words'); nltk.download("punkt");
import string; from string import punctuation as string_punct;
# import enchant;

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import classification_report

from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.utils import resample

# %%writefile code.py

from .model import Model

'''
'''

class NaiveBayesEmbeddingToy(Model):
    def __init__(self, model_configs):
        task_type = model_configs['task_type']
        if task_type == 'classification':
            self.task_type = task_type
            self.training_data = None
            self.gnb = GaussianNB();
            # self.gnb = LogisticRegression(random_state = 123)
            self.instances = None
            self.labels = None
            self.X_train_split = None;
            self.y_train_split = None;

            ### for combining token embeddings
            self.tfidf = None
            self.weighted_embedding_by_tfidf = None;
            self.average_embedding = None;

            ## for holding inference results
            self.inference_result = None;

            ## model train_m1
            self.gnb_m1 = GaussianNB();
            ## model train_m12
            self.gnb_m12 = GaussianNB();

            ## model train ratio
            self.train_ratio_result = {};
        else:
            raise ValueError(f'Model is not supported for {task_type} task.')

    def train(self, training_configs):

        if training_configs['split_ratio'] > 0:
            # Split the dataset into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(self.instances, self.labels, test_size=0.2, random_state=42)
        else:
            X_train = self.instances;
            y_train = self.labels;

        self.X_train_split = X_train;
        self.y_train_split = y_train;

        #############################################################
        # MODEL 0, Flat prediction ... dimension: all data
        # Initialize the Gaussian Naive Bayes classifier
        model = self.gnb;
        model.fit(X_train, y_train);
        model_proba = model.predict_proba(X_train);

        # Predict on the test set
        y_pred = model.predict(X_train)
        self.training_data['predicted_basic'] = y_pred;
        # Evaluate the model
        print("Accuracy:", accuracy_score(self.labels, y_pred))
        print("Classification Report:\n", classification_report(self.labels, y_pred));

    def inference_m1(self, inference_config, model_configs, arg_run_using_train = True):

        #############################################################
        # MODEL 0, Flat prediction ... dimension: all data
        # Initialize the Gaussian Naive Bayes classifier
        model = self.gnb; # copy to here
        model_c2i = {c:i for i, c in enumerate(model.classes_)}
        model_i2c = {i:c for i, c in enumerate(model.classes_)}
        X_train = self.X_train_split;
        y_train = self.y_train_split;

        # y_pred = model.predict(X_train)
        # print("Accuracy:", accuracy_score(self.labels, y_pred))
        # print("Classification Report:\n", classification_report(self.labels, y_pred));

        if arg_run_using_train == False:
            self.inference(inference_config, model_configs);

        #############################################################
        ## MODEL 1: naively readjust weight of each non-neutral class
        m1_n_nn = self.gnb_m1; # GaussianNB(); # initialize
        m1_n_nn.fit(X_train, self.training_data['n-nn'].tolist()); # only 2 classes
        # m1_n_nn_proba = m1_n_nn.predict_proba(X_train);
        # m1_n_nn_proba_ldict = [{m1_n_nn.classes_[i]: p for i,p in enumerate(row)} for row in m1_n_nn_proba];
        # m1_predict = m1_n_nn.predict(X_train)

        # print("Accuracy:", accuracy_score(self.training_data['n-nn'].tolist(), m1_predict))
        # print("Classification Report:\n", classification_report(self.training_data['n-nn'].tolist(), m1_predict));

        if arg_run_using_train == False:
            X_new = np.vstack(self.inference_result['vector'].values);
        else:
            X_new = X_train;


        model_proba_X_new = model.predict_proba(X_new);
        m1_n_nn_proba_X_new = m1_n_nn.predict_proba(X_new);
        m1_n_nn_proba_ldict_X_new = [{m1_n_nn.classes_[i]: p for i,p in enumerate(row)} for row in m1_n_nn_proba_X_new];

        model_m1_proba_X_new =  [[v*m1_n_nn_proba_ldict_X_new[rj]['Non-neutral'] if model_i2c[i] != 'Neutral' else v for i,v in enumerate(row) ] for rj,row in enumerate(model_proba_X_new)];

        temp = [sum(row) for rj,row in enumerate(model_m1_proba_X_new)]
        model_m1_proba1_X_new = [[v/temp[rj] for v in row ] for rj,row in enumerate(model_m1_proba_X_new)];
        model_m1_pred_X_new = [model_i2c[np.argmax(row)] for row in model_m1_proba1_X_new];

        if arg_run_using_train == False:
            self.inference_result['Labels'] = model_m1_pred_X_new;
            output_filename = inference_config['output_filename_m1'];
            self.inference_result[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            print("Accuracy:", accuracy_score(self.labels, model_m1_pred_X_new))
            print("Classification Report:\n", classification_report(self.labels, model_m1_pred_X_new));


    def inference_m12(self, inference_config, model_configs, arg_run_using_train = True):

        #############################################################
        # MODEL 0, Flat prediction ... dimension: all data
        # Initialize the Gaussian Naive Bayes classifier
        model = self.gnb; # copy to here, 6 classes
        model_c2i = {c:i for i, c in enumerate(model.classes_)}
        model_i2c = {i:c for i, c in enumerate(model.classes_)}
        X_train = self.X_train_split;
        y_train = self.y_train_split;
        # model_proba = model.predict_proba(X_train);
        # y_pred = model.predict(X_train)
        # print("Accuracy:", accuracy_score(self.labels, y_pred))
        # print("Classification Report:\n", classification_report(self.labels, y_pred));

        if arg_run_using_train == False:
            self.inference(inference_config, model_configs);

        #############################################################
        ## MODEL 1-sophi: hier-readjust weight of each non-neutral class... dimension: only those non-neutral

        m1_n_nn = self.gnb_m1 #GaussianNB(); # initialize
        m1_n_nn.fit(X_train, self.training_data['n-nn'].tolist()); # train 2 classes.
        # m1_n_nn_proba = m1_n_nn.predict_proba(X_train);
        m1_n_nn_predict = m1_n_nn.predict(X_train);
        m1_n_nn_c2i = {c:i for i, c in enumerate(model.classes_)}
        m1_n_nn_i2c = {i:c for i, c in enumerate(model.classes_)}
        print("Accuracy:", accuracy_score(self.training_data['n-nn'].tolist(), m1_n_nn_predict))
        print("Classification Report:\n", classification_report(self.training_data['n-nn'].tolist(), m1_n_nn_predict));

        p_neutral_false = 1/6;
        temp_prior = {c: p_neutral_false if c == "Neutral" else (1-p_neutral_false)/5 for c in model.classes_}

        m12_n_nn = GaussianNB(priors = [temp_prior[model_i2c[i]] for i in model_i2c]); # initialize
        temp_cond = (m1_n_nn_predict == "Non-neutral");
        temp_X = X_train[temp_cond];
        temp_y = y_train[temp_cond]; ## there are still 6! some predictions are wrong!
        # temp_counter = 0;
        # temp_origidx2subidx = {};
        # for i,b in enumerate(temp_cond):
        #     if b:
        #         temp_origidx2subidx[i] = temp_counter;
        #         temp_counter += 1;

        m12_n_nn.fit(temp_X, temp_y); # fit SUBSET of data
        # m12_n_nn_proba = m12_n_nn.predict_proba(temp_X);
        # m12_n_nn_proba_ldict = [{m12_n_nn.classes_[i]: p for i,p in enumerate(row)} for row in m12_n_nn_proba];
        # m12_n_nn_i2c = {i:c for i, c in enumerate(m12_n_nn.classes_)}
        # m12_n_nn_c2i = {c:i for i, c in enumerate(m12_n_nn.classes_)}
        # m12_predict = m12_n_nn.predict(temp_X);

        '''
        start inference
        '''
        if arg_run_using_train == False:
            X_new = np.vstack(self.inference_result['vector'].values);
        else:
            X_new = X_train;

        model_proba_X_new = model.predict_proba(X_new); # flat 6-class prediction

        m1_n_nn_predict_X_new = m1_n_nn.predict(X_new); # first, use 2-class prediction
        m1_n_nn_proba_X_new = m1_n_nn.predict_proba(X_new);
        m1_n_nn_proba_ldict_X_new = [{m1_n_nn.classes_[i]: p for i,p in enumerate(row)} for row in m1_n_nn_proba_X_new];

        temp_cond_X_new = (m1_n_nn_predict_X_new == "Non-neutral");
        temp_X_new = X_new[temp_cond_X_new];
        # temp_y = y_new[temp_cond]; ## there are still 6! some predictions are wrong!
        temp_counter_X_new = 0;
        temp_origidx2subidx_X_new = {};
        for j,b in enumerate(temp_cond_X_new):
            if b:
                temp_origidx2subidx_X_new[j] = temp_counter_X_new;
                temp_counter_X_new += 1;

        m12_n_nn_proba_X_new = m12_n_nn.predict_proba(temp_X_new);
        # m12_n_nn_proba_ldict = [{m12_n_nn.classes_[i]: p for i,p in enumerate(row)} for row in m12_n_nn_proba];
        m12_n_nn_i2c = {i:c for i, c in enumerate(m12_n_nn.classes_)}
        m12_n_nn_c2i = {c:i for i, c in enumerate(m12_n_nn.classes_)}
        # m12_predict = m12_n_nn.predict(temp_X);

        sub_m1_n_nn_proba_ldict_X_new = np.array(m1_n_nn_proba_ldict_X_new)[temp_cond_X_new];
        model_m12_proba = [[v*sub_m1_n_nn_proba_ldict_X_new[rj]['Non-neutral'] if m12_n_nn_i2c[i] != 'Neutral' else v for i,v in enumerate(row) ] for rj,row in enumerate(m12_n_nn_proba_X_new)];
        model_m12_proba1 = [];
        for j, b in enumerate(temp_cond_X_new):
            temp_row = copy.deepcopy(model_proba_X_new[j]); # just for placehoder.
            # assert abs(sum(temp_row) -1) <= 1e-3;
            if b:
                for i,v in enumerate(model_proba_X_new[j]):
                    if model_i2c[i] != 'Neutral':
                        temp_row[i] = model_m12_proba[temp_origidx2subidx_X_new[j]][m12_n_nn_c2i[model_i2c[i]]];
                    else:
                        temp_row[i] = m1_n_nn_proba_ldict_X_new[j]['Neutral'];
                temp_row_sum = sum(temp_row);
                model_m12_proba1.append([v/temp_row_sum for v in temp_row]);
                # assert abs(sum(model_m12_proba1[-1]) - 1) < 1e-4
                # assert np.argmax(temp_row) != model_c2i['Neutral']
            else:
                temp_non_neutral_sum = 0;
                for i,v in enumerate(model_proba_X_new[j]):
                    if model_i2c[i] == 'Neutral':
                        temp_row[i] = m1_n_nn_proba_ldict_X_new[j]['Neutral'];
                        temp_neutral_val = temp_row[i];
                    else:
                        temp_non_neutral_sum += model_proba_X_new[j][i];
                # assert temp_non_neural_sum + temp_neural_val == 1;
                # print(temp_non_neutral_sum, m1_n_nn_proba_ldict_X_new[j]['Neutral'], temp_neutral_val)

                for i,v in enumerate(model_proba_X_new[j]):
                    if model_i2c[i] != 'Neutral':
                        temp_row[i] = (1-temp_neutral_val)*v/temp_non_neutral_sum;

                # temp_row_sum = sum(temp_row);
                # model_m12_proba1.append([v/temp_row_sum for v in temp_row]);
                model_m12_proba1.append(temp_row);
                assert np.argmax(temp_row) == model_c2i['Neutral']

        model_m12_pred = [model_i2c[np.argmax(row)] for row in model_m12_proba1];

        if arg_run_using_train == False:
            self.inference_result['Labels'] = model_m12_pred;
            output_filename = inference_configs['output_filename_m12'];
            self.inference_result[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            print("Accuracy:", accuracy_score(self.labels, model_m12_pred));
            print("Classification Report:\n", classification_report(self.labels, model_m12_pred));

    '''
    ##### below are for ratios
    '''
    def train_ratio(self, training_configs, arg_type = 1):

        if model_configs['split_ratio'] > 0:
            # Split the dataset into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(self.instances, self.labels, test_size=0.2, random_state=42)
        else:
            X_train = self.instances;
            y_train = self.labels;

        if arg_type != 1:
            optimal_mu_class1, optimal_sigma_class1 = self.train_ratio_helper(X_train, 'Neutral')
            optimal_mu_class2, optimal_sigma_class2 = self.train_ratio_helper(X_train, 'Non-neutral')
        else:
            optimal_mu_class1, optimal_sigma_class1, optimal_mu_class2, optimal_sigma_class2 = self.train_ratio_helper_h1(X_train, 'Neutral')

        params_class1 = np.concatenate([optimal_mu_class1, optimal_sigma_class1])
        params_class2 = np.concatenate([optimal_mu_class2, optimal_sigma_class2])


        # Example usage
        # X_new = X_train[:10, :]
        # probabilities, predicted_class = self.predict_ratio_class_probabilities(X_new, params_class1, params_class2)
        # print("Probabilities for each class:", probabilities)
        # print("Predicted class (0 for Class 1, 1 for Class 2):", predicted_class)

        self.train_ratio_result['pos'] = params_class1 # (optimal_mu_class1, optimal_sigma_class1)
        self.train_ratio_result['neg'] = params_class2 # (optimal_mu_class2, optimal_sigma_class2)
        # self.train_ratio_result['pred_func'] = predict_class_probabilities;

    def train_ratio_helper(self, arg_X, arg_pos = 'Non-neutral'):

        X_train_pos = arg_X[self.training_data['n-nn'] == arg_pos];
        X_train_neg = arg_X[self.training_data['n-nn'] != arg_pos];

        len_A = len(X_train_pos);
        len_B = len(X_train_neg);

        A_upsampled = X_train_pos;
        B_downsampled = X_train_neg;

        if len_A / len_B < 0.8:
            # Upsample minority class
            A_upsampled = resample(X_train_pos, replace=True, n_samples=len(X_train_neg), random_state=123)  # upsample to match size of B
            # Downsample majority class
            B_downsampled = resample(X_train_neg, replace=False, n_samples=len(X_train_pos), random_state=123)  # downsample to match size of A

        def log_likelihood(data, mu, sigma):
            # Data shape is (n_samples, n_features), mu and sigma are vectors of shape (n_features,)
            return np.sum(norm.logpdf(data, mu, np.abs(sigma)))  # Ensure sigma is positive

        def objective(params, A, B, lambda_ = 0):
            # Split params into mu and sigma parts
            d = A.shape[1]  # number of features
            mu = params[:d]
            sigma = params[d:]
            if np.any(sigma <= 0):
                return np.inf  # Ensure sigma is positive, penalize illegal values
            # L2 Regularization term
            regularization = lambda_ * (np.sum(mu**2))  + 0 * np.sum(sigma**2)
            log_likelihood_A = log_likelihood(A, mu, sigma)
            log_likelihood_B = log_likelihood(B, mu, sigma)

            return -(log_likelihood_A - log_likelihood_B) - regularization # Negate because we want to maximize


        # Initial guesses for means and sigmas for each feature
        initial_means = np.mean(X_train_pos, axis=0)
        initial_sigmas = np.std(X_train_pos, axis=0)
        initial_params = np.concatenate([initial_means, initial_sigmas])

        nb_features = arg_X.shape[1];

        # Optimization
        result = minimize(objective, initial_params, args=(A_upsampled, B_downsampled),
                          bounds=[(None, None)]*nb_features + [(1e-4, None)]*nb_features)
        optimal_params = result.x
        optimal_mu = optimal_params[:nb_features]
        optimal_sigma = optimal_params[nb_features:]

        # print("Optimal mus:", optimal_mu)
        # print("Optimal sigmas:", optimal_sigma)

        return optimal_mu, optimal_sigma

    def train_ratio_helper_h1(self, arg_X,  arg_pos= 'Non-neutral'):

        X_train_pos = arg_X[self.training_data['n-nn'] == arg_pos];
        X_train_neg = arg_X[self.training_data['n-nn'] != arg_pos];

        nb_features = X_train_pos.shape[1]; # print(nb_features)

        def log_likelihood(data, mu, sigma):
            # Data shape is (n_samples, n_features), mu and sigma are vectors of shape (n_features,)
            return np.sum(norm.logpdf(data, mu, np.abs(sigma)))  # Ensure sigma is positive

        def revised_objective(params, A,B):
            # Extract parameters for each class
            d = A.shape[1];
            mean_yes = params[:d]
            sigma_yes = params[d:2*d]
            mean_no  = params[2*d:3*d]
            sigma_no = params[3*d:4*d]

            X_yes = A;
            X_no = B;

            log_likelihood_yes = log_likelihood(X_yes, mean_yes, sigma_yes)
            log_likelihood_no =  log_likelihood(X_no, mean_no, sigma_no)

            # Calculate between-class separation (Mahalanobis distance)
            separation = (mean_yes - mean_no)**2 / (sigma_yes**2 + sigma_no**2)

            # Objective to minimize negative log likelihoods (maximize likelihoods) and maximize separation
            return -(log_likelihood_yes + log_likelihood_no) - np.sum(separation)

        # Example for optimization
        from scipy.optimize import minimize

        # Initial guesses for parameters: [mean_yes, sigma_yes, mean_no, sigma_no]
        initial_params = np.concatenate([np.mean(X_train_pos, axis = 0), np.std(X_train_pos, axis = 0),
                          np.mean(X_train_neg, axis = 0), np.std(X_train_neg, axis = 0)])
        # print(len(initial_params))
        # result = minimize(objective, initial_params, args=(A_upsampled, B_downsampled),
        #                   bounds=[(None, None)]*nb_features + [(1e-4, None)]*nb_features)
        result = minimize(revised_objective, initial_params, args = (X_train_pos, X_train_neg),
                          bounds=[(None, None)]*nb_features + [(1e-4, None)]*nb_features + [(None, None)]*nb_features + [(1e-4, None)]*nb_features);

        optimal_params = result.x
        optimal_mu_pos = optimal_params[:nb_features]
        optimal_sigma_pos = optimal_params[nb_features:2*nb_features]
        optimal_mu_neg = optimal_params[2*nb_features:3*nb_features]
        optimal_sigma_neg = optimal_params[3*nb_features:4*nb_features];

        print(len(optimal_params))

        # print("Optimal mus:", optimal_mu)
        # print("Optimal sigmas:", optimal_sigma)

        return optimal_mu_pos, optimal_sigma_pos, optimal_mu_neg, optimal_sigma_neg

    def predict_ratio_class_probabilities(self, X, params_class1, params_class2, prior1=0.5, prior2=0.5):
        mu1, sigma1 = params_class1[:X.shape[1]], params_class1[X.shape[1]:]
        mu2, sigma2 = params_class2[:X.shape[1]], params_class2[X.shape[1]:]

        def log_prob_x_given_class(x, mu, sigma):
            sigma = np.where(sigma <= 0, 1e-10, sigma)  # Avoid division by zero or negative sigmas
            return np.sum(norm.logpdf(x, mu, sigma))

        log_prob_x_given_class1 = np.apply_along_axis(log_prob_x_given_class, 1, X, mu1, sigma1)
        log_prob_x_given_class2 = np.apply_along_axis(log_prob_x_given_class, 1, X, mu2, sigma2)

        print("Log probabilities for class 1:", log_prob_x_given_class1)
        print("Log probabilities for class 2:", log_prob_x_given_class2)

        log_prior1 = np.log(prior1)
        log_prior2 = np.log(prior2)

        log_post_prob_class1 = log_prior1 + log_prob_x_given_class1
        log_post_prob_class2 = log_prior2 + log_prob_x_given_class2

        max_log_prob = np.maximum(log_post_prob_class1, log_post_prob_class2)
        sum_exp_log_prob = np.exp(log_post_prob_class1 - max_log_prob) + np.exp(log_post_prob_class2 - max_log_prob)

        prob_class1 = np.exp(log_post_prob_class1 - max_log_prob - np.log(sum_exp_log_prob))
        prob_class2 = np.exp(log_post_prob_class2 - max_log_prob - np.log(sum_exp_log_prob))

        # print("Normalized probabilities for class 1:", prob_class1)
        # print("Normalized probabilities for class 2:", prob_class2)

        return np.vstack((prob_class1, prob_class2)).T, np.argmax([prob_class1, prob_class2], axis=0)

    def inference_ratio(self, inference_configs, model_configs):
        self.inference(inference_configs, model_configs, arg_use_custom_model=True);

        X_dev = np.vstack(self.inference_result['vector'].values);

        params_class1 = self.train_ratio_result['pos'];
        params_class2 = self.train_ratio_result['neg'];

        _, temp = self.predict_ratio_class_probabilities(X_dev, params_class1, params_class2);
        self.inference_result['Labels'] = ["Neutral" if i == 1 else "Non-neutral" for i in temp];
        # test_data["Labels"] = test_data.apply(lambda x: self.single_instance_inference(x), axis=1)
        # test_data[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        output_filename = inference_configs['output_filename_ratio']
        self.inference_result[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)

    '''
    ##### above are for ratios
    '''

    def load(self, parameters_configs, model_configs):
        training_data_filename = parameters_configs['training_data_filename']
        self.training_data = pd.read_csv(training_data_filename, sep='\t').iloc[:];

        # add more labels
        self.training_data['n-nn'] = self.training_data['Labels'].apply(lambda x: "Non-neutral" if x!= "Neutral" else "Neutral");
        self.training_data['n-pos-neg'] = self.training_data['Labels'].apply(lambda x: "Positive" if x in ["Joy", "Love"] else ("Neutral" if x== "Neutral" else "Negative"));
        # print(self.training_data)

        nlp_en = spacy.load('en_core_web_sm');

        # Apply the preprocessing function to the tweets
        self.training_data['vector'] = self.preprocess_and_get_vector(texts = self.training_data['Texts'], arg_nlp = nlp_en, arg_weight = model_configs['weighting_type']);

        # lengths = [len(vec) for vec in self.training_data['vector'].values]
        # print("Vector lengths:", lengths)

        # Stack all tweet vectors into a 2D array
        X_vectors = np.vstack(self.training_data['vector'].values)
        self.instances = X_vectors;
        self.labels = np.array(self.training_data["Labels"].tolist());

        return;

    # def single_instance_inference(self, instance):
    #     predicted_label = self.knn.predict(np.array(ast.literal_eval(instance['embedding'])).reshape(1, -1))[0]
    #     print(instance.name, 'tweet: ', instance['Texts'])
    #     print('predicted label: ', predicted_label)
    #     return predicted_label

    def inference(self, inference_configs, model_configs, arg_use_custom_model = False):
        test_data_filename = inference_configs['test_data_filename'];
        output_filename = inference_configs['output_filename'];
        test_data = pd.read_csv(test_data_filename, sep='\t');
        test_data['Labels'] = None;
        test_data_en = test_data.iloc[:100];
        test_data_es = test_data.iloc[100:200];
        test_data_fr = test_data.iloc[200:300];
        test_data_nl = test_data.iloc[300:400];
        test_data_ru = test_data.iloc[400:];

        nlp_en = spacy.load('en_core_web_sm');
        nlp_es = spacy.load('es_core_news_sm');
        nlp_fr = spacy.load('fr_core_news_sm');
        nlp_nl = spacy.load('nl_core_news_sm');
        nlp_ru = spacy.load('ru_core_news_sm');

        # Apply the preprocessing to get embeddings
        test_data_en['vector'] = self.preprocess_and_get_vector(texts = test_data_en['Texts'], arg_nlp = nlp_en, arg_weight = model_configs['weighting_type']);
        test_data_es['vector'] = self.preprocess_and_get_vector(texts = test_data_es['Texts'], arg_nlp = nlp_es, arg_weight = model_configs['weighting_type']);
        test_data_fr['vector'] = self.preprocess_and_get_vector(texts = test_data_fr['Texts'], arg_nlp = nlp_fr, arg_weight = model_configs['weighting_type']);
        test_data_nl['vector'] = self.preprocess_and_get_vector(texts = test_data_nl['Texts'], arg_nlp = nlp_nl, arg_weight = model_configs['weighting_type']);
        test_data_ru['vector'] = self.preprocess_and_get_vector(texts = test_data_ru['Texts'], arg_nlp = nlp_ru, arg_weight = model_configs['weighting_type']);

        if arg_use_custom_model == False:
            test_data_en.loc[:,'Labels'] = self.gnb.predict(np.vstack(test_data_en['vector'].values));
            test_data_es.loc[:,'Labels'] = self.gnb.predict(np.vstack(test_data_es['vector'].values));
            test_data_fr.loc[:,'Labels'] = self.gnb.predict(np.vstack(test_data_fr['vector'].values));
            test_data_nl.loc[:,'Labels'] = self.gnb.predict(np.vstack(test_data_nl['vector'].values));
            test_data_ru.loc[:,'Labels'] = self.gnb.predict(np.vstack(test_data_ru['vector'].values));

        result_vertical = pd.concat([test_data_en, test_data_es, test_data_fr, test_data_nl, test_data_ru]);
        self.inference_result = result_vertical;
        # test_data["Labels"] = test_data.apply(lambda x: self.single_instance_inference(x), axis=1)
        # test_data[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        result_vertical[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)

    '''
    below for preprocessing and token combinations
    '''
    def preprocess_text(self, text):
        pass;
        return text;

    def preprocess_and_get_vector(self, texts, arg_nlp, arg_weight, arg_test = False):

        texts = [self.preprocess_text(text) for text in texts]

        processed_texts = [];

        token_embedding_weight = [];
        token_embeddings_by_doc = [];
        tokens_concatenated_by_doc = [];

        doc_idx = 0;
        for doc in arg_nlp.pipe(texts, disable=["parser", "ner"]):
            # tokens = [token for token in doc if not token.is_punct and not token.is_space and not token.is_stop];
            # tokens = [token for token in doc if not token.is_stop and not token.is_space]
            # tokens [] can be zero.
            tokens = [token for token in doc if not token.is_space]

            token_embeddings_in_doc = [];
            tokens_in_doc = [];

            for token in tokens:
                if token.has_vector:
                    '''be careful here, if append token, then the token is an object, not a str!!!!'''
                    tokens_in_doc.append(token.text);
                    token_embeddings_in_doc.append(token.vector);
                    # print(len(token.vector))

            # If no vectors were available for tokens, append zero vector
            if not token_embeddings_in_doc:
                processed_texts.append(np.zeros((96,)))
            else:
                token_embeddings_by_doc.append(token_embeddings_in_doc)
                # Calculate the average vector of the remaining tokens
                vector = np.mean(token_embeddings_in_doc, axis=0);
                processed_texts.append(vector);

            tokens_concatenated_by_doc.append(" ".join([token for token in tokens_in_doc]));
            token_embedding_weight.append({'tokens': tokens_in_doc,
                                                'embeddings':  token_embeddings_in_doc,
                                                'weights': []});
            # assert (len(self.token_embedding_weight[-1]['tokens']) == len(self.token_embedding_weight[-1]['embeddings']))
            doc_idx+=1;

        result = processed_texts;
        # self.average_embedding = result;

        if arg_weight == 'tfidf':
            tfidf_matrix = self.get_X_tfidf(token_embedding_weight);
            result = self.get_weighted_embeddings_X_tfidf(token_embedding_weight, tfidf_matrix)
        elif arg_weight == 'pole':
            result = self.get_weighted_embeddings_X_pole(token_embedding_weight)
        elif arg_weight == 'maxmin':
            result = self.get_weighted_embeddings_X_maxmin(token_embedding_weight)
        elif arg_weight == 'add':
            result = self.get_weighted_embeddings_X_add(token_embedding_weight)
        else:
            pass;

        return result;

    def get_X_tfidf(self, arg_token_embed_weight):

        """The input is a list of dictionary of lists"""
        # Calculate TF (term frequency)
        tf = collections.defaultdict(dict)
        for index, row in enumerate(arg_token_embed_weight):
            tokens = row['tokens']
            token_counts = collections.defaultdict(int)
            for token in tokens:
                token_counts[token] += 1
            for token, count in token_counts.items():
                tf[token][index] = count / len(tokens)
        # print(tf)

        # Calculate IDF (inverse document frequency)
        idf = {}
        N = len(arg_token_embed_weight)
        for token, docs in tf.items():
            idf[token] = math.log((N + 1) / (1 + len(docs))) + 1

        # Calculate TF-IDF and create a new DataFrame
        tfidf_matrix = collections.defaultdict(dict)
        for token, docs in tf.items():
            for index, tf_value in docs.items():
                tfidf_matrix[index][token] = tf_value * idf[token]

        # for i in tfidf_matrix:
        #     print(tfidf_matrix[i])
        #     assert len(set(tfidf_matrix[i].values())) == 1
        # # Convert the TF-IDF matrix to a DataFrame
        # tfidf_df = pd.DataFrame.from_dict(tfidf_matrix, orient='index').fillna(0);
        result = tfidf_matrix;

        return result;

    def get_weighted_embeddings_X_tfidf(self, arg_token_embed_weight, arg_tfidf_matrix):

        for index in range(len(arg_token_embed_weight)):
            row = arg_token_embed_weight[index];
            tokens = row['tokens'];
            for token in tokens:
                row['weights'].append(arg_tfidf_matrix[index][token]);

        result = [];
        for index in range(len(arg_token_embed_weight)):
            row = arg_token_embed_weight[index];
            temp_sum = 0;
            for j,v in enumerate(row['weights']):
                temp_sum += v * row['embeddings'][j];

            # assert np.linalg.norm(temp_sum - np.mean(row['embeddings']), axis = 0) < 1e-1

            result.append(temp_sum/sum(row['weights']))

        return result;

    def get_weighted_embeddings_X_pole(self, arg_token_embed_weight, arg_correlation = 'corr'):

        len_each_row = [len(arg_token_embed_weight[index]["tokens"]) for index in range(len(arg_token_embed_weight))]
        self.training_data['len_row'] = len_each_row;

        avg_std_each_row = [];
        avg_cor_each_row = [];

        for index in range(len(arg_token_embed_weight)):
            row = arg_token_embed_weight[index];
            embed = row['embeddings'];
            if arg_correlation != 'corr':
                correlation_matrix = np.corrcoef(np.vstack(embed));
            else:
                correlation_matrix = cosine_similarity(np.array(embed));

            std_deviations = np.array([np.std([correlation_matrix[i, j] for j in range(correlation_matrix.shape[1]) if i != j]) \
                                       for i in range(correlation_matrix.shape[0])])
            abs_corr_list = np.mean(np.abs(correlation_matrix), axis=0);
            row['weights'] = list(std_deviations + 0*abs_corr_list);

            # print(len(std_deviations), {row['tokens'][i]: np.mean(np.abs(correlation_matrix), axis=0)[i] for i in range(correlation_matrix.shape[0])})
            # print({row['tokens'][i]: std_deviations[i] for i in range(correlation_matrix.shape[0])})

            avg_std_each_row.append(np.mean(std_deviations));
            avg_cor_each_row.append(np.mean(abs_corr_list));
        self.training_data['std_row'] = avg_std_each_row;
        self.training_data['cor_row'] = avg_cor_each_row;

        result = [];
        for index in range(len(arg_token_embed_weight)):
            row = arg_token_embed_weight[index];
            temp_sum = 0;
            for j,v in enumerate(row['weights']):
                temp_sum += v * row['embeddings'][j];
            result.append(temp_sum/sum(row['weights']))

        return result;

    def get_weighted_embeddings_X_maxmin(self, arg_token_embed_weight):

        # Function to count column-wise maxima and minima occurrences across all arrays

        def count_col_max_min(data, col_maxima, col_minima):
            total = [];
            max_counts = [];
            min_counts = [];
            ph = 0
            for line in data:
                max_c = (line == col_maxima).sum(axis=0)
                min_c = (line == col_minima).sum(axis=0)
                if max_c == 0:
                    max_c = ph;
                if min_c == 0:
                    min_c = ph;
                max_counts.append(max_c);
                min_counts.append(min_c);
                total.append(max_c + min_c);

            return max_counts, min_counts, total

        for index in range(len(arg_token_embed_weight)):
            row = arg_token_embed_weight[index];
            stacked_arrays = np.array(row['embeddings']);

            # Compute column-wise maxima and minima
            col_maxima = np.max(stacked_arrays, axis=0)
            col_minima = np.min(stacked_arrays, axis=0)

            # Calculate the counts of maxima and minima in each column
            max_counts, min_counts, total_ex = count_col_max_min(stacked_arrays, col_maxima, col_minima);
            row['weights'] = np.array(total_ex);

        result = [];
        for index in range(len(arg_token_embed_weight)):
            row = arg_token_embed_weight[index];
            temp_sum = 0;
            for j,v in enumerate(row['weights']):
                temp_sum += v * row['embeddings'][j];
            result.append(temp_sum/sum(row['weights']))

        return result;




    def get_weighted_embeddings_X_add(self, arg_token_embed_weight):

        ## read data
        training_data_filename = parameters_configs['training_data_filename1']
        self.trigger_data = pd.read_csv(training_data_filename, sep='\t');
        ## get data as lists.
        texts = self.trigger_data['Texts'];
        triggers = self.trigger_data['Labels'].apply(ast.literal_eval);

        arg_nlp = spacy.load('en_core_web_sm');

        '''make sure the it parses the sentence based on the space splits'''
        def custom_tokenizer(text):
            words = text.split()  # Split text by spaces
            spaces = [True] * (len(words) - 1) + [False]  # Space is present after each word except the last one
            return spacy.tokens.Doc(arg_nlp.vocab, words=words, spaces=spaces)
        # Set the custom tokenizer to the nlp object
        arg_nlp.tokenizer = custom_tokenizer

        ## get tokens
        def extract_trigger_words(texts, triggers):
            # This list will hold dictionaries for each word and its trigger status
            data = []

            # print(list(zip(texts,triggers)))

            # Iterate over each text and corresponding trigger string
            for text, trigger in zip(texts, triggers):
                # Split the text into words and the trigger into individual numbers
                words = text.split()
                trigger_flags = trigger # trigger.split(); # already a list no need to split

                # Safety check to avoid index errors
                if len(words) != len(trigger_flags):
                    print("Warning: Mismatched lengths in text and triggers.")
                    data.append({});
                    continue;

                # print(list(zip(words, trigger_flags)))
                # Append each word and its trigger status to the data list
                row_result = {};
                for word, flag in zip(words, trigger_flags):
                    row_result[word] = int(flag)  # Convert flag to integer (0 or 1)

                data.append(row_result);

            return data;

        loc_dict = extract_trigger_words(texts, triggers);
        print(len(loc_dict))

        all_P_matrices = [];
        doc_idx = 0;
        for doc in arg_nlp.pipe(texts, disable=["parser", "ner"]):
            tokens = [token for token in doc]
            token_embeddings_in_doc = [];
            tokens_in_doc = [];
            for token in tokens:
                """still problems with matching the tokens if using [token.text]"""
                if loc_dict[doc_idx].get(token.text) == 1:
                    if not token.has_vector:
                        print("NO!")
                    tokens_in_doc.append(token.text);
                    token_embeddings_in_doc.append(token.vector);
            # Calculate the projection matrix P
            if not token_embeddings_in_doc or loc_dict[doc_idx] == {}:
                # A = np.zeros((96, 96));
                P = np.zeros((96, 96));
            else:
                A = np.column_stack(token_embeddings_in_doc);
                if np.sum(A) == 0:
                    P = np.zeros((96, 96));
                else:
                    P = A @ np.linalg.pinv(A.T @ A) @ A.T
            all_P_matrices.append(P);
            doc_idx += 1;


        # Calculate the projection of each w_i and their distance from the subspace
        for index in range(len(arg_token_embed_weight)):
            row = arg_token_embed_weight[index];

            es = np.stack(row['embeddings']);
            distances = [];
            for e in all_P_matrices[:]:
                p = es @ e;
                # distance = np.linalg.norm(w - p)
                # distances.append(distance);
                distance = np.linalg.norm(p, axis = 1);
                distances.append(distance);
            # print(len(list(np.mean(np.stack(distances), axis = 0))))
            # print(np.mean(np.stack(distances), axis = 0).shape)
            row['weights']=list(np.mean(np.stack(distances), axis = 0));

        result = [];
        for index in range(len(arg_token_embed_weight)):
            row = arg_token_embed_weight[index];
            temp_sum = 0;
            for j,v in enumerate(row['weights']):
                temp_sum += v * row['embeddings'][j];
            result.append(temp_sum/sum(row['weights']))

        return result;

    def analyze(self):
        """
        TODO all post analyse
        """
        return;

    def save(self, parameters_config):
        pass;

