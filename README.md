# ling573-project

## Environment Setup

<!-- First create project virtualenv, `573project`, to ensure you have the right packages.

1. `python3 -m venv 573project`
2. Activate the virtual environment:
On Windows:
`573project\Scripts\activate`
On macOS/Linux:
`source 573project/bin/activate`
3. Install required packages: `pip install -r requirements.txt`
4. **Make sure to do this whenever you install a new package, to keep the team's virtualenv in-sync**: `pip freeze > requirements.txt`  -->

 First create project virtualenv, `573-env`, to ensure you have the right packages.
1. Create the virtual environment: 
`conda env create -f environment.yml`
2. Activate the virtual environment:
`conda activate 573-env`
3. **Make sure to do this whenever you install a new package, to keep the team's virtualenv in-sync**: 
`echo -e "    - new_package==version" >> environment.yml`

4. You can use `conda env update --file environment.yml --prune` to update the existing environment to match the specifications in the `environment.yml` file, adding any new packages and removing any packages that are no longer listed in the file. The `--prune` option ensures that any packages not listed in the `environment.yml` file are removed from the environment. 
<!--1. Activate the virtual environment on patas:
`conda activate /nopt/dropbox/23-24/570/envs/570`
2. Install the OpenAI package:
`pip install openai` -->
Next, download the required files from https://drive.google.com/drive/u/3/folders/1whNEQpjhCW_LDK8M02qPWewVAoln946Y and put them into the subdirectory, `data/`. These files are necessary for running BERT-KNN but could not be uploaded to Github.

## D4 Instructions
Here we only mention the differences with respect to D2 and D3.
### Running Agentic Workflow Trigger Model
```
python inference.py -mn AWTrigger -mc configs/model_configs/agentic_workflow_trigger.json -ic configs/inference_configs/agentic_workflow_trigger.json
```

## D3 Instructions
Here we only mention the differences with respect to D2.

### Running Models for D3
Use `run_d3_ensembles.sh` under the `src/` directory and simply run
```
./run_d3_ensembles.sh
```
After the run finishes, you should be able to see all the outputs under `outputs/D3/ensemble/` for this specific run with configs defined for D3.
Please follow the same instruction of D2 to set up the OpenAI API.
### (Optional) Running Explain Zero Shot Model
This model costs more in terms of both money and time (around 1.5 hours) to run. As discussed in the email, there is no need to re-run it and we will provide updates in our report regarding this improvement.

Run the following command uner the `src/` directory
```
python inference.py -mn ExplainZeroShotGPT -mc configs/model_configs/explain_zero_shot_gpt.json -ic configs/inference_configs/explain_zero_shot_gpt_classification_d3_unlabelled.json
```
Note that in order to run this command, both OpenAI API key and Anthropic API key need to be set up.

## D2 Instructions
### Important Note for D2
Since we used OpenAI GPT models, the behavior could be flaky and differ a bit for each run.
Use `run_d2_model_configs_and_evaluations.sh` under the `src/` directory and simly run
```
./run_d2_model_configs_and_evaluations.sh
```
After the run finishes, you should be able to see all the outputs under `outputs/D2/D2_Outputs/` for this specific run with configs defined for D2.

### Setup for D2
Please follow the Environment Setup above. If there is no conda available on Patas, please follow:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
to get miniconda3.

For more details, see https://docs.anaconda.com/free/miniconda/

Since we use OpenAI models, before running the models, rememebr to run the following
```
export OPENAI_API_KEY='your-api-key-here'
```
to set up the correct key.

After running the models, please reset the key to empty string
```
export OPENAI_API_KEY=''
```
### Running Models for D2
Use `run_d2_model_configs_and_evaluations.sh` under the `src/` directory and simly run
```
./run_d2_model_configs_and_evaluations.sh
```
After the run finishes, you should be able to see all the outputs under `outputs/D2/D2_Outputs/` for this specific run with configs defined for D2.

To see the outputs with other configs, you can find them under `D2` and each model has its own directory for outputs, e.g. `D2/SimpleKNN/`.

Note: since there is some special dependencies used in the NaiveBayes model, we could not integrate it with other models as of now, please see the model in Colab https://colab.research.google.com/drive/1d6IfVodvKacbwNkSaE_J4Mzj4_F2xoeV?usp=sharing

## Running the Models
### Batch Model Configs Inference
Use `run_all_simple_knn_configs.sh` under the `src/` directory to run all configurations for the SimpleKNN model
```
./run_all_simple_knn_configs.sh
```
### Individual Model Config Inference
Use `inference.py` under the `src/` directory.

Sample usage for run inference on classification task using FewShotGPT
```
python inference.py -mn FewShotGPT -mc configs/model_configs/few_shot_gpt_classification.json -pc configs/parameters_configs/few_shot_gpt_classification.json -ic configs/inference_configs/few_shot_gpt_classification.json
```
Sample usage for run inference on trigger word dection task using FewShotGPT
```
python inference.py -mn FewShotGPT -mc configs/model_configs/zero_shot_gpt_trigger_dection.json -ic configs/inference_configs/zero_shot_gpt_trigger_detection.json
```
Sample usage for run inference on classification task using AgenticWorkflow
```
python inference.py -mn AWClassification -mc configs/model_configs/once_agentic_workflow_classification_both.json -pc configs/parameters_configs/once_agentic_workflow_classification.json -ic configs/inference_configs/once_agentic_workflow_classification_both.json
```
Sample usage for run inference on classification task using FineTuneGPT
```
python inference.py -mn FineTuneGPT -mc configs/model_configs/fine_tune_gpt_classification_both.json -ic configs/inference_configs/fine_tune_gpt_classification_both.json
```
Sample usage for run inference on classification task using BERTKNN. **Do not forget the required files needed to be downloaded in the environment setup!**

**If this command doesn't work (due to Patas downtime, we couldn't fully test after some changes), use this Colab notebook: https://colab.research.google.com/drive/1KOnXbKIuKDsjDaGs21QEQI25hbHg3nC0#scrollTo=oERluB2jyYnk**
```
python inference.py -mn BERTKNN -mc configs/model_configs/bert_knn_config.json -pc configs/parameters_configs/bert_knn_config.json -ic configs/inference_configs/bert_knn_config.json
```

<hr>

Sample usage for run inference on classification task using Naive Bayes (with a simple reweight on non-neutral classes)
```
python3 inference.py -mn NaiveBayesEmbeddingToy -mc configs/model_configs/naive_bayes_embedding_toy.json -pc configs/parameters_configs/naive_bayes_embedding_toy.json -ic configs/inference_configs/naive_bayes_embedding_toy.json
```

🎈 As for NaiveBayes, although it can be run using the above command line, we also provide a colab link since it is simply for experiment purpose.
🎈 https://colab.research.google.com/drive/1d6IfVodvKacbwNkSaE_J4Mzj4_F2xoeV?usp=sharing



<!--### Evaluation
1. Please check the `results/` directory to see the evaluations on the dev set.
2. Use `evaluate.py` under the `src/` directory to get the evaluations on the training set:

Sample usage for evaluating classification results
```
python evaluate.py -f /experiment/zero_shot_classification/classification_results -t classification
```
Sample usage for evaluating trigger detection results
```
python evaluate.py -f /experiment/zero_shot_classification/classification_results -t trigger_detection
```-->
<!-- ## Cross-lingual Emotion Detection
### Agentic Workflow
#### Get Emotion Detection Results
The complete execution procedure is in `src/agentic_workflow/Emotion.sh`. In terms of the names of config files in `src/agentic_workflow/config`: `*_none.json` means no preprocessing, `*_translate.json` means only translation, `*_clean.json` means only cleaning, and `*_both.json` means both preprocessings.
1. Train a fine-tuned model: 
```
python src/agentic_workflow/fine_tune.py
```
2. Use the fine-tuned model: 
```
python src/agentic_workflow/use_fine_tuned_model.py --config config/xxx.json
``` 
Here you should specify the name of the config file. The output tsv files will be in`/outputs/agentic_workflow`.

3. Use the agentic workflow: 
```
python src/agentic_workflow/emotion_detection.py --config config/xxx.json
``` 
Here you should specify the name of the config file. The output tsv files will be in`/outputs/agentic_workflow`.
#### Evaluation
Upload tsv files in`/outputs/agentic_workflow` to [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/17730#learn_the_details-submission-format) and get official scores. -->
