import argparse
import json
# import defined models here

def main():
    # add imported models into models
    models = []
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('-mn', '--model_name', choices=[m.__name__ for m in models], required=True)
    parser.add_argument('-mc', '--model_config', required=True)
    parser.add_argument('-tc', '--train_config', required=True)
    parser.add_argument('-pc', '--parameters_config', required=True)
    args = parser.parse_args()
    with open(args.model_config, 'r') as mc:
        model_config = json.load(mc)
    with open(args.train_config, 'r') as tc:
        train_config = json.load(tc)
    with open(args.parameters_config, 'r') as pc:
        parameters_config = json.load(pc)
    # initialize model
    for m in models:
        if m.__name__ == args.model_name:
            model = m(model_config)
            break
    # if model_parameters is set, load model parameters
    model.train(train_config)
    model.save(parameters_config)

if __name__ == '__main__':
    main()