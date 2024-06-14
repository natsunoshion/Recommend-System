import os
from utils import *
from item import *
from user import *

if __name__ == '__main__':
    base_dir = './experiments/usercf'

    is_retrain = input("Do you want to retrain the model? (y/n): ").strip().lower()
    assert is_retrain in ['y', 'n'], "Invalid input! Please type 'y' or 'n'."

    if is_retrain == 'y':
        archive_existing_directory(base_dir)

    log_directory = os.path.join(base_dir, 'logs')
    model_directory = os.path.join(base_dir, 'models')

    setup_logging(log_directory, 'training')

    model_path = os.path.join(model_directory, 'model.pkl')
    trained_model_path = os.path.join(model_directory, 'model_trained.pkl')
    tested_model_path = os.path.join(model_directory, 'model_tested.pkl')

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    if is_retrain == 'y':
        model = UserCF('./datasets/train.txt', './datasets/test.txt', model_directory)
        model.build(model.train_p)
        save_model(model, model_path)
        model.static_analyse()
        model.train()
        save_model(model, trained_model_path)
        model.test(model.test_p)
        save_model(model, tested_model_path)
    else:
        model = load_model(trained_model_path)
        assert model is not None, "Failed to load the trained model!"
        if not model.is_build:
            model.build('./datasets/train.txt')
            model.train()
            save_model(model, model_path)
        elif not model.is_train:
            model.train()
            save_model(model, trained_model_path)
        elif not model.is_test:
            model.test('./datasets/test.txt')
            save_model(model, tested_model_path)
        else:
            assert False, "Unexpected state: Model is already built, trained, and tested."