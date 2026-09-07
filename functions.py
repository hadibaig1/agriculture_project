# We are going to train our Models with Temporal Cross Validation, or LOYO (Leave One Year Out)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix # We are going to use accuracy and Conf Matrix as our MEtrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost

# Making a fucniton we can use reusably when training different Models
def train_xgb(params, full_df):
    # Setting up our plotting config
    fig, axes  = plt.subplots(3,2, figsize=(15, 15))
    axes = axes.flatten()


    # Storing our metrics in the following dicts
    METRICS = {}
    accuracies = []
    train_accuracies = []

    # Storing our Feature Impiortances
    feature_importances = {}

    # Getting our unique years
    years = list(set(full_df['year']))
    years

    for i, _ in enumerate(years):
        ax = axes[i]
        testing_years_dict = {
            0: 2018,
            1: 2019,
            2: 2020,
            3: 2021,
            4: 2022
        }
        # We are going to select one of the years to validation on
        # We are going to go from validating on 2018-2022 sequentially
        year_test = testing_years_dict[i]
        print(f'Training for Year: {year_test}...')


        # We are going to split into train and test
        # We dont need to standardize our data, since TREE-BASED model dont need standardized data
        # Filtering by LOYO (Leave One Year Out)
        training_data = full_df[full_df['year'] != year_test]
        validation_data = full_df[full_df['year'] == year_test]

        #------TEMPORAL CV-------------------
        # Now we are going to split the features with the target label.
        # Our target label is 'class', which has the following values:
        # {0: 'Barley', 1: 'Oats', 2: 'Winter Wheat', 3: 'Spring Wheat'}

        X_train = training_data.drop(columns=['class', 'year'])
        y_train = training_data['class']

        X_test = validation_data.drop(columns=['class', 'year'])
        y_test = validation_data['class']

        # We are going to split into validation set, so we can use it for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, train_size=0.85, shuffle=True, random_state=42
        )


        # Instanitaiting our Model
        xgb_model = xgboost.XGBClassifier(**params, early_stopping_rounds=50)

        # Training our XGB Model using the XGBoost API
        xgb_model.fit(
            X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)], # We use validation sets here, to avoid data leakage.
        )

        # Evaluating
        y_pred = xgb_model.predict(X_test)
        y_pred_train = xgb_model.predict(X_train)

        accuracy_test = accuracy_score(y_test, y_pred)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        # Appending metrics
        accuracies.append(accuracy_test)
        train_accuracies.append(accuracy_train)
        METRICS[year_test] = accuracy_test

        matrix= confusion_matrix(y_test, y_pred)

        # Visualizing our Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, 
                                      display_labels=['Barley', 'Oats', 'Winter_Wheat', 'Spring_Wheat'])
        ax.set_title(f'Conf Matrix for {year_test}. Accuracy: {accuracy_test}')
        disp.plot(ax=ax)

         # Appending our feature importances
        feature_importances[year_test] = xgb_model.feature_importances_


    # We have 6 plots. but only using 5, so we hide the unused one
    axes[5].set_visible(False)
    plt.tight_layout()
    plt.show()

    print(f'MEAN ACCURACY: {np.mean(accuracies)}')
    print(f'MEAN TRAIN ACCURACY: {np.mean(train_accuracies)}')
    return feature_importances



# We are going to train our Models with Temporal Cross Validation, or LOYO (Leave One Year Out)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix # We are going to use accuracy and Conf Matrix as our MEtrics
import matplotlib.pyplot as plt
import numpy as np

# Making a fucniton we can use reusably when training different Models
def train_model(model, full_df):
    # Setting up our plotting config
    fig, axes  = plt.subplots(3,2, figsize=(15, 15))
    axes = axes.flatten()


    # Storing our metrics in the following dicts
    METRICS = {}
    accuracies = []

    # Getting our unique years
    years = list(set(full_df['year']))
    years

    for i, _ in enumerate(years):
        ax = axes[i]
        testing_years_dict = {
            0: 2018,
            1: 2019,
            2: 2020,
            3: 2021,
            4: 2022
        }
        # We are going to select one of the years to validation on
        # We are going to go from validating on 2018-2022 sequentially
        year_test = testing_years_dict[i]
        print(f'Training for Year: {year_test}...')


        # We are going to split into train and test
        # We dont need to standardize our data, since TREE-BASED model dont need standardized data
        # Filtering by LOYO (Leave One Year Out)
        training_data = full_df[full_df['year'] != year_test]
        validation_data = full_df[full_df['year'] == year_test]

        #------TEMPORAL CV-------------------
        # Now we are going to split the features with the target label.
        # Our target label is 'class', which has the following values:
        # {0: 'Barley', 1: 'Oats', 2: 'Winter Wheat', 3: 'Spring Wheat'}

        X_train = training_data.drop(columns=['class', 'year'])
        y_train = training_data['class']

        # Trainging Our Model
        model.fit(X_train, y_train)

        # Getting our Validation results
        X_test = validation_data.drop(columns=['class', 'year'])
        y_test = validation_data['class']

        y_pred = model.predict(X_test)

        #-------METRICS-----------
        accuracy = accuracy_score(y_test, y_pred)

        # appending our Accuracy to our MEtrics dictionary
        METRICS[year_test] = accuracy
        accuracies.append(accuracy)

        matrix = confusion_matrix(y_test, y_pred)
        disp  =ConfusionMatrixDisplay(confusion_matrix=matrix, 
                                    display_labels=['Barley', 'Oats', 'Winter_Wheat', 'Spring_Wheat'])
        ax.set_title(f'Confusion Matrix for: {year_test}. Accuracy: {accuracy}')
        disp.plot(ax=ax)

    # We have 6 plots. but only using 5, so we hide the unused one
    axes[5].set_visible(False)
    plt.tight_layout()
    plt.show()

    print(f'MEAN ACCURACY: {np.mean(accuracies)}')
    return METRICS




# Performing Hyperaprmeter tuning with Optuna
import optuna
from optuna.samplers import TPESampler

# Defingin our objective funciton
def objective(trial, full_df):
    # Defining the param search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.005, 0.2, log=True
        ),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-7, 1.0, log=True),
        'random_state': 42,
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 0.3)
    }

    unique_years = list(set(full_df['year']))
    accuracies = []
    train_accs = []
    # 2. Run LOYO cross-validation loop to evaluate selected hyperparameter set
    for step, year_test in enumerate(unique_years):
        train_data = full_df[full_df['year'] != year_test]
        test_data = full_df[full_df['year'] == year_test]

        X_train = train_data.drop(columns=['class', 'year'])
        y_train = train_data['class']
        X_test = test_data.drop(columns=['class', 'year'])
        y_test = test_data['class']

        # 3. Train model on fold training data using trial hyperparameters
        model = xgboost.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Predict on holdout year and record fold accuracy
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

        #5. We are also going to recorde the accuracy on our Train set, to check if our Model is overfitting
        y_pred_train = model.predict(X_train)
        train_accs.append(accuracy_score(y_train, y_pred_train))

        # Now we are going to be Pruning bad trials
        intermediate_value = np.mean(accuracies)
        trial.report(intermediate_value, step)

        # Checking if we should prune or not
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned
    # 6. Return mean LOYO cross-validation accuracy to Optuna for maximization
    return np.mean(accuracies)






