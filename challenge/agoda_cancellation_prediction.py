from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

from sklearn import model_selection
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score

import re
import numpy as np
import pandas as pd
import datetime


ID_COLS = ['h_booking_id', 'hotel_id', 'hotel_country_code', 'h_customer_id']

DATETIME_COLS = ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date']

CODES_COLS = ['origin_country_code', 'hotel_area_code', 'hotel_city_code']

# (dropped) 'hotel_brand_code', 'hotel_chain_code' (have ~43K nulls!)


CATEGORICAL_COLS = ['accommadation_type_name', 'charge_option', 'customer_nationality',
                    'guest_nationality_country_name', 'language', 'original_payment_method',
                    'original_payment_type', 'original_payment_currency']

NUMERICAL_COLS = ['hotel_star_rating', 'no_of_adults',  'no_of_children', 'no_of_extra_bed', 'no_of_room',
                  'original_selling_amount']

SHOULD_BE_BOOLEAN_COLS = ['guest_is_not_the_customer', 'request_nonesmoke', 'request_latecheckin',
                          'request_highfloor', 'request_largebed', 'request_twinbeds', 'request_airport',
                          'request_earlycheckin']

# The following columns have 25040 nulls:
# request_nonesmoke, request_latecheckin, request_highfloor, request_largebed, request_twinbeds,
# request_airport, request_earlycheckin
# remove them - or fill in nan with 0

BOOLEAN_COLS = ['is_user_logged_in', 'is_first_booking']

LABEL_COL = 'cancellation_datetime'

NO_SHOW_PATTERN = '_(\d+)(N|P)'
POLICY_PATTERN = '(\d+)D(\d+)(N|P)'


def parse_cancellation_policy_no_show(data):
    print("parse_cancellation_policy_no_show")
    for i, row in data.iterrows():
        policy = row["cancellation_policy_code"]
        n_nights = row["n_nights"]

        no_show = re.findall(NO_SHOW_PATTERN, policy)
        if no_show:
            if no_show[0][1] == "N":
                days = int(no_show[0][0])
                percent = int(no_show[0][0]) / n_nights * 100
            else:
                days = int(no_show[0][0]) * n_nights / 100
                percent = int(no_show[0][0])

        else:
            worse_policy_without_no_show = re.findall(POLICY_PATTERN, policy)[-1]

            if worse_policy_without_no_show[-1] == "N":
                days = int(worse_policy_without_no_show[1])
                percent = int(worse_policy_without_no_show[1]) / n_nights * 100
            else:
                days = int(worse_policy_without_no_show[1]) * n_nights / 100
                percent = int(worse_policy_without_no_show[1])

        data.loc[i, "no_show_days"] = days
        data.loc[i, "no_show_percentage"] = percent
    print("Done...")
    return data


def parse_cancellation_policy(data):
    print("parse_cancellation_policy")
    for i, row in data.iterrows():
        policy = row["cancellation_policy_code"]
        n_nights = row["n_nights"]

        cancel_policy = re.findall(POLICY_PATTERN, policy)
        worse_policy = cancel_policy[-1]
        basic_policy = cancel_policy[-2] if len(cancel_policy) > 1 else worse_policy

        if worse_policy[2] == "N":
            nights = int(worse_policy[1])
            percent = int(worse_policy[1]) / n_nights * 100
        else:
            nights = int(worse_policy[1]) * n_nights / 100
            percent = int(worse_policy[1])
        if basic_policy[2] == "N":
            basic_by_nights = int(basic_policy[1])
            basic_percent = int(basic_policy[1]) / n_nights * 100
        else:
            basic_by_nights = int(basic_policy[1]) * n_nights / 100
            basic_percent = int(basic_policy[1])

        days = int(worse_policy[0])
        basic_days = int(basic_policy[0])
        data.loc[i, "basic_charge_percentage"] = basic_percent
        data.loc[i, "basic_charge_by_nights"] = basic_by_nights
        data.loc[i, "basic_charge_days"] = basic_days
        #data.loc[i, "basic_charge_days_times_nights"] = basic_days * basic_by_nights
        #data.loc[i, "basic_charge_days_times_percentage"] = basic_days * basic_percent
        data.loc[i, "charge_percentage"] = percent
        data.loc[i, "charge_by_nights"] = nights
        data.loc[i, "charge_days"] = days
        #data.loc[i, "charge_days_times_nights"] = days * nights
        #data.loc[i, "charge_days_times_percentage"] = days * percent
    print("Done...")
    return data


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    parsed_data = parse_data(full_data)

    features, labels = parsed_data.loc[:, parsed_data.columns != LABEL_COL], parsed_data[LABEL_COL]
    print("Finished load data. Features:", features.shape, "lables:", labels.shape)
    return features, labels

# todo uncomment and use!
def evaluate_and_export(estimator: AgodaCancellationEstimator, X: np.ndarray, y_true, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    predictions = pd.DataFrame(estimator.predict(X))
    conf_matrix = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = conf_matrix.ravel()
    print("Confusion matrix: ")
    print(conf_matrix)

    print("roc_auc_score", roc_auc_score(y_true, predictions))
    print("Accuracy", accuracy_score(y_true, predictions))

    print(f"True negative: {tn}, False positive: {fp}, False Negative: {fn}, True Positive: {tp}")
    #pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)

# todo delete
def parse_gilad(full_data):
    data = full_data.loc[:, ['charge_option', 'no_of_room', 'request_largebed', 'request_twinbeds',
                             LABEL_COL]]

    data = data.dropna().drop_duplicates()
    labels = data["cancellation_datetime"].between("2018-07-12", "2018-13-12").astype('int')


    data = pd.get_dummies(data, columns=['charge_option'], drop_first=True)

    features = data.drop(LABEL_COL, axis=1)

    return features, labels


def parse_data(full_data):
    # choose only numerical, dates, boolean and label:
    data = full_data.loc[:, NUMERICAL_COLS + DATETIME_COLS + SHOULD_BE_BOOLEAN_COLS + BOOLEAN_COLS +
                            CATEGORICAL_COLS + ["cancellation_policy_code"] + [LABEL_COL]]

    data = data.dropna().drop_duplicates()

    # handle labels
    # min_date = datetime.date(2018, 12, 7)
    # max_date = datetime.date(2018, 12, 13)

    # todo not sure which pne to use
    # data.loc[:, LABEL_COL] = pd.to_datetime(data[LABEL_COL]).between(min_date, max_date).astype('int')
    data.loc[:, LABEL_COL] = data.loc[:, LABEL_COL].between("2018-07-12", "2018-13-12").astype('int')

    data = data.drop(data[data["cancellation_policy_code"] == "UNKNOWN"].index)

    # fill in 0 instead of None's
    for col_name in SHOULD_BE_BOOLEAN_COLS:
        data.loc[:, col_name] = data.loc[:, col_name].fillna(0)

    # handle boolean cols
    for col_name in SHOULD_BE_BOOLEAN_COLS:
        data.loc[:, col_name] = data.loc[:, col_name] == 1  # todo use astype instead?

    # handle datetime cols (convert to timestamps)
    for col_name in DATETIME_COLS:
        as_datetime = pd.to_datetime(data[col_name])
        data.loc[:, col_name + "_year"] = as_datetime.dt.year
        data.loc[:, col_name + "_month"] = as_datetime.dt.month
        data.loc[:, col_name + "_day"] = as_datetime.dt.day
        data.loc[:, col_name + "_day_in_week"] = as_datetime.dt.day_of_week

    data.loc[:, "n_nights"] = (pd.to_datetime(full_data["checkout_date"]) -
                               pd.to_datetime(full_data["checkin_date"])).dt.days
    data.loc[:, "n_days_from_booking_to_checkin"] = (pd.to_datetime(full_data["checkin_date"]) -
                                                     pd.to_datetime(full_data["booking_datetime"])).dt.days
    data = data.drop(DATETIME_COLS, axis=1)

    # replace categorical features with their dummies
    data = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=True)

    # todo not sure where to do this
    data = data.dropna().drop_duplicates()

    data = parse_cancellation_policy_no_show(data)
    data = parse_cancellation_policy(data)

    data = data.drop(['cancellation_policy_code'], axis=1)

    print("After preprocessing:")
    print(data.shape)
    print(data.columns)
    print(data.head())
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")

    train_X, test_X, train_y, test_y = model_selection.train_test_split(df, cancellation_labels,
                                                                        test_size=0.25, random_state=0)

    print("Train shape", train_X.shape, train_y.shape)

    # # # Fit model over data
    estimator = AgodaCancellationEstimator()
    estimator.fit(train_X, train_y)
    #
    # # Store model predictions over test set
    print("Misclasification", estimator.loss(test_X, test_y))  # todo this goes none, maybe in the loss we
    # need to return sklearn.loss ?
    evaluate_and_export(estimator, test_X, test_y, "id1_id2_id3.csv") # todo edit

    print("DONE")
