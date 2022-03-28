from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

from sklearn import model_selection

import numpy as np
import pandas as pd


ID_COLS = ['h_booking_id', 'hotel_id', 'hotel_country_code', 'h_customer_id']

DATETIME_COLS = ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date']

CODES_COLS = ['origin_country_code', 'hotel_area_code', 'hotel_brand_code', 'hotel_chain_code',
              'hotel_city_code']

CATEGORICAL_COLS = ['accommadation_type_name', 'charge_option', 'customer_nationality',
                    'guest_nationality_country_name' , 'language', 'original_payment_method',
                    'original_payment_type', 'original_payment_currency']

# 'cancellation_policy_code' !!

NUMERICAL_COLS = ['hotel_star_rating', 'no_of_adults',  'no_of_children', 'no_of_extra_bed', 'no_of_room',
                  'original_selling_amount']

SHOULD_BE_BOOLEAN_COLS = ['guest_is_not_the_customer', 'request_nonesmoke', 'request_latecheckin',
                          'request_highfloor', 'request_largebed', 'request_twinbeds', 'request_airport',
                          'request_earlycheckin']

BOOLEAN_COLS = ['is_user_logged_in', 'is_first_booking']

LABEL_COL = 'cancellation_datetime'


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
def evaluate_and_export(estimator: AgodaCancellationEstimator, X: np.ndarray, filename: str):
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
    print("Predictions!")
    print(pd.DataFrame(estimator.predict(X)))
    #pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


def parse_data(full_data):
    # choose only numerical, dates, boolean and label:
    data = full_data[NUMERICAL_COLS + DATETIME_COLS + SHOULD_BE_BOOLEAN_COLS + BOOLEAN_COLS +
                     CATEGORICAL_COLS + [LABEL_COL]]
    print("Choosing features", data.shape)

    # handle labels
    data[LABEL_COL] = data[LABEL_COL].notnull()

    data = data.dropna()  # maybe use inplace=True to avoid copies

    # handle boolean cols
    for col_name in SHOULD_BE_BOOLEAN_COLS:
        data[col_name] = data[col_name] == 1

    # handle datetime cols (convert to timestamps)
    for col_name in DATETIME_COLS:
        data[col_name] = pd.to_datetime(data[col_name]).apply(lambda x: x.value)

    # replace categorical features with their dummies
    data = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=True)

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

    # train_X, train_y = df, cancellation_labels  # todo use the split later?
    print(train_X.shape, train_y.shape)

    # # # Fit model over data
    estimator = AgodaCancellationEstimator()
    estimator.fit(train_X, train_y)
    #
    # # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")

    print("DONE")