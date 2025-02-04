"""
Instructions:

- Please upload a requirements.txt file with any additional packages you want to use.
- Please upload a README.md along with your solution that contains an overview of the
  solution you implemented.
- Fill in the methods of the DataModeler class to produce the same printed results
  as in the comments labeled '<Expected Output>' in the second half of the file.
- The DataModeler should predict the 'outcome' from the columns 'amount' and 'transaction date.'
  Your model should ignore the 'customer_id' column.
- For the modeling methods `fit`, `predict` and `model_summary` you can use any appropriate method.
  Try to get 100% accuracy on both training and test, as indicated in the output.
- Your solution will be judged on correctness, code quality, and quality of the documentation.
- Good luck, and have fun!

"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from datetime import datetime


class DataModeler:
    def __init__(self, sample_df: pd.DataFrame):
        """
        This implementation uses an SVM with a large C and auto-determined gamma (RBF kernel).
        It aims to perfectly memorize the small training dataset, potentially reaching 100% accuracy
        on both the training and a similarly distributed test sample.
        """
        self.train_df = sample_df.copy()
        self.original_df = sample_df.copy()
        self.model = SVC(
            kernel="rbf",
            C=1e6,  # Large C to minimize the margin, aiming for perfect classification
            gamma="auto",
            random_state=42,
        )
        self._fitted = False
        self._date_means = None
        self._amount_mean = None

    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Convert transaction_date from string to numeric timestamps,
        remove the customer_id column, and keep only amount and transaction_date.
        """
        df = oos_df.copy() if oos_df is not None else self.train_df.copy()

        def convert_date(date_str):
            if pd.isna(date_str):
                return np.nan
            return datetime.strptime(date_str, "%Y-%m-%d").timestamp()

        df["transaction_date"] = df["transaction_date"].apply(convert_date)

        if oos_df is None:
            self.train_df = df.set_index("customer_id")[["amount", "transaction_date"]]
            return self.train_df
        else:
            return df.set_index("customer_id")[["amount", "transaction_date"]]

    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fill any missing values in amount and transaction_date with their respective means.
        If oos_df is None, modify self.train_df. Otherwise, modify the provided out-of-sample df.
        """
        df = oos_df if oos_df is not None else self.train_df

        if oos_df is None:
            self._amount_mean = df["amount"].mean()
            self._date_means = df["transaction_date"].mean()
            self.train_df["amount"] = df["amount"].fillna(self._amount_mean)
            self.train_df["transaction_date"] = df["transaction_date"].fillna(
                self._date_means
            )
            return self.train_df
        else:
            df["amount"] = df["amount"].fillna(self._amount_mean)
            df["transaction_date"] = df["transaction_date"].fillna(self._date_means)
            return df

    def fit(self) -> None:
        """
        Train the SVM on the prepared and imputed training data.
        The original_df contains the "outcome" column, which is our target.
        """
        X = self.train_df[["amount", "transaction_date"]]
        y = self.original_df["outcome"]
        self.model.fit(X, y)
        self._fitted = True

    def model_summary(self) -> str:
        """
        Provide a short summary of the fitted SVM.
        """
        if not self._fitted:
            return "Model not yet fitted"
        return (
            "SVM Classifier (RBF kernel)\n"
            f"C: {self.model.C}, gamma: {self.model.gamma}\n"
        )

    def predict(self, oos_df: pd.DataFrame = None) -> pd.Series[bool]:
        """
        Predict using the fitted SVM, returning a boolean Series. If oos_df is None,
        predict on the training set. Otherwise, predict on the provided df.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        X = oos_df if oos_df is not None else self.train_df
        preds = self.model.predict(X[["amount", "transaction_date"]])
        return pd.Series(preds, index=X.index, dtype=bool)

    def save(self, path: str) -> None:
        """
        Save the current DataModeler state (including the SVM, means, training data, etc.)
        for future reloading.
        """
        save_dict = {
            "model": self.model,
            "date_means": self._date_means,
            "fitted": self._fitted,
            "amount_mean": self._amount_mean,
            "original_df": self.original_df,
            "train_df": self.train_df,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @staticmethod
    def load(path: str) -> DataModeler:
        """
        Load a previously saved DataModeler object from the given path.
        """
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        dummy_df = pd.DataFrame()
        modeler = DataModeler(dummy_df)
        modeler.model = save_dict["model"]
        modeler._amount_mean = save_dict["amount_mean"]
        modeler._date_means = save_dict["date_means"]
        modeler.train_df = save_dict["train_df"]
        modeler._fitted = save_dict["fitted"]
        modeler.original_df = save_dict["original_df"]
        return modeler


#################################################################################
# You should not have to modify the code below this point

transact_train_sample = pd.DataFrame(
    {
        "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
        "transaction_date": [
            "2022-01-01",
            "2022-08-01",
            None,
            "2022-12-01",
            "2022-02-01",
            None,
            "2022-02-01",
            "2022-01-01",
            "2022-11-01",
            "2022-01-01",
        ],
        "outcome": [False, True, True, True, False, False, True, True, True, False],
    }
)


print(f"Training sample:\n{transact_train_sample}\n")

# <Expected Output>
# Training sample:
#    customer_id  amount transaction_date  outcome
# 0           11     1.0       2022-01-01    False
# 1           12     3.0       2022-08-01     True
# 2           13    12.0             None     True
# 3           14     6.0       2022-12-01     True
# 4           15     0.5       2022-02-01    False
# 5           16     0.2             None    False
# 6           17     NaN       2022-02-01     True
# 7           18     5.0       2022-01-01     True
# 8           19     NaN       2022-11-01     True
# 9           20     3.0       2022-01-01    False


print(f"Current dtypes:\n{transact_train_sample.dtypes}\n")

# <Expected Output>
# Current dtypes:
# customer_id           int64
# amount              float64
# transaction_date     object
# outcome                bool
# dtype: object

transactions_modeler = DataModeler(transact_train_sample)

transactions_modeler.prepare_data()

print(f"Changed columns to:\n{transactions_modeler.train_df.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

transactions_modeler.impute_missing()

print(f"Imputed missing as mean:\n{transactions_modeler.train_df}\n")

# <Expected Output>
# Imputed missing as mean:
#               amount  transaction_date
# customer_id
# 11            1.0000      1.640995e+18
# 12            3.0000      1.659312e+18
# 13           12.0000      1.650845e+18
# 14            6.0000      1.669853e+18
# 15            0.5000      1.643674e+18
# 16            0.2000      1.650845e+18
# 17            3.8375      1.643674e+18
# 18            5.0000      1.640995e+18
# 19            3.8375      1.667261e+18
# 20            3.0000      1.640995e+18


print("Fitting  model")
transactions_modeler.fit()

print(f"Fit model:\n{transactions_modeler.model_summary()}\n")

# <Expected Output>
# Fitting  model
# Fit model:
# <<< ANY SHORT SUMMARY OF THE MODEL YOU CHOSE >>>

in_sample_predictions = transactions_modeler.predict()
print(f"Predicted on training sample: {in_sample_predictions}\n")
print(
    f"Accuracy = {sum(in_sample_predictions ==  [False, True, True, True, False, False, True, True, True, False])/.1}%"
)

# <Expected Output>
# Predicting on training sample [False  True  True  True False False True  True  True False]
# Accuracy = 100.0%

transactions_modeler.save("transact_modeler")
loaded_modeler = DataModeler.load("transact_modeler")

print(f"Loaded DataModeler sample df:\n{loaded_modeler.model_summary()}\n")

# <Expected Output>
# Loaded DataModeler sample df:
# <<< THE SUMMARY OF THE MODEL YOU CHOSE >>>

transact_test_sample = pd.DataFrame(
    {
        "customer_id": [21, 22, 23, 24, 25],
        "amount": [0.5, np.nan, 8, 3, 2],
        "transaction_date": [
            "2022-02-01",
            "2022-11-01",
            "2022-06-01",
            None,
            "2022-02-01",
        ],
    }
)

adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)

print(f"Changed columns to:\n{adjusted_test_sample.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)

print(f"Imputed missing as mean:\n{filled_test_sample}\n")

# <Expected Output>
# Imputed missing as mean:
#              amount  transaction_date
# customer_id
# 21           0.5000      1.643674e+18
# 22           3.8375      1.667261e+18
# 23           8.0000      1.654042e+18
# 24           3.0000      1.650845e+18
# 25           2.0000      1.643674e+18

oos_predictions = transactions_modeler.predict(filled_test_sample)
print(f"Predicted on out of sample data: {oos_predictions}\n")
print(f"Accuracy = {sum(oos_predictions == [False, True, True, False, False])/.05}%")

# <Expected Output>
# Predicted on out of sample data: [False True True False False] ([0 1 1 0 0])
# Accuracy = 100.0%
