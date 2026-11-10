# Data

The original educational telecom dataset is not redistributed. The pipelines accept an authorized pandas DataFrame containing the selected customer features and either `Churn` or `MonthlyCharge` as the target.

Keep local datasets under `data/raw/`; that directory is excluded from version control. The test suite uses synthetic records so repository validation does not require the original data.
