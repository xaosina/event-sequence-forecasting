import argparse
import subprocess
from pathlib import Path

from typing import Optional, Sequence
import numpy as np
import pandas as pd
from common import add_row_order, code_indexes, code_categories, save_to_parquet
from pyspark.sql import SparkSession
import logging

logging.basicConfig(level=logging.INFO)

METADATA = {
    "cat_features": [
        "currency",
        "operation_kind",
        "card_type",
        "operation_type",
        "operation_type_group",
        "ecommerce_flag",
        "payment_system",
        "income_flag",
        "mcc",
        "country",
        "city",
        "mcc_category",
    ],
    "num_features": [
        "day_of_week",
        "hour",
        "amnt",
        "days_before",
        "weekofyear",
        "hour_diff",
    ],
    "index_columns": ["app_id"],
    "target_columns": [],
    "ordering_columns": ["hours_since_first_tx", "transaction_number"],
}

SEQ_LEN_MIN = 128
SEQ_LEN_MAX = 1200
DIFF_MAX_LT = 800.0
DIFF_MEDIAN_LT = 70.0


def filter_user_seq_diff_bounds(df: pd.DataFrame) -> pd.DataFrame:
    user_col = METADATA["index_columns"][0]
    seq_len = df.groupby(user_col, sort=False).size().rename("n")
    agg = df.groupby(user_col, sort=False)["hour_diff"].agg(mx="max", md="median")
    st = seq_len.to_frame().join(agg, how="left")
    ok = (
        (st["n"] >= SEQ_LEN_MIN)
        & (st["n"] <= SEQ_LEN_MAX)
        & (st["mx"] < DIFF_MAX_LT)
        & (st["md"] < DIFF_MEDIAN_LT)
    )
    keep = st.loc[ok].index
    return df.loc[df[user_col].isin(keep)]


def select_clients(df: pd.DataFrame, feature_name, cn) -> pd.DataFrame:
    sampled_clients = np.random.choice(df[feature_name].unique(), cn, replace=False)
    return df[df[feature_name].isin(sampled_clients)]


def encode_data(train_data, test_data, path):

    df_concated = train_data.unionByName(test_data).repartition(256)

    _ = code_indexes(df_concated, METADATA["index_columns"], save_path=path)
    _ = code_categories(df_concated, METADATA["cat_features"], save_path=path)


def currency_drop(df: pd.DataFrame, remain_currency: int):
    currency_per_user = df.groupby("app_id")["currency"].nunique()
    keep_ids = currency_per_user[currency_per_user == remain_currency]
    return df[df["app_id"].isin(keep_ids)]


def feature_drop(
    df: pd.DataFrame,
    feature_name: str,
    drop_cats: Optional[Sequence] = None,
    remain_cats: Optional[Sequence] = None,
    user_col: str = "app_id",
) -> pd.DataFrame:
    assert (drop_cats is None) ^ (
        remain_cats is None
    ), "Задайте ровно один из параметров: drop_cats или remain_cats."

    if remain_cats is not None:
        allowed = set(remain_cats)
        mask_allowed = df[feature_name].isin(allowed)
        bad_ids = df.loc[
            ~mask_allowed, user_col
        ].unique()  # есть хоть одно запрещённое значение
        keep_mask = ~df[user_col].isin(bad_ids)
        return df.loc[keep_mask]

    # drop_cats is not None
    forbidden = set(drop_cats)
    bad_ids = df.loc[df[feature_name].isin(forbidden), user_col].unique()
    keep_mask = ~df[user_col].isin(bad_ids)
    return df.loc[keep_mask]


def extract_time_feature(df: pd.DataFrame):
    assert df["hour_diff"].min() == -1
    df["hour_diff"] = df["hour_diff"].clip(lower=0)
    df["hours_since_first_tx"] = df.groupby("app_id")["hour_diff"].cumsum()
    return df


def ensure_raw_sources(train_path: Path, test_path: Path) -> tuple[Path, Path]:
    if train_path.exists() and test_path.exists():
        return train_path, test_path

    raw_dir = train_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    for url, folder_name, target in (
        (
            "https://storage.yandexcloud.net/ds-ods/files/materials/6e991b7f/train_transactions_contest.zip",
            "train_transactions_contest",
            train_path,
        ),
        (
            "https://storage.yandexcloud.net/ds-ods/files/materials/fc0f1aa3/test_transactions_contest.zip",
            "test_transactions_contest",
            test_path,
        ),
    ):
        zip_path = raw_dir / f"{target.name}.zip"
        subprocess.run(["curl", "-L", "-o", str(zip_path), url], check=True)
        subprocess.run(
            ["unzip", "-o", str(zip_path), "-d", str(raw_dir), "-x", "__MACOSX/*"],
            check=True,
        )
        extracted = raw_dir / folder_name
        if not extracted.exists():
            raise FileNotFoundError(f"Expected extracted file was not found: {extracted}")
        extracted.rename(target)
        zip_path.unlink(missing_ok=True)

    return raw_dir / "train_transactions", raw_dir / "test_transactions"


def preprocess_full_alphabattle(
    train_path,
    test_path,
    clients_number: int,
    output_path: Path,
) -> None:
    payment_drop: list | None = None
    mcc_drop: list | None = None
    card_drop: list | None = None

    clients_number = [
        clients_number,
        int(clients_number * (0.35 if clients_number != -1 else 1)),
    ]
    print(clients_number)
    for p, cn, pth in zip(["train", "test"], clients_number, (train_path, test_path)):
        print("reading data from", pth)
        df = pd.read_parquet(pth)
        print(pth, cn)
        print("Original size:", df.shape)
        df = feature_drop(df, "currency", remain_cats=[1], user_col="app_id")
        if payment_drop is None:
            payment_drop = df["payment_system"].value_counts().index[-1:].tolist()
        df = feature_drop(
            df,
            "payment_system",
            drop_cats=payment_drop,
            user_col="app_id",
        )
        if mcc_drop is None:
            mcc_drop = df["mcc_category"].value_counts().index[-1:].tolist()
        df = feature_drop(
            df,
            "mcc_category",
            drop_cats=mcc_drop,
            user_col="app_id",
        )
        if card_drop is None:
            card_drop = df["card_type"].value_counts().index[-20:].tolist()
        df = feature_drop(
            df,
            "card_type",
            drop_cats=card_drop,
            user_col="app_id",
        )
        df = extract_time_feature(df)
        df = filter_user_seq_diff_bounds(df)
        # df = filter_by_trx_in_month(df, 2, 300)
        if cn != -1:
            df = select_clients(df, METADATA["index_columns"][0], cn)
        print("Filtered size:", df.shape)
        df.to_parquet(output_path / f"temp_{p}.parquet")


def main(
    train_path,
    test_path,
    dataset_name,
    clients_number=100_000,
):
    train_path = Path(train_path)
    test_path = Path(test_path)
    train_path, test_path = ensure_raw_sources(train_path, test_path)

    output_path = Path(f"data/{dataset_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info("Start data preprocessing")

    preprocess_full_alphabattle(train_path, test_path, clients_number, output_path)
    logging.info("Dataset was preprocessed")

    spark = SparkSession.builder.master("local[32]").getOrCreate()

    train_data = spark.read.parquet(str(output_path / "temp_train.parquet"))
    test_data = spark.read.parquet(str(output_path / "temp_test.parquet"))

    encode_data(train_data, test_data, path=output_path)
    logging.info("Encoding was saved.")

    train_data = code_indexes(
        train_data, METADATA["index_columns"], load_path=output_path / "idx"
    )
    logging.info("Indexes was created")
    train_data = code_categories(
        train_data, METADATA["cat_features"], load_path=output_path / "cat_codes"
    )
    logging.info("Codes was created")

    test_data = code_indexes(
        test_data, METADATA["index_columns"], load_path=output_path / "idx"
    )
    logging.info("Indexes was created")
    test_data = code_categories(
        test_data, METADATA["cat_features"], load_path=output_path / "cat_codes"
    )
    logging.info("Codes was created")

    logging.info("Saving to parquet...")
    train_data = add_row_order(train_data)
    test_data = add_row_order(test_data)
    save_to_parquet(
        train_data,
        save_path=output_path / "train",
        cat_codes_path=output_path / "cat_codes",
        idx_codes_path=output_path / "idx",
        metadata=METADATA,
        overwrite=True,
    )
    logging.info("Train was saved.")

    save_to_parquet(
        test_data,
        save_path=output_path / "test",
        cat_codes_path=output_path / "cat_codes",
        idx_codes_path=output_path / "idx",
        metadata=METADATA,
        overwrite=True,
    )
    logging.info("Test was saved.")
    logging.info("Dataset was successfully preprocessed.")
    spark.stop()
    logging.info("Spark was successfully stopped.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess and convert AlphaBattle dataset to Parquet"
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default="data/alphabattle/train_transactions",
        help="Path to raw train data",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/alphabattle/test_transactions",
        help="Path to raw test data",
    )
    parser.add_argument(
        "--dataname",
        type=str,
        default="alphabattle",
        help="Dataset name",
    )
    parser.add_argument("--nc", type=int, default=100_000, help="Number of clients")
    args = parser.parse_args()

    main(
        train_path=args.train_path,
        test_path=args.test_path,
        dataset_name=args.dataname,
        clients_number=args.nc,
    )
