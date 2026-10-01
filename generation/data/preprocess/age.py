from argparse import ArgumentParser
from pathlib import Path
import subprocess

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, FloatType

from common import add_row_order, cat_freq, collect_lists, train_test_split


CAT_FEATURES = ["small_group", "age"]
NUM_FEATURES = ["amount_rur"]
INDEX_COLUMNS = ["client_id", "bins"]
ORDERING_COLUMNS = ["trans_date"]
TARGET_VALS = [0, 1, 2, 3]
TEST_FRACTION = 0.2


def ensure_raw_sources(data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)

    required_files = (
        "transactions_train.csv",
        "transactions_test.csv",
        "train_target.csv",
    )
    if all((data_path / name).is_file() for name in required_files):
        return

    gz_files = {
        "transactions_train.csv.gz": "https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz?download=true",
        "transactions_test.csv.gz": "https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_test.csv.gz?download=true",
    }
    for file_name, url in gz_files.items():
        subprocess.run(
            ["curl", "-L", "-o", str(data_path / file_name), url],
            check=True,
        )

    subprocess.run(["curl", "-L", "-o", str(data_path / "train_target.csv"),
                    "https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/train_target.csv?download=true"], check=True)

    for gz_name in gz_files:
        subprocess.run(["gunzip", "-f", str(data_path / gz_name)], check=True)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to directory containing CSV files",
        default="data/age",
        type=Path,
    )
    parser.add_argument(
        "--save-path",
        help="Where to save preprocessed parquets",
        default="data/age/preprocessed",
        type=Path,
    )
    parser.add_argument(
        "--split-seed",
        help="Random seed used to split the data on train and test",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"
    ensure_raw_sources(args.data_path)

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore
    df, df_kag_train = None, None

    df_kag_train = spark.read.csv(
        (args.data_path / "transactions_train.csv").as_posix(), header=True
    )
    df_kag_train = add_row_order(
        df_kag_train.select(
            F.col("client_id").cast(LongType()),
            F.col("trans_date").cast(LongType()),
            F.col("small_group").cast(LongType()),
            F.col("amount_rur").cast(FloatType()),
        )
    )

    df_label = spark.read.csv(
        (args.data_path / "train_target.csv").as_posix(), header=True
    ).select(F.col("client_id").cast(LongType()), F.col("bins").cast(LongType()))

    df_kag_train = df_kag_train.join(df_label, on="client_id")
    df_kag_train = df_kag_train.withColumn("age", F.col("bins"))


    df = df_kag_train

    vcs = cat_freq(df, CAT_FEATURES)
    for vc in vcs:
        df = vc.encode(df)
        vc.write(args.save_path / "cat_codes" / vc.feature_name, mode=mode)

    df = collect_lists(
        df,
        group_by=INDEX_COLUMNS,
        order_by=ORDERING_COLUMNS,
    )

    stratify_col = "bins"
    stratify_col_vals = TARGET_VALS

    # stratified splitting on train and test
    train_df, test_df = train_test_split(
        df=df,
        test_frac=TEST_FRACTION,
        index_col="client_id",
        stratify_col=stratify_col,
        stratify_col_vals=stratify_col_vals,
        random_seed=args.split_seed,
    )

    train_df.repartition(20).write.parquet((args.save_path / "train").as_posix(), mode=mode)
    test_df.repartition(3).write.parquet((args.save_path / "test").as_posix(), mode=mode)


if __name__ == "__main__":
    main()
