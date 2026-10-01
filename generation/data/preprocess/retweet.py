import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyspark.sql.functions as F
from datasets import concatenate_datasets, load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType


HF_DATASET = "easytpp/retweet"
TRAIN_REPARTITION = 20
TEST_REPARTITION = 3
OUTPUT_COLS = ["seq_idx", "time_since_start", "_seq_len", "type_event"]

MAX_DURATION = 2200
MIN_LENGTH = 10
MAX_LENGTH = 100


def parse_args():
    parser = ArgumentParser(description="Download EasyTPP Retweet and save to parquet.")
    parser.add_argument("--data-path", default="data/retweet", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_splits(cache_dir: Path):
    collection = load_dataset(HF_DATASET, cache_dir=str(cache_dir))
    return {
        "train": concatenate_datasets([collection["train"], collection["validation"]]),
        "test": collection["test"],
    }


def cast_output_dtypes(df):
    return (
        df.withColumn("seq_idx", F.col("seq_idx").cast("long"))
        .withColumn(
            "time_since_start",
            F.transform("time_since_start", lambda x: x.cast("float")),
        )
        .withColumn(
            "type_event",
            F.transform("type_event", lambda x: x.cast("long")),
        )
        .withColumn("_seq_len", F.size("time_since_start"))
    )


def apply_filters(df):
    length_udf = F.udf(
        lambda x: int(np.searchsorted(x, x[0] + MAX_DURATION)),
        LongType(),
    )
    df = df.withColumn("_length", length_udf("time_since_start"))
    # df = df.withColumn(
    #     "_length",
    #     F.when(F.col("_length") > MAX_LENGTH, MAX_LENGTH).otherwise(F.col("_length")),
    # )
    df = df.filter(F.col("_length") >= MIN_LENGTH)
    return (
        df.withColumn("time_since_start", F.slice(F.col("time_since_start"), 1, F.col("_length")))
        .withColumn("type_event", F.slice(F.col("type_event"), 1, F.col("_length")))
        .drop("_length")
    )


def write_parquet(dataset, out_dir: Path, staging: Path, n_partitions: int, mode: str, spark):
    dataset.to_parquet(staging.as_posix())
    df = cast_output_dtypes(apply_filters(spark.read.parquet(staging.as_posix())))
    n_rows = df.count()
    df.sort("seq_idx").select(*OUTPUT_COLS).repartition(n_partitions).write.parquet(
        out_dir.as_posix(), mode=mode
    )
    staging.unlink()
    return n_rows


def main():
    args = parse_args()
    args.data_path.mkdir(parents=True, exist_ok=True)
    mode = "overwrite" if args.overwrite else "error"

    cache_dir = args.data_path / "cache"
    splits = prepare_splits(cache_dir)
    layout = [("train", TRAIN_REPARTITION), ("test", TEST_REPARTITION)]

    spark = SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore
    spark.sparkContext.setLogLevel("WARN")

    for name, n_parts in layout:
        out_dir = args.data_path / name
        n_rows = write_parquet(
            splits[name], out_dir, cache_dir / f"{name}.parquet", n_parts, mode, spark
        )
        print(f"Dump {name}: {n_rows} sequences (from {len(splits[name])}) -> {out_dir}")

    shutil.rmtree(cache_dir)
    print("OK")


if __name__ == "__main__":
    main()
