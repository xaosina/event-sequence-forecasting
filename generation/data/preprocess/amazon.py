import shutil
from argparse import ArgumentParser
from pathlib import Path

import pyspark.sql.functions as F
from datasets import concatenate_datasets, load_dataset
from pyspark.sql import SparkSession


HF_DATASET = "easytpp/amazon"
TRAIN_REPARTITION = 20
TEST_REPARTITION = 3
OUTPUT_COLS = ["seq_idx", "time_since_start", "_seq_len", "type_event"]


def parse_args():
    parser = ArgumentParser(description="Download EasyTPP Amazon and save to parquet.")
    parser.add_argument("--data-path", default="data/amazon", type=Path)
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


def write_parquet(dataset, out_dir: Path, staging: Path, n_partitions: int, mode: str, spark):
    dataset.to_parquet(staging.as_posix())
    cast_output_dtypes(spark.read.parquet(staging.as_posix())).sort("seq_idx").select(
        *OUTPUT_COLS
    ).repartition(n_partitions).write.parquet(out_dir.as_posix(), mode=mode)
    staging.unlink()


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
        write_parquet(splits[name], out_dir, cache_dir / f"{name}.parquet", n_parts, mode, spark)
        print(f"Dump {name}: {len(splits[name])} sequences -> {out_dir}")

    shutil.rmtree(cache_dir)
    print("OK")


if __name__ == "__main__":
    main()
