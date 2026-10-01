import datetime
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, LongType

from common import add_row_order, cat_freq, collect_lists, train_test_split

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/gender")
TRX_CSV = "transactions.csv"
COL_CLIENT_ID = "customer_id"
COL_EVENT_TIME = "tr_datetime"
COLS_CATEGORY = ["mcc_code", "tr_type"]
AMOUNT_COL = "amount"

SEQ_LEN_MIN = 20
SEQ_LEN_LT = 1500
DIFF_MEDIAN_MAX = 10.0
DIFF_MAX_MAX = 100.0

SAVE_PATH = Path("data/gender/preprocessed")
TEST_FRACTION = 0.2
SPLIT_SEED = 0
TRAIN_REPARTITION = 20
TEST_REPARTITION = 3


def ensure_raw_sources() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trx = DATA_DIR / TRX_CSV
    if not trx.is_file():
        gz = DATA_DIR / "transactions.csv.gz"
        subprocess.run(
            [
                "curl",
                "-L",
                "-o",
                str(gz),
                "https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main/transactions.csv.gz?download=true",
            ],
            check=True,
        )
        subprocess.run(["gunzip", "-f", str(gz)], check=True)


def with_event_time_pd(df: pd.DataFrame) -> pd.DataFrame:
    s = df[COL_EVENT_TIME].astype(str).str.strip()
    padded = s.str.rjust(15, "0")
    day = pd.to_numeric(padded.str.slice(0, 6), errors="coerce").fillna(0.0)
    time_str = padded.str.slice(7, 15)
    time_str = time_str.str.replace(r":60$", ":59", regex=True)
    frac = pd.to_timedelta(time_str, errors="coerce").dt.total_seconds() / (24.0 * 60 * 60)
    out = df.assign(
        event_time=day.astype("float32") + frac.fillna(0.0).astype("float32")
    )
    return out.drop(columns=[COL_EVENT_TIME])


def filter_clients_pd(df: pd.DataFrame) -> pd.DataFrame:
    u = COL_CLIENT_ID
    df = df.sort_values([u, "event_time"])
    df["_dt"] = df.groupby(u, sort=False)["event_time"].transform(
        lambda x: pd.Series(np.diff(x.to_numpy(), prepend=x.iloc[0]), index=x.index)
    )
    agg = df.groupby(u, sort=False).agg(
        n=("event_time", "size"),
        mx_dt=("_dt", "max"),
        md_dt=("_dt", "median"),
    )
    mask = (
        (agg["n"] >= SEQ_LEN_MIN)
        & (agg["n"] < SEQ_LEN_LT)
        & (agg["mx_dt"] <= DIFF_MAX_MAX)
        & (agg["md_dt"] <= DIFF_MEDIAN_MAX)
    )
    keep = agg.index[mask]
    return df.loc[df[u].isin(keep), [c for c in df.columns if c != "_dt"]]


def cast_types_pd(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COL_CLIENT_ID] = pd.to_numeric(out[COL_CLIENT_ID], errors="coerce").astype("int64")
    out["event_time"] = out["event_time"].astype("float32")
    for c in COLS_CATEGORY:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
    out[AMOUNT_COL] = pd.to_numeric(out[AMOUNT_COL], errors="coerce").astype("float32")
    return out


def logging_config() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(funcName)-20s   : %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    _start = datetime.datetime.now()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    logging_config()

    ensure_raw_sources()

    path = DATA_DIR / TRX_CSV
    pdf = pd.read_csv(path)
    logger.info("Loaded %s rows from %s", len(pdf), path)

    pdf = with_event_time_pd(pdf)
    pdf = cast_types_pd(pdf)
    pdf = filter_clients_pd(pdf)

    spark = SparkSession.builder.master("local[32]").getOrCreate()
    df = add_row_order(spark.createDataFrame(pdf))

    vcs = cat_freq(df, COLS_CATEGORY)
    for vc in vcs:
        df = vc.encode(df)
        vc.write(SAVE_PATH / "cat_codes" / vc.feature_name, mode="overwrite")

    # Same casts as save_to_parquet (AlphaBattle) before collect_lists; index stays long here.
    df = df.select(
        F.col(COL_CLIENT_ID).cast(LongType()),
        *[F.col(c).cast(LongType()) for c in COLS_CATEGORY],
        F.col("event_time").cast(FloatType()),
        F.col(AMOUNT_COL).cast(FloatType()),
    )

    df = collect_lists(df, group_by=COL_CLIENT_ID, order_by="event_time")
    df.persist()
    _ = df.count()

    train_df, test_df = train_test_split(
        df=df,
        test_frac=TEST_FRACTION,
        index_col=COL_CLIENT_ID,
        stratify_col=None,
        random_seed=SPLIT_SEED,
    )

    train_df.repartition(TRAIN_REPARTITION).write.parquet(
        (SAVE_PATH / "train").as_posix(),
        mode="overwrite",
    )
    test_df.repartition(TEST_REPARTITION).write.parquet(
        (SAVE_PATH / "test").as_posix(),
        mode="overwrite",
    )
    logger.info('Saved train to "%s", test to "%s"', SAVE_PATH / "train", SAVE_PATH / "test")

    spark.stop()

    _duration = datetime.datetime.now() - _start
    logger.info("Data collected in %s sec (%s)", _duration.seconds, _duration)


if __name__ == "__main__":
    main()
