import concurrent.futures
import logging

import aind_session
import npc_session
import polars as pl
import requests
import tqdm
import upath

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)s |  %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

tracked = upath.UPath(
        "https://raw.githubusercontent.com/AllenInstitute/npc_lims/refs/heads/main/tracked_sessions.yaml"
    ).read_text()

def get_info(session_id: str) -> list[dict[str, str]]:
    """Read the session's zarr ecephys data on S3 and find the probes included"""
    session = aind_session.Session(session_id)
    info = []
    try:
        compressed_dir = session.ecephys.compressed_dir
    except AttributeError:
        logger.debug(f"No compressed data found for {session_id}")
        return info
    for zarr_path in compressed_dir.iterdir():
        if "NI-DAQmx" in zarr_path.name:
            continue
        if "-LFP" in zarr_path.name:
            band = "LFP"
        elif "-AP" in zarr_path.name:
            band = "AP"
        else:
            logger.debug(f"Could not parse AP/LFP band from {zarr_path.name}: skipping")
            continue
        try:
            probe = npc_session.ProbeRecord(zarr_path.name)
        except ValueError:
            continue
        info.append(
            dict(
                session_id=session.id,
                path=zarr_path.as_posix(),
                probe=probe,
                band=band,
            )
        )
    return info


def get_probe_info_all_sessions() -> pl.DataFrame:
    """Get a df with zarr metadata for all probes for all ecephys sessions in docdb"""
    records = aind_session.get_docdb_api_client().retrieve_docdb_records(
        filter_query={
            "name": {"$regex": "^ecephys_"},
        },
        sort={"created": 1},
    )
    sessions = {aind_session.Session(record["name"]) for record in records}
    logger.debug(f"Got records for {len(sessions)} sessions from docdb")
    future_to_session = {}
    info = []
    session_ids = [s.id for s in sessions]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for session_id in session_ids:
            future = executor.submit(get_info, session_id)
            future_to_session[future] = session_id
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_session), total=len(session_ids)
        ):
            try:
                info.extend(future.result())
            except requests.HTTPError as exc:
                if exc.response.status_code == 401:
                    # asset doesn't exist or is inaccessible
                    logger.debug(f"{future_to_session[future]}: {exc!r}")
                    continue
                raise
            else:
                logger.debug(f"Got info for {len(info)} probes for {session_id}")
    return pl.DataFrame(info)


def get_duplicates_to_delete(df: pl.DataFrame) -> pl.DataFrame:
    x = (
        df
        .with_columns(
            pl.col('path').str.extract(r"/experiment(\d+)_").cast(int).alias('experiment'),
        )
        .drop_nulls('experiment')    
        .sort('path')
        .group_by(['session_id', 'probe', 'band', 'experiment'])
        .agg(
            pl.col('probe').len().alias('count'),
            pl.col('path').first().alias('first_path'),
            pl.col('path').last().alias('last_path'),
        )
        .filter(
            pl.col('count') > 1,
        )
        .with_columns(
            pl.when(
                pl.col('probe').is_in(['A', 'B', 'C'])
            ).then(
                pl.col('last_path').alias('to_delete')
            ).otherwise(
               pl.col('first_path').alias('to_delete')
            )
        )
        # .explode('node')
        .sort('session_id')
        .with_columns(
            pl.col('to_delete').str.extract(r"_Record Node (\d+)").cast(int).alias('node'),
        )
        .drop(['first_path', 'last_path'])
    )
    assert sum(x['count']) == 2 * len(x) # duplicate == probe on 2 record nodes
    return x.drop('count')

def filter_dr_sessions(df):
    return (
        df
        .with_columns(
            pl.col('session_id').str.split("_").list.slice(1,2).list.join("_").str.replace("-", "", literal=True, n=2).alias('dr_id')
        )
        
        .filter(
            pl.col("dr_id").map_elements(lambda x: x in tracked, return_dtype=bool),
            # pl.col("dr_id").is_in(tracked),
        )
        .drop('dr_id')
    )

def add_clipped_paths_to_delete(df: pl.DataFrame) -> pl.DataFrame:
    to_delete = []
    for row in df.iter_rows(named=True):
        _row_paths = []
        compressed = upath.UPath(row['to_delete'])
        # compressed contains "#" which messes up S3 path with upath 
        # - don't convert to/from upath, only use for getting parents
        _row_paths.append(row['to_delete'])
        session_dir = next(
            p for p in compressed.parents
            if p.name == row['session_id']
        )
        if (session_dir / 'ecephys').exists():
            clipped = session_dir / 'ecephys' / 'ecephys_clipped'
        else:
            clipped = session_dir / 'ecephys_clipped'
        experiment = clipped / f"Record Node {row['node']}/experiment{row['experiment']}"
        assert experiment.exists()
        probe_rec_name = next((experiment / "recording1/continuous").glob(f"*{row['probe']}-{row['band']}")).name
        
        # we only upload a single recording folder for DR
        if (experiment / "recording2" / "continuous" / probe_rec_name).exists():
            logger.warning(f"2 or more recording folders exist in {experiment}: data may need re-uploading")
        for d in ("continuous", "events"):
            p = experiment / "recording1" / d / probe_rec_name
            assert p.exists(), f"{p} does not exist"
            assert all(
                s in p.as_posix() for s in (
                    f"experiment{row['experiment']}",
                    f"Record Node {row['node']}",
                    f"{row['probe']}-{row['band']}",
                )
            ), f"Some expected components not found in {p}"
            _row_paths.append(p.as_posix())            
        to_delete.append(_row_paths)
    return df.drop('to_delete').hstack(
        [pl.Series('to_delete', to_delete)]
    )

def add_oebin_files_to_modify(df: pl.DataFrame) -> pl.DataFrame:
    to_modify = []
    for row in df.iter_rows(named=True):
        assert row['to_delete'], f"No files to delete: {row}"
        p = upath.UPath(next(p for p in row['to_delete'] if 'continuous' in p))
        recording = next(d for d in p.parents if d.name == 'continuous').parent
        paths = list(recording.glob('*.oebin'))
        assert len(paths) == 1, f"Expected one file, got: {paths}"
        to_modify.append(paths[0].as_posix())
    return df.hstack(
        [pl.Series('to_modify', to_modify)]
    ) 
           
def simplify(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_columns(
            pl.concat_str([pl.col('probe'), pl.col('band')], separator='-').alias('names'),
        )
        .group_by(['session_id', 'to_modify'])
        .agg(
            pl.col('to_delete').explode(),
            pl.col('names'),

        )
    )

def cast_lists_to_strings(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if df[col].dtype == pl.List:
            df = df.with_columns(
                pl.col(col).cast(pl.List(pl.String)).list.join(";").alias(col)
            )
    return df

if __name__ == "__main__":
    df = (
        get_probe_info_all_sessions()
        .pipe(get_duplicates_to_delete)
        .pipe(filter_dr_sessions)
        .pipe(add_clipped_paths_to_delete)
        .pipe(add_oebin_files_to_modify)
        .pipe(simplify)
    )
    df.write_parquet('/results/duplicate_probes.parquet')
    df.pipe(cast_lists_to_strings).write_csv('/results/duplicate_probes.csv')
