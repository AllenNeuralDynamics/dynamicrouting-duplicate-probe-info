from __future__ import annotations

import concurrent.futures
import logging
import json
import pathlib
from typing import Iterable

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
                session_id=str(session.id),
                path=zarr_path.as_posix(),
                probe=str(probe),
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
        to_modify.extend([p.as_posix() for p in recording.glob('*.oebin')])
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

def remove_devices_from_oebin(contents: dict, device_names: Iterable[str]) -> dict:
    if not device_names:
        raise ValueError('device_names must not be empty')
    any_deleted = False
    for name in device_names:
        assert name.endswith('AP') or name.endswith('LFP')
        assert name.split('-')[0][0] in 'ABCDEF'
        for subdir_name in ('events', 'continuous'):    
            # iterate over copy of list so as to not disrupt iteration when elements are removed
            for device_info in (d for d in contents[subdir_name]):
                stream_name = device_info['stream_name']
                if stream.endswith(name):
                    logger.debug(f'Removing {subdir_name}/{stream_name} from structure.oebin')
                    contents[subdir_name].remove(device_info)
                    any_deleted = True
    assert any_deleted, "No devices removed from oebin contents"
    return contents


def remove_devices_from_oebin(contents: dict, device_names: Iterable[str]) -> dict:
    if not device_names:
        raise ValueError('device_names must not be empty')
    any_deleted = False
    for name in device_names:
        assert name.endswith('AP') or name.endswith('LFP')
        assert name.split('-')[0][0] in 'ABCDEF'
        for subdir_name in ('events', 'continuous'):    
            # iterate over copy of list so as to not disrupt iteration when elements are removed
            for device_info in (d for d in contents[subdir_name]):
                stream = device_info['stream_name']
                if stream.endswith(name):
                    logger.debug(f'Removing {subdir_name}/{stream} from structure.oebin')
                    contents[subdir_name].remove(device_info)
                    any_deleted = True
    assert any_deleted, "No devices removed from oebin contents"
    return contents

def remove_missing_from_oebin(oebin_path: str | upath.UPath) -> dict:
    oebin_path = upath.UPath(oebin_path)
    contents = json.loads(oebin_path.read_text())
    for subdir_name in ('events', 'continuous'):    
        for device_info in (d for d in contents[subdir_name]):
            # iterate over copy of list so as to not disrupt iteration when elements are removed
            
            stream_name = device_info['stream_name']
            folder_name = device_info['folder_name']
            if "MessageCenter" in folder_name:
                continue
            clipped_dir = oebin_path.parent / subdir_name / folder_name
            logger.debug(f"Checking {clipped_dir} exists and is not empty")
            if not clipped_dir.exists():
                logger.info(f'Removing {subdir_name}/{stream_name} from structure.oebin as {clipped_dir} does not exist on S3')
                contents[subdir_name].remove(device_info)
            elif not (paths := next(clipped_dir.rglob('*'), None) or tuple(clipped_dir.iterdir())):
                logger.info(f'Removing {subdir_name}/{stream_name} from structure.oebin as {clipped_dir} is empty on S3')
                contents[subdir_name].remove(device_info)
            else:
                logger.debug(f"Found at least {len(paths)} paths in {clipped_dir} - leaving info in oebin file")
            
            if subdir_name == 'events':
                continue # duplicate device info in events doesn't cause a problem at this stage, but may be removed at a later stage 
            experiment = oebin_path.parent.parent.name
            node = oebin_path.parent.parent.parent.name
            # currently-used version of upath (pre v2?) cannot handle paths with '#'
            compressed_dir = oebin_path.parent.parent.parent.parent.parent / "ecephys_compressed"
            zarr_name = f"{experiment}_{node}#{folder_name.replace('/', '').replace('TTL', '')}.zarr"
            _zarr_path = f"{compressed_dir}/{zarr_name}"
            logger.debug(f"Checking {_zarr_path} exists")
            if not next(compressed_dir.glob(zarr_name), None):
                logger.warning(f'Removing {subdir_name}/{stream_name} from structure.oebin as {_zarr_path} does not exist: paths available {[p.name for p in compressed_dir.iterdir()]}')
    return contents

if __name__ == "__main__":
    test = False

    if test:
        logger.warning(f"Test mode: reading .parquet from /code")
        df = pl.read_parquet('/code/duplicate_probes.parquet')
    else:
        df = (
            get_probe_info_all_sessions()
            .pipe(get_duplicates_to_delete)
            .pipe(filter_dr_sessions)
            .pipe(add_clipped_paths_to_delete)
            .pipe(add_oebin_files_to_modify)
            .pipe(simplify)
        )

    for row in df.iter_rows(named=True):
        orig_path = row['to_modify']
        original = json.loads(upath.UPath(orig_path).read_text())
        updated = remove_missing_from_oebin(orig_path)
        new = remove_devices_from_oebin(updated, row['names'])
        new_path = upath.UPath(orig_path.replace('s3:/', '/results/to_replace'))
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(json.dumps(new, indent=4))
    df.rename({'to_modify': 'to_replace'})
    df.write_parquet('/results/to_delete.parquet')
    df.pipe(cast_lists_to_strings).write_csv('/results/to_delete.csv')
