"""S3 / S3-compatible (e.g. Backblaze B2) helpers for the WebDataset loaders.

Deliberately free of heavy imports (torch, webdataset). This module doubles as a
``pipe:`` subprocess entry point (``python -m stable_audio_tools.data.s3_utils
<s3-url>``) that streams a single shard via boto3, so spawning it per shard open
stays cheap and shard downloads:

* authenticate on every open (no presigned-URL expiry mid-run), and
* never place credentials on the command line (only the ``s3://`` path).

S3 code originally based on the implementation by Scott Hawley in
https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py
"""
import os
import posixpath
import shlex
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


@lru_cache(maxsize=1)
def _user_agent():
    "Base ``stable-audio-tools/<version>`` token, looked up once on first use."
    try:
        ver = version("stable-audio-tools")
    except PackageNotFoundError:  # source/editable checkout without dist metadata
        ver = "dev"
    return f"stable-audio-tools/{ver}"


def _build_user_agent_extra(user_agent_extra=None):
    """``stable-audio-tools/<version>`` with any caller- or env-provided
    (``STABLE_AUDIO_TOOLS_USER_AGENT_EXTRA``) value appended, not replacing it.
    Pass ``user_agent_extra=""`` to suppress the env value and use the base only."""
    base = _user_agent()
    if user_agent_extra is None:
        user_agent_extra = os.environ.get("STABLE_AUDIO_TOOLS_USER_AGENT_EXTRA")
    return f"{base} {user_agent_extra}" if user_agent_extra else base


@lru_cache(maxsize=32)
def _build_s3_client(profile, endpoint_url, user_agent_extra):
    try:
        import boto3  # local import so boto3 is only required when S3 is used
        from botocore.config import Config
    except ModuleNotFoundError as e:
        raise ImportError(
            "S3 dataset access requires boto3. Install it with "
            "'pip install boto3' or 'pip install stable-audio-tools[train]'."
        ) from e

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        # Backblaze B2's S3 API only accepts SigV4; force it. retries restore the
        # transient-failure resilience the previous `aws s3` CLI path provided.
        config=Config(
            signature_version="s3v4",
            user_agent_extra=user_agent_extra,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def _get_s3_client(profile=None, user_agent_extra=None):
    """
    Build (and reuse) a boto3 S3 client. Honors AWS_ENDPOINT_URL when set so the
    same code path works against any S3-compatible endpoint (AWS S3 by default;
    set AWS_ENDPOINT_URL to a Backblaze B2 endpoint to point it at B2). When the
    env var is unset, behavior matches the default AWS client.

    Clients are cached per (profile, endpoint, user-agent) so listing and
    streaming share one client instead of building a new one on each call.
    """
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL") or None
    return _build_s3_client(
        profile, endpoint_url, _build_user_agent_extra(user_agent_extra)
    )


def _parse_s3_url(url):
    "Split an ``s3://bucket/key`` URL into (bucket, key). Raises ValueError otherwise."
    if not url.startswith("s3://"):
        raise ValueError(f"expected an s3:// URL, got: {url!r}")
    bucket, _, key = url[len("s3://"):].partition("/")
    if not bucket.strip():
        raise ValueError(f"s3:// URL is missing a bucket name: {url!r}")
    return bucket, key


def get_s3_contents(
    dataset_path,
    s3_url_prefix=None,
    filter_str='',       # only keep keys containing this substring
    recursive=True,
    debug=False,
    profile=None,
):
    """
    Returns a list of S3 object keys relative to ``dataset_path``, matching the
    output shape of the previous ``aws s3 ls`` based implementation. Uses boto3
    directly so it works against any S3-compatible endpoint when
    ``AWS_ENDPOINT_URL`` is set.
    """
    # Ensure dataset_path ends with a trailing slash
    if dataset_path != '' and not dataset_path.endswith('/'):
        dataset_path += '/'
    # Use posixpath to construct the S3 URL path (e.g. "s3://bucket/prefix/")
    bucket_path = posixpath.join(s3_url_prefix or '', dataset_path)

    bucket, prefix = _parse_s3_url(bucket_path)

    s3 = _get_s3_client(profile=profile)
    paginator = s3.get_paginator("list_objects_v2")
    list_kwargs = {"Bucket": bucket, "Prefix": prefix}
    if not recursive:
        list_kwargs["Delimiter"] = "/"

    keys = []
    for page in paginator.paginate(**list_kwargs):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key", "")
            if not key or key.endswith("/"):
                continue
            keys.append(key)

    # Apply the filter, if specified
    if filter_str:
        keys = [k for k in keys if filter_str in k]

    # Strip the listed prefix so keys are relative to dataset_path (matches the
    # previous implementation's output shape).
    if prefix:
        keys = [k[len(prefix):] if k.startswith(prefix) else k for k in keys]
        keys = [k.lstrip('/') for k in keys]

    if debug:
        print("contents = \n", keys)

    return keys


def shard_pipe_command(url, profile=None):
    """Build the WebDataset ``pipe:`` command that streams ``url`` via boto3.

    Only the ``s3://`` path (never credentials) lands on the command line, and the
    object is fetched with fresh credentials on each open, so shards never expire.
    """
    _parse_s3_url(url)  # validate early
    cmd = f"{shlex.quote(sys.executable)} -m stable_audio_tools.data.s3_utils {shlex.quote(url)}"
    if profile:
        cmd += f" --profile {shlex.quote(profile)}"
    return f"pipe:{cmd}"


def get_all_s3_urls(
    names=None,         # list of [LAION AudioDataset] dataset names; None -> []
    subsets=None,       # list of subsets, e.g. ['train','valid']; None -> ['']
    s3_url_prefix=None,  # prefix for those dataset names
    recursive=True,     # recursively list all tar files in all subdirs
    filter_str='tar',   # only grab files with this substring
    debug=False,        # print debugging info
    profiles=None,      # dict of profiles per name, e.g. {'dataset1': 'profile1'}; None -> {}
):
    """Get WebDataset ``pipe:`` commands that stream shards (tar files) for
    multiple datasets in one S3 bucket.

    Each command streams its shard with boto3 at open time (see
    ``shard_pipe_command`` / ``stream_object``), so shards re-authenticate on
    every open, never expire mid-run, and never expose credentials on the
    command line.
    """
    names = [] if names is None else names
    subsets = [''] if subsets is None else subsets
    profiles = profiles or {}
    urls = []
    for name in names:
        # If s3_url_prefix is not specified, assume the full S3 path is included in each element of the names list
        if s3_url_prefix is None:
            contents_str = name
        else:
            contents_str = posixpath.join(s3_url_prefix, name)
        if debug:
            print(f"get_all_s3_urls: {contents_str}:")
        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print(f"subset_str = {subset_str}")
            profile = profiles.get(name, None)
            tar_list = get_s3_contents(
                subset_str, s3_url_prefix=None, recursive=recursive,
                filter_str=filter_str, debug=debug, profile=profile)
            for tar in tar_list:
                if s3_url_prefix is None:
                    full_s3_url = posixpath.join(name, subset, tar)
                else:
                    full_s3_url = posixpath.join(s3_url_prefix, name, subset, tar)
                request_str = shard_pipe_command(full_s3_url, profile=profile)
                if debug:
                    print("request_str =", request_str)
                urls.append(request_str)
    return urls


def stream_object(url, profile=None, out=None):
    """Stream the bytes of an S3 object to ``out`` (default: stdout's binary buffer)."""
    bucket, key = _parse_s3_url(url)
    client = _get_s3_client(profile=profile)
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    # Binary destination required; sys.stdout.buffer is the binary handle of a
    # real process stdout (the pipe: use case). Tests pass an explicit `out`.
    out = out if out is not None else sys.stdout.buffer
    try:
        for chunk in body.iter_chunks(chunk_size=1024 * 1024):
            out.write(chunk)
    finally:
        body.close()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="stable_audio_tools.data.s3_utils",
        description="Stream an S3 object to stdout (used as a WebDataset pipe: source).",
    )
    parser.add_argument("url", help="s3://bucket/key to stream")
    parser.add_argument("--profile", default=None, help="optional AWS/B2 profile name")
    args = parser.parse_args(argv)
    stream_object(args.url, profile=args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
