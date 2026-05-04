import importlib.abc
import io
import os
import shlex
import sys
import types
from unittest import mock

import pytest

from stable_audio_tools.data import s3_utils as s3


@pytest.fixture(autouse=True)
def _clear_s3_client_cache():
    "_build_s3_client caches clients via lru_cache; clear it so tests don't share state."
    s3._build_s3_client.cache_clear()
    yield
    s3._build_s3_client.cache_clear()


class _FakeConfig:
    "Minimal stand-in for botocore.config.Config that records its kwargs."

    def __init__(self, **kwargs):
        self.user_agent_extra = kwargs.get("user_agent_extra")
        self.kwargs = kwargs


def _patch_boto3(fake_boto3):
    "Patch boto3 + botocore.config so _get_s3_client runs without them installed."
    botocore = types.ModuleType("botocore")
    botocore.__path__ = []  # mark as a package so submodule import resolves
    botocore_config = types.ModuleType("botocore.config")
    botocore_config.Config = _FakeConfig
    botocore.config = botocore_config
    return mock.patch.dict(
        "sys.modules",
        {
            "boto3": fake_boto3,
            "botocore": botocore,
            "botocore.config": botocore_config,
        },
    )


def _fake_session():
    fake_boto3 = mock.MagicMock()
    fake_session = mock.MagicMock()
    fake_boto3.Session.return_value = fake_session
    return fake_boto3, fake_session


def _fake_paginator(pages):
    "Paginator-like mock; records last paginate(**kwargs) on .last_kwargs."
    pag = mock.MagicMock()
    pag.last_kwargs = {}

    def paginate(**kwargs):
        pag.last_kwargs = kwargs
        return iter(pages)

    pag.paginate.side_effect = paginate
    return pag


def _fake_client(pages=None):
    client = mock.MagicMock()
    client.get_paginator.return_value = _fake_paginator(pages or [])
    return client


# ---- client construction -------------------------------------------------

def test_get_s3_client_uses_aws_endpoint_url_env():
    fake_boto3, fake_session = _fake_session()

    with mock.patch.dict(os.environ, {"AWS_ENDPOINT_URL": "https://s3.us-west-004.backblazeb2.com"}, clear=False):
        with _patch_boto3(fake_boto3):
            s3._get_s3_client()

    fake_session.client.assert_called_once()
    args, kwargs = fake_session.client.call_args
    assert args == ("s3",)
    assert kwargs["endpoint_url"] == "https://s3.us-west-004.backblazeb2.com"
    assert kwargs["config"].user_agent_extra.startswith("stable-audio-tools/")


def test_get_s3_client_default_when_env_unset():
    fake_boto3, fake_session = _fake_session()

    env = {k: v for k, v in os.environ.items() if k != "AWS_ENDPOINT_URL"}
    with mock.patch.dict(os.environ, env, clear=True):
        with _patch_boto3(fake_boto3):
            s3._get_s3_client()

    fake_session.client.assert_called_once()
    args, kwargs = fake_session.client.call_args
    assert args == ("s3",)
    assert kwargs["endpoint_url"] is None
    assert kwargs["config"].user_agent_extra.startswith("stable-audio-tools/")


def test_get_s3_client_uses_profile_when_given():
    fake_boto3, fake_session = _fake_session()

    with _patch_boto3(fake_boto3):
        s3._get_s3_client(profile="myprofile")

    fake_boto3.Session.assert_called_once_with(profile_name="myprofile")


def test_s3_client_configured_for_sigv4():
    fake_boto3, fake_session = _fake_session()

    with _patch_boto3(fake_boto3):
        s3._get_s3_client()

    _, kwargs = fake_session.client.call_args
    # B2 only supports SigV4; the client must be configured for it.
    assert kwargs["config"].kwargs.get("signature_version") == "s3v4"


def test_get_s3_client_presigns_with_sigv4():
    # Real boto3: the client our code builds must sign with SigV4 (B2 rejects v2).
    pytest.importorskip("boto3")
    with mock.patch.dict(os.environ, {
        "AWS_ENDPOINT_URL": "https://s3.us-west-004.backblazeb2.com",
        "AWS_ACCESS_KEY_ID": "test-key-id",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "AWS_DEFAULT_REGION": "us-west-004",
    }, clear=False):
        client = s3._get_s3_client()
        url = client.generate_presigned_url(
            "get_object", Params={"Bucket": "b", "Key": "k"}, ExpiresIn=3600)

    assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in url
    assert "AWSAccessKeyId=" not in url  # SigV2 marker must be absent


def test_get_s3_client_missing_boto3_raises_actionable_error(monkeypatch):
    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "boto3" or name.startswith("botocore"):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    for mod in ("boto3", "botocore", "botocore.config"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])

    with pytest.raises(ImportError, match="boto3"):
        s3._get_s3_client()


# ---- user agent ----------------------------------------------------------

def test_user_agent_is_versioned_product_form():
    fake_boto3, fake_session = _fake_session()

    with mock.patch.dict(os.environ, {}, clear=True):
        with _patch_boto3(fake_boto3):
            s3._get_s3_client()

    _, kwargs = fake_session.client.call_args
    ua = kwargs["config"].user_agent_extra
    assert ua == s3._user_agent()
    assert ua.startswith("stable-audio-tools/")


def test_user_agent_appends_extra_argument():
    fake_boto3, fake_session = _fake_session()

    with mock.patch.dict(os.environ, {}, clear=True):
        with _patch_boto3(fake_boto3):
            s3._get_s3_client(user_agent_extra="myapp/1.0")

    _, kwargs = fake_session.client.call_args
    assert kwargs["config"].user_agent_extra == f"{s3._user_agent()} myapp/1.0"


def test_user_agent_empty_extra_suppresses_env():
    fake_boto3, fake_session = _fake_session()

    with mock.patch.dict(os.environ, {"STABLE_AUDIO_TOOLS_USER_AGENT_EXTRA": "fromenv/2"}, clear=True):
        with _patch_boto3(fake_boto3):
            s3._get_s3_client(user_agent_extra="")

    _, kwargs = fake_session.client.call_args
    assert kwargs["config"].user_agent_extra == s3._user_agent()


def test_user_agent_appends_extra_from_env():
    fake_boto3, fake_session = _fake_session()

    with mock.patch.dict(os.environ, {"STABLE_AUDIO_TOOLS_USER_AGENT_EXTRA": "fromenv/2"}, clear=True):
        with _patch_boto3(fake_boto3):
            s3._get_s3_client()

    _, kwargs = fake_session.client.call_args
    assert kwargs["config"].user_agent_extra == f"{s3._user_agent()} fromenv/2"


# ---- _parse_s3_url -------------------------------------------------------

def test_parse_s3_url():
    assert s3._parse_s3_url("s3://bucket/a/b.tar") == ("bucket", "a/b.tar")
    assert s3._parse_s3_url("s3://bucket") == ("bucket", "")
    with pytest.raises(ValueError):
        s3._parse_s3_url("https://not-s3/x")
    with pytest.raises(ValueError):
        s3._parse_s3_url("s3://")  # empty bucket


# ---- get_s3_contents -----------------------------------------------------

def test_get_s3_contents_returns_keys_relative_to_dataset_path():
    # Matches the legacy aws s3 ls output shape: keys relative to dataset_path.
    pages = [
        {"Contents": [
            {"Key": "prefix/a.tar"},
            {"Key": "prefix/sub/b.tar"},
            {"Key": "prefix/"},  # directory marker -> skipped
        ]},
    ]
    client = _fake_client(pages=pages)

    with mock.patch.object(s3, "_get_s3_client", return_value=client):
        keys = s3.get_s3_contents("s3://bucket/prefix/", recursive=True)

    client.get_paginator.assert_called_once_with("list_objects_v2")
    pag = client.get_paginator.return_value
    assert pag.last_kwargs == {"Bucket": "bucket", "Prefix": "prefix/"}
    assert keys == ["a.tar", "sub/b.tar"]


def test_get_s3_contents_non_recursive_adds_delimiter():
    client = _fake_client(pages=[{"Contents": []}])

    with mock.patch.object(s3, "_get_s3_client", return_value=client):
        s3.get_s3_contents("s3://bucket/prefix/", recursive=False)

    pag = client.get_paginator.return_value
    assert pag.last_kwargs == {"Bucket": "bucket", "Prefix": "prefix/", "Delimiter": "/"}


def test_get_s3_contents_non_recursive_strips_prefix_from_keys():
    pages = [{"Contents": [{"Key": "prefix/a.tar"}, {"Key": "prefix/b.tar"}]}]
    client = _fake_client(pages=pages)

    with mock.patch.object(s3, "_get_s3_client", return_value=client):
        keys = s3.get_s3_contents("s3://bucket/prefix/", recursive=False)

    assert keys == ["a.tar", "b.tar"]


def test_get_s3_contents_applies_filter():
    pages = [{"Contents": [
        {"Key": "prefix/a.tar"},
        {"Key": "prefix/b.txt"},
        {"Key": "prefix/c.tar"},
    ]}]
    client = _fake_client(pages=pages)

    with mock.patch.object(s3, "_get_s3_client", return_value=client):
        keys = s3.get_s3_contents("s3://bucket/prefix/", filter_str="tar", recursive=True)

    assert keys == ["a.tar", "c.tar"]


def test_get_s3_contents_rejects_non_s3_url():
    with pytest.raises(ValueError):
        s3.get_s3_contents("not-an-s3-url/")


# ---- shard_pipe_command + streaming -------------------------------------

def test_shard_pipe_command_builds_streaming_pipe():
    cmd = s3.shard_pipe_command("s3://bucket/key.tar")
    assert cmd.startswith(f"pipe:{shlex.quote(sys.executable)} -m stable_audio_tools.data.s3_utils ")
    assert "s3://bucket/key.tar" in cmd
    assert "--profile" not in cmd
    # No credentials or signatures are ever placed on the command line.
    assert "X-Amz-" not in cmd and "Signature" not in cmd


def test_shard_pipe_command_includes_profile():
    cmd = s3.shard_pipe_command("s3://bucket/key.tar", profile="myprof")
    assert f"--profile {shlex.quote('myprof')}" in cmd


def test_shard_pipe_command_rejects_non_s3():
    with pytest.raises(ValueError):
        s3.shard_pipe_command("https://nope/x")


def test_get_all_s3_urls_emits_streaming_pipe_command():
    pages = [{"Contents": [{"Key": "name/train/shard-000.tar"}]}]
    client = _fake_client(pages=pages)

    with mock.patch.object(s3, "_get_s3_client", return_value=client):
        urls = s3.get_all_s3_urls(
            names=["name"],
            subsets=["train"],
            s3_url_prefix="s3://bucket",
            recursive=True,
            filter_str="tar",
        )

    assert urls == [s3.shard_pipe_command("s3://bucket/name/train/shard-000.tar")]
    assert urls[0].startswith("pipe:")
    assert "s3://bucket/name/train/shard-000.tar" in urls[0]


def test_get_all_s3_urls_with_full_url_names_and_no_prefix():
    # s3_url_prefix=None: each name must already be a full s3:// URL.
    pages = [{"Contents": [{"Key": "prefix/name/train/shard-000.tar"}]}]
    client = _fake_client(pages=pages)

    with mock.patch.object(s3, "_get_s3_client", return_value=client):
        urls = s3.get_all_s3_urls(
            names=["s3://bucket/prefix/name"],
            subsets=["train"],
            s3_url_prefix=None,
            recursive=True,
            filter_str="tar",
        )

    assert urls == [s3.shard_pipe_command("s3://bucket/prefix/name/train/shard-000.tar")]


def test_get_all_s3_urls_passes_profile_into_command():
    pages = [{"Contents": [{"Key": "name/train/shard.tar"}]}]
    client = _fake_client(pages=pages)

    with mock.patch.object(s3, "_get_s3_client", return_value=client):
        urls = s3.get_all_s3_urls(
            names=["name"], subsets=["train"], s3_url_prefix="s3://bucket",
            profiles={"name": "myprof"},
        )

    assert f"--profile {shlex.quote('myprof')}" in urls[0]


def test_get_all_s3_urls_defaults_no_names_returns_empty():
    assert s3.get_all_s3_urls() == []


def test_stream_object_writes_object_bytes():
    body = mock.MagicMock()
    body.iter_chunks.return_value = [b"abc", b"def"]
    client = mock.MagicMock()
    client.get_object.return_value = {"Body": body}
    out = io.BytesIO()

    with mock.patch.object(s3, "_get_s3_client", return_value=client):
        s3.stream_object("s3://bucket/key", out=out)

    assert out.getvalue() == b"abcdef"
    client.get_object.assert_called_once_with(Bucket="bucket", Key="key")


def test_main_invokes_stream_object_with_profile():
    with mock.patch.object(s3, "stream_object") as m:
        rc = s3.main(["s3://bucket/key", "--profile", "myprof"])

    assert rc == 0
    m.assert_called_once_with("s3://bucket/key", profile="myprof")
