# Datasets
`stable-audio-tools` supports loading data from local file storage, as well as loading audio files and JSON files in the [WebDataset](https://github.com/webdataset/webdataset/tree/main/webdataset) format from Amazon S3 buckets.

# Dataset configs
To specify the dataset used for training, you must provide a dataset config JSON file to `train.py`.

The dataset config consists of a `dataset_type` property specifying the type of data loader to use, a `datasets` array to provide multiple data sources, and a `random_crop` property, which decides if the cropped audio from the training samples is from a random place in the audio file, or always from the beginning.

## Local audio files
To use a local directory of audio samples, set the `dataset_type` property in your dataset config to `"audio_dir"`, and provide a list of objects to the `datasets` property including the `path` property, which should be the path to your directory of audio samples.

This will load all of the compatible audio files from the provided directory and all subdirectories.

### Example config 
```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "my_audio",
            "path": "/path/to/audio/dataset/"
        }
    ],
    "random_crop": true
}
```

## S3 WebDataset
To load audio files and related metadata from .tar files in the WebDataset format hosted in Amazon S3 buckets, you can set the `dataset_type` property to `s3`, and provide the `datasets` parameter with a list of objects containing the AWS S3 path to the shared S3 bucket prefix of the WebDataset .tar files. The S3 bucket will be searched recursively given the path, and assumes any .tar files found contain audio files and corresponding JSON files where the related files differ only in file extension (e.g. "000001.flac", "000001.json", "00002.flac", "00002.json", etc.)

### Example config
```json
{
    "dataset_type": "s3",
    "datasets": [
        {
            "id": "s3-test",
            "s3_path": "s3://my-bucket/datasets/webdataset/audio/"
        }
    ],
    "random_crop": true
}
```

# Custom metadata
To customize the metadata provided to the conditioners during model training, you can provide a separate custom metadata module to the dataset config. This metadata module should be a Python file that must contain a function called `get_custom_metadata` that takes in two parameters, `info`, and `audio`, and returns a dictionary. 

For local training, the `info` parameter will contain a few pieces of information about the loaded audio file, such as the path, and information about how the audio was cropped from the original training sample. For WebDataset datasets, it will also contain the metadata from the related JSON files. 

The `audio` parameter contains the audio sample that will be passed to the model at training time. This lets you analyze the audio for extra properties that you can then pass in as extra conditioning signals.

The dictionary returned from the `get_custom_metadata` function will have its properties added to the `metadata` object used at training time. For more information on how conditioning works, please see the [Conditioning documentation](./conditioning.md)

## Example config and custom metadata module
```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "my_audio",
            "path": "/path/to/audio/dataset/",
            "custom_metadata_module": "/path/to/custom_metadata.py",
        }
    ],
    "random_crop": true
}
```

`custom_metadata.py`:
```py
def get_custom_metadata(info, audio):

    # Pass in the relative path of the audio file as the prompt
    return {"prompt": info["relpath"]}
```