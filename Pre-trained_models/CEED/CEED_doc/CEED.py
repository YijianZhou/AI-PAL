# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Address all TODOs and remove all explanatory comments
# Lint as: python3
"""CEED: California Earthquake Dataset for Machine Learning and Cloud Computing"""


from typing import Dict, List, Optional, Tuple, Union

import datasets
import fsspec
import h5py
import numpy as np
import torch

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {CEED: California Earthquake Dataset for Machine Learning and Cloud Computing},
author={Zhu et al.},
year={2025}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
A dataset of earthquake waveforms organized by earthquake events and based on the HDF5 format.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_REPO_NC = "https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/waveform_h5"
_FILES_NC = [
    "1987.h5",
    "1988.h5",
    "1989.h5",
    "1990.h5",
    "1991.h5",
    "1992.h5",
    "1993.h5",
    "1994.h5",
    "1995.h5",
    "1996.h5",
    "1997.h5",
    "1998.h5",
    "1999.h5",
    "2000.h5",
    "2001.h5",
    "2002.h5",
    "2003.h5",
    "2004.h5",
    "2005.h5",
    "2006.h5",
    "2007.h5",
    "2008.h5",
    "2009.h5",
    "2010.h5",
    "2011.h5",
    "2012.h5",
    "2013.h5",
    "2014.h5",
    "2015.h5",
    "2016.h5",
    "2017.h5",
    "2018.h5",
    "2019.h5",
    "2020.h5",
    "2021.h5",
    "2022.h5",
    "2023.h5",
]
_REPO_SC = "https://huggingface.co/datasets/AI4EPS/quakeflow_sc/resolve/main/waveform_h5"
_FILES_SC = [
    "1999.h5",
    "2000.h5",
    "2001.h5",
    "2002.h5",
    "2003.h5",
    "2004.h5",
    "2005.h5",
    "2006.h5",
    "2007.h5",
    "2008.h5",
    "2009.h5",
    "2010.h5",
    "2011.h5",
    "2012.h5",
    "2013.h5",
    "2014.h5",
    "2015.h5",
    "2016.h5",
    "2017.h5",
    "2018.h5",
    "2019_0.h5",
    "2019_1.h5",
    "2019_2.h5",
    "2020_0.h5",
    "2020_1.h5",
    "2021.h5",
    "2022.h5",
    "2023.h5",
]

_URLS = {
    "train": [f"{_REPO_NC}/{x}" for x in _FILES_NC[:-1]] + [f"{_REPO_SC}/{x}" for x in _FILES_SC[:-1]],
    "test": [f"{_REPO_NC}/{x}" for x in _FILES_NC[-1:]] + [f"{_REPO_SC}/{x}" for x in _FILES_SC[-1:]],
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class CEED(datasets.GeneratorBasedBuilder):
    """CEED: A dataset of earthquake waveforms organized by earthquake events and based on the HDF5 format."""

    VERSION = datasets.Version("1.1.0")

    nt = 8192

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    # default config, you can change batch_size and num_stations_list when use `datasets.load_dataset`
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="station", version=VERSION, description="yield station-based samples one by one of whole dataset"
        ),
        datasets.BuilderConfig(
            name="event", version=VERSION, description="yield event-based samples one by one of whole dataset"
        ),
        datasets.BuilderConfig(
            name="station_train",
            version=VERSION,
            description="yield station-based samples one by one of training dataset",
        ),
        datasets.BuilderConfig(
            name="event_train", version=VERSION, description="yield event-based samples one by one of training dataset"
        ),
        datasets.BuilderConfig(
            name="station_test", version=VERSION, description="yield station-based samples one by one of test dataset"
        ),
        datasets.BuilderConfig(
            name="event_test", version=VERSION, description="yield event-based samples one by one of test dataset"
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        "station_test"  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if (
            (self.config.name == "station")
            or (self.config.name == "station_train")
            or (self.config.name == "station_test")
        ):
            features = datasets.Features(
                {
                    "data": datasets.Array2D(shape=(3, self.nt), dtype="float32"),
                    "phase_time": datasets.Sequence(datasets.Value("string")),
                    "phase_index": datasets.Sequence(datasets.Value("int32")),
                    "phase_type": datasets.Sequence(datasets.Value("string")),
                    "phase_polarity": datasets.Sequence(datasets.Value("string")),
                    "begin_time": datasets.Value("string"),
                    "end_time": datasets.Value("string"),
                    "event_time": datasets.Value("string"),
                    "event_time_index": datasets.Value("int32"),
                    "event_location": datasets.Sequence(datasets.Value("float32")),
                    "station_location": datasets.Sequence(datasets.Value("float32")),
                },
            )
        elif (self.config.name == "event") or (self.config.name == "event_train") or (self.config.name == "event_test"):
            features = datasets.Features(
                {
                    "data": datasets.Array3D(shape=(None, 3, self.nt), dtype="float32"),
                    "phase_time": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "phase_index": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                    "phase_type": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "phase_polarity": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "begin_time": datasets.Value("string"),
                    "end_time": datasets.Value("string"),
                    "event_time": datasets.Value("string"),
                    "event_time_index": datasets.Value("int32"),
                    "event_location": datasets.Sequence(datasets.Value("float32")),
                    "station_location": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                },
            )
        else:
            raise ValueError(f"config.name = {self.config.name} is not in BUILDER_CONFIGS")

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        if self.config.name in ["station", "event"]:
            urls = _URLS["train"] + _URLS["test"]
        elif self.config.name in ["station_train", "event_train"]:
            urls = _URLS["train"]
        elif self.config.name in ["station_test", "event_test"]:
            urls = _URLS["test"]
        else:
            raise ValueError("config.name is not in BUILDER_CONFIGS")

        # files = dl_manager.download(urls)
        files = dl_manager.download_and_extract(urls)
        # files = ["waveform_h5/1989.h5", "waveform_h5/1990.h5"]
        print(files)

        if self.config.name == "station" or self.config.name == "event":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": files[:-2],
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": files[-2:], "split": "test"},
                ),
            ]
        elif self.config.name == "station_train" or self.config.name == "event_train":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": files,
                        "split": "train",
                    },
                ),
            ]
        elif self.config.name == "station_test" or self.config.name == "event_test":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": files, "split": "test"},
                ),
            ]
        else:
            raise ValueError("config.name is not in BUILDER_CONFIGS")

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        for file in filepath:
            with fsspec.open(file, "rb") as fs:
                with h5py.File(fs, "r") as fp:
                    event_ids = list(fp.keys())
                    for event_id in event_ids:
                        event = fp[event_id]
                        event_attrs = event.attrs
                        begin_time = event_attrs["begin_time"]
                        end_time = event_attrs["end_time"]
                        event_location = [
                            event_attrs["longitude"],
                            event_attrs["latitude"],
                            event_attrs["depth_km"],
                        ]
                        event_time = event_attrs["event_time"]
                        event_time_index = event_attrs["event_time_index"]
                        station_ids = list(event.keys())
                        if len(station_ids) == 0:
                            continue
                        if ("station" in self.config.name):
                            waveforms = np.zeros([3, self.nt], dtype="float32")

                            for i, sta_id in enumerate(station_ids):
                                waveforms[:, : self.nt] = event[sta_id][:, : self.nt]
                                attrs = event[sta_id].attrs
                                phase_type = attrs["phase_type"]
                                phase_time = attrs["phase_time"]
                                phase_index = attrs["phase_index"]
                                phase_polarity = attrs["phase_polarity"]
                                station_location = [attrs["longitude"], attrs["latitude"], -attrs["elevation_m"] / 1e3]

                                yield f"{event_id}/{sta_id}", {
                                    "data": waveforms,
                                    "phase_time": phase_time,
                                    "phase_index": phase_index,
                                    "phase_type": phase_type,
                                    "phase_polarity": phase_polarity,
                                    "begin_time": begin_time,
                                    "end_time": end_time,
                                    "event_time": event_time,
                                    "event_time_index": event_time_index,
                                    "event_location": event_location,
                                    "station_location": station_location,
                                }

                        elif ("event" in self.config.name):
                            waveforms = np.zeros([len(station_ids), 3, self.nt], dtype="float32")
                            phase_type = []
                            phase_time = []
                            phase_index = []
                            phase_polarity = []
                            station_location = []

                            for i, sta_id in enumerate(station_ids):
                                waveforms[i, :, : self.nt] = event[sta_id][:, : self.nt]
                                attrs = event[sta_id].attrs
                                phase_type.append(list(attrs["phase_type"]))
                                phase_time.append(list(attrs["phase_time"]))
                                phase_index.append(list(attrs["phase_index"]))
                                phase_polarity.append(list(attrs["phase_polarity"]))
                                station_location.append(
                                    [attrs["longitude"], attrs["latitude"], -attrs["elevation_m"] / 1e3]
                                )
                            yield event_id, {
                                "data": waveforms,
                                "phase_time": phase_time,
                                "phase_index": phase_index,
                                "phase_type": phase_type,
                                "phase_polarity": phase_polarity,
                                "begin_time": begin_time,
                                "end_time": end_time,
                                "event_time": event_time,
                                "event_time_index": event_time_index,
                                "event_location": event_location,
                                "station_location": station_location,
                            }
