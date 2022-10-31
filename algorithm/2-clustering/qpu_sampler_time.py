# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Classical and quantum :class:`.Runnable`
`dimod <https://docs.ocean.dwavesys.com/en/stable/docs_dimod/sdk_index.html>`_
samplers for problems and subproblems.
"""

import time
import logging
import threading
from collections import namedtuple

import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import AutoEmbeddingComposite, FixedEmbeddingComposite
from dwave.embedding.chimera import find_clique_embedding as find_chimera_clique_embedding
from dwave.embedding.pegasus import find_clique_embedding as find_pegasus_clique_embedding

from tabu import TabuSampler
from neal import SimulatedAnnealingSampler
from greedy import SteepestDescentSolver

from hybrid.core import Runnable, SampleSet
from hybrid.flow import Loop
from hybrid.utils import random_sample
from hybrid import traits

__all__ = [
    'QPUTimeSubproblemAutoEmbeddingSampler'
    ]

logger = logging.getLogger(__name__)

 #
 # Updated by Sue Mniszewski
 # Updated QPUTimeSubproblemAutoEmbeddingSampler to collect number of calls and timing infomation
 #

class QPUTimeSubproblemAutoEmbeddingSampler(traits.SubproblemSampler, traits.SISO, Runnable):
    r"""A quantum sampler for a subproblem with automated heuristic
    minor-embedding.
    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.
        num_retries (int, optional, default=0):
            Number of times the sampler will retry to embed if a failure occurs.
        qpu_sampler (:class:`dimod.Sampler`, optional, default=\ :class:`~dwave.system.samplers.DWaveSampler()`):
            Quantum sampler such as a D-Wave system. Subproblems that do not fit the
            sampler's structure are minor-embedded on the fly with
            :class:`~dwave.system.composites.AutoEmbeddingComposite`.
        sampling_params (dict):
            Dictionary of keyword arguments with values that will be used
            on every call of the (embedding-wrapped QPU) sampler.
        auto_embedding_params (dict, optional):
            If provided, parameters are passed to the
            :class:`~dwave.system.composites.AutoEmbeddingComposite` constructor
            as keyword arguments.
    See :ref:`samplers-examples`.
    """

    def __init__(self, num_reads=100, num_retries=0, qpu_sampler=None, sampling_params=None,
                 auto_embedding_params=None, **runopts):
        super(QPUTimeSubproblemAutoEmbeddingSampler, self).__init__(**runopts)

        self.num_reads = num_reads
        self.num_retries = num_retries
        self.num_accesses = 0
        self.total_qpu_time = 0

        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()

        if sampling_params is None:
            sampling_params = {}
        self.sampling_params = sampling_params

        # embed on the fly and only if needed
        if auto_embedding_params is None:
            auto_embedding_params = {}
        self.sampler = AutoEmbeddingComposite(qpu_sampler, **auto_embedding_params)

    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r}, "
                       "sampling_params={self.sampling_params!r})").format(self=self)

    def next(self, state, **runopts):
        num_reads = runopts.get('num_reads', self.num_reads)
        sampling_params = runopts.get('sampling_params', self.sampling_params)

        params = sampling_params.copy()
        params.update(num_reads=num_reads)

        num_retries = runopts.get('num_retries', self.num_retries)

        embedding_success = False
        num_tries = 0

        while not embedding_success:
            try:
                num_tries += 1
                response = self.sampler.sample(state.subproblem, **params)
            except ValueError as exc:
                if num_tries <= num_retries:
                    pass
                else:
                    raise exc
            else:
                embedding_success = True

            # Collect number of times called and timing information
            self.num_accesses += 1
            self.total_qpu_time += response.info['timing']['qpu_access_time']

        return state.updated(subsamples=response)

