# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""CUDA graph utilities."""

import gc

from dragon.core.device import cuda as cuda_internal
from dragon.core.framework import backend
from dragon.core.framework import context
from dragon.core.framework import tapes
from dragon.core.framework import workspace


class TraceGraph(object):
    """Graph to capture traceable operators."""

    def __init__(self):
        """Create a ``TraceGraph``."""
        self.pool = None
        self.tape = None
        self.variable_scope = None
        self.workspace = None

    def capture_begin(self, pool=None, **kwargs):
        """Begin the graph capture.

        Parameters
        ----------
        pool : str, optional
            The memory pool name.

        """
        if self.tape is not None:
            raise RuntimeError("Graph has captured.")
        self.pool = pool if pool else f"{id(self)}"
        self.tape = tapes.Tape()
        self.variable_scope = context.variable_scope(f"{self.pool}/Variable")
        self.workspace = workspace.get_workspace()
        self.tape._tracing = f"{self.pool}/"
        self.tape.__enter__()
        self.variable_scope.__enter__()

    def capture_end(self):
        """End the capture."""
        self.variable_scope.__exit__(None, None, None)
        self.tape.__exit__(None, None, None)
        self.variable_scope = None

    def replay(self):
        """Launch graph on current workspace."""
        self.workspace.run_operator(self.tape.get_elements())

    def reset(self):
        """Reset the graph capture."""
        self.pool = None
        self.tape = None


class CUDAGraph(TraceGraph, backend.CUDAGraph):
    """Graph to capture cuda kernels."""

    def __init__(self):
        """Create a ``CUDAGraph``."""
        super(CUDAGraph, self).__init__()
        super(TraceGraph, self).__init__()
        self.capture_error_mode = "global"

    def capture_begin(self, pool=None, capture_error_mode='global'):
        """Begin the graph capture."""
        super(CUDAGraph, self).capture_begin(pool)
        self.capture_error_mode = capture_error_mode

    def capture_end(self):
        """End the graph capture."""
        super(CUDAGraph, self).capture_end()
        device_index = cuda_internal.current_device()
        stream_index = self.workspace.get_stream()
        cuda_internal.synchronize(device_index, stream_index)
        self.BeginCapture(device_index, stream_index, self.capture_error_mode)
        self.workspace.run_operator(self.tape.get_elements())
        self.EndCapture()

    def replay(self):
        """Launch graph on captured stream."""
        self.Launch()

    def reset(self):
        """Reset the graph capture."""
        super(CUDAGraph, self).reset()
        self.Reset()


class graph(object):
    """Context-manager to capture a graph."""

    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode="global"):
        """Create a ``graph`` context manager.

        Parameters
        ----------
        cuda_graph : Union[torch.cuda.OpGraph, torch.cuda.CUDAGraph]
            The graph for capturing.
        pool : str, optional
            The memory pool name.
        stream : int, optional
            The index of stream to capture on.
        capture_error_mode : str, optional
            The capture mode.

        """
        self.pool = pool if pool else f"{id(cuda_graph)}"
        self.capture_stream = stream if stream is not None else 0
        self.cuda_graph = cuda_graph
        self.capture_error_mode = capture_error_mode
        self.stream_ctx = None

    def __enter__(self):
        """Enter the graph capture."""
        gc.collect()
        stream_stack = workspace.get_workspace()._stream_stack
        self.stream_ctx = stream_stack.get_controller(self.capture_stream)
        self.stream_ctx.__enter__()
        self.cuda_graph.capture_begin(self.pool, capture_error_mode=self.capture_error_mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the graph capture."""
        self.cuda_graph.capture_end()
        self.stream_ctx.__exit__(exc_type, exc_value, traceback)
        self.stream_ctx = None
