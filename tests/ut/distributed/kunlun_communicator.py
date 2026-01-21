"""kunlun_communicator"""
from contextlib import contextmanager
from typing import Optional

import torch
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

class KunlunCommunicator(CudaCommunicator):
    """KunlunCommunicator"""
    def __init__(self,
            device,
            device_group,
            cpu_group,
            unique_name):
        """
            Initializes the CUDA Communicator.
        
        Args:
            cpu_group (ProcessGroup): The CPU process group.
            device (Optional[torch.device], optional): The device to use. Defaults to None.
            device_group (Optional[ProcessGroup], optional): The device process group. Defaults to None.
            unique_name (str, optional): The unique name of this communicator. Defaults to "".
        
        Raises:
            ValueError: If both ``device`` and ``device_group`` are not specified.
        """        
        DeviceCommunicatorBase.__init__(self, cpu_group, device, device_group, unique_name)
        self.ca_comm = None
        self.disabled = False
        with torch.cuda.device(device):
            self.stream = torch.cuda.Stream()

            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            self.stream.synchronize()
            del data

    def all_reduce(self, input_):
        """all_reduce"""
        return DeviceCommunicatorBase.all_reduce(self, input_)

    def all_gather(self, input_, dim):
        """all_gather"""
        return DeviceCommunicatorBase.all_gather(self, input_, dim)

    def gather(self, input_, dst, dim):
        """gather"""
        return DeviceCommunicatorBase.gather(self, input_, dst, dim)

    def send(self, tensor, dst):
        """send"""
        DeviceCommunicatorBase.send(self, tensor, dst)

    def recv(self, size, dtype, src):
        """recv"""
        return DeviceCommunicatorBase.recv(self, size, dtype, src)

    def destroy(self):
        """destroy"""
        pass

    @contextmanager
    def change_state(self, enable, stream):
        """
        A context manager to change the state of the communicator.
        """
        if enable is None:
            # guess a default value when not specified
            enable = self.available

        if stream is None:
            stream = self.stream

        old_disable = self.disabled
        old_stream = self.stream

        self.stream = stream
        self.disabled = not enable
        yield

        self.disabled = old_disable
        self.stream = old_stream