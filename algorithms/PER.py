from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import torch

from skrl.memories.torch import Memory


class SumTree:
    """
    A binary sum tree data structure used for efficient sampling based on priorities.
    This is an essential component for prioritized experience replay.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize a SumTree with a given capacity.

        :param capacity: Maximum number of elements that can be stored in the tree
        :type capacity: int
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree structure to store priorities
        self.data = np.zeros(capacity, dtype=object)  # Array to store indices, not the full data
        self.n_entries = 0  # Current number of entries
        self.write_index = 0  # Current write position

    def _propagate(self, idx: int, change: float) -> None:
        """
        Propagate the priority update up through the tree.

        :param idx: Index of the element being updated
        :type idx: int
        :param change: Change in priority value
        :type change: float
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve the leaf node index for a given sample value s.

        :param idx: Current index in the tree
        :type idx: int
        :param s: Sample value
        :type s: float
        :return: Index of the leaf node
        :rtype: int
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """
        Get the total priority of the tree.

        :return: Sum of all priorities
        :rtype: float
        """
        return self.tree[0]

    def add(self, p: float, data: Any) -> None:
        """
        Add a new experience with priority p.

        :param p: Priority value
        :type p: float
        :param data: Memory index (not the full data)
        :type data: int
        """
        idx = self.write_index + self.capacity - 1

        self.data[self.write_index] = data
        self.update(idx, p)

        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float) -> None:
        """
        Update the priority of a specific node.

        :param idx: Index of the node to update
        :type idx: int
        :param p: New priority value
        :type p: float
        """
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get an experience based on a sample value s.

        :param s: Sample value
        :type s: float
        :return: (tree_index, priority, memory_index) tuple
        :rtype: tuple
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedMemory(Memory):
    def __init__(
            self,
            memory_size: int,
            num_envs: int = 1,
            device: Optional[Union[str, torch.device]] = None,
            export: bool = False,
            export_format: str = "pt",
            export_directory: str = "",
            alpha: float = 0.6,
            beta: float = 0.4,
            beta_increment: float = 0.001,
            epsilon: float = 0.01,
    ) -> None:
        """Prioritized Experience Replay (PER) memory

        Sample a batch from memory based on TD error priorities

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: ``1``)
        :type num_envs: int, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param export: Export the memory to a file (default: ``False``).
                       If True, the memory will be exported when the memory is filled
        :type export: bool, optional
        :param export_format: Export format (default: ``"pt"``).
                              Supported formats: torch (pt), numpy (np), comma separated values (csv)
        :type export_format: str, optional
        :param export_directory: Directory where the memory will be exported (default: ``""``).
                                 If empty, the agent's experiment directory will be used
        :type export_directory: str, optional
        :param alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization) (default: ``0.6``)
        :type alpha: float, optional
        :param beta: Importance sampling factor (0 = no correction, 1 = full correction) (default: ``0.4``)
        :type beta: float, optional
        :param beta_increment: Increment for beta after each sampling (default: ``0.001``)
        :type beta_increment: float, optional
        :param epsilon: Small constant to avoid zero priority (default: ``0.01``)
        :type epsilon: float, optional

        :raises ValueError: The export format is not supported
        """
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Create sum tree for prioritized experience storage
        self.sum_tree = SumTree(memory_size * num_envs)  # Adjust capacity to handle all transitions

        # Default priority for new experiences
        self._max_priority = 1.0

        # To store tree indexes for updating priorities
        self.tree_indexes = None
        self.sampling_indexes = None

    def _get_priority(self, error: Union[float, torch.Tensor, np.ndarray]) -> float:
        """Calculate priority based on error

        :param error: TD error or other priority metric
        :type error: float or torch.Tensor or numpy.ndarray
        :return: Priority value
        :rtype: float
        """
        try:
            if isinstance(error, torch.Tensor):
                error = error.abs().cpu().numpy()
            elif isinstance(error, float):
                error = abs(error)
            else:
                error = np.abs(error)

            # Clip error to prevent extreme values
            error = min(max(error, 0.0), 10.0)

            # Calculate priority with safety bounds
            priority = (error + self.epsilon) ** self.alpha

            # Ensure priority is within reasonable bounds
            return max(1e-8, min(priority, 1e8))

        except (ValueError, OverflowError) as e:
            print(f"Error calculating priority: {e}, using default value")
            return 1.0

    def add_samples(self, errors, **tensors: torch.Tensor) -> None:
        """Record samples in memory with prioritized experience replay

        :param errors: TD error(s) associated with the sample(s)
        :param tensors: Sampled data as key-value arguments where the keys are the names of the tensors to be modified.
                        Non-existing tensors will be skipped
        :raises ValueError: No tensors were provided or the tensors have incompatible shapes
        """
        if not tensors:
            raise ValueError(
                "No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)")

        # Use one of the tensors to determine shape
        tmp = tensors.get("states", tensors[next(iter(tensors))])
        dim, shape = tmp.ndim, tmp.shape

        # Ensure errors is a tensor or ndarray for consistent handling
        if isinstance(errors, (int, float)):
            errors = torch.tensor([errors])
        elif isinstance(errors, list):
            errors = torch.tensor(errors)

        # Clip priorities to prevent overflow
        if isinstance(errors, torch.Tensor):
            errors = torch.clamp(errors, -10.0, 10.0)
        elif isinstance(errors, np.ndarray):
            errors = np.clip(errors, -10.0, 10.0)

        # Store data in the main memory first
        # Multi-environment (num_envs == shape[0])
        if dim > 1 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index].copy_(tensor)

            # Store priorities in sum tree
            for i in range(self.num_envs):
                mem_idx = self.memory_index * self.num_envs + i
                priority = self._get_priority(errors[i] if errors.numel() > 1 else errors[0])
                self.sum_tree.add(priority, mem_idx)

            self.memory_index += 1

        # Multi-sample, fewer than num_envs
        elif dim > 1 and shape[0] < self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index: self.env_index + tensor.shape[0]].copy_(
                        tensor)

            # Store priorities in sum tree
            for i in range(tensor.shape[0]):
                mem_idx = self.memory_index * self.num_envs + self.env_index + i
                priority = self._get_priority(errors[i] if errors.numel() > 1 else errors[0])
                # Safety check for priority value
                priority = max(1e-8, min(priority, 1e8))
                self.sum_tree.add(priority, mem_idx)

            self.env_index += tensor.shape[0]

        # Single environment with multiple samples (num_envs == 1)
        elif dim > 1 and self.num_envs == 1:
            num_samples = min(shape[0], self.memory_size - self.memory_index)
            remaining_samples = shape[0] - num_samples

            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index: self.memory_index + num_samples].copy_(
                        tensor[:num_samples].unsqueeze(dim=1))
                    if remaining_samples > 0:
                        self.tensors[name][:remaining_samples].copy_(tensor[num_samples:].unsqueeze(dim=1))

            # Store priorities in sum tree
            for i in range(shape[0]):
                index = (self.memory_index + i) % self.memory_size
                priority = self._get_priority(errors[i] if errors.numel() > 1 else errors[0])
                # Safety check for priority value
                priority = max(1e-8, min(priority, 1e8))
                self.sum_tree.add(priority, index)

            self.memory_index = (self.memory_index + shape[0]) % self.memory_size

        # Single environment, single sample
        elif dim == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index].copy_(tensor)

            mem_idx = self.memory_index * self.num_envs + self.env_index
            priority = self._get_priority(errors)
            # Safety check for priority value
            priority = max(1e-8, min(priority, 1e8))
            self.sum_tree.add(priority, mem_idx)

            self.env_index += 1

        else:
            raise ValueError(f"Expected shape (number of environments = {self.num_envs}, data size), got {shape}")

        # Update indices and flags
        if self.env_index >= self.num_envs:
            self.env_index = 0
            self.memory_index += 1
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

            if self.export:
                self.save(directory=self.export_directory, format=self.export_format)

    def sample(
            self, names: Tuple[str], batch_size: int, mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample a batch from memory using prioritized experience replay

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names,
                 and importance sampling weights as the last element.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        # Update beta parameter (annealing)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Check if there's enough data in memory
        size = len(self)
        if sequence_length > 1:
            sequence_indexes = torch.arange(0, self.num_envs * sequence_length, self.num_envs)
            size -= sequence_indexes[-1].item()

        actual_batch_size = min(batch_size, size)

        # Check if we have any entries in the SumTree
        if self.sum_tree.n_entries == 0:
            # Fallback to uniform sampling if the tree is empty
            indexes = torch.randint(0, size, (actual_batch_size,), device=self.device)
            batches = self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)

            # Add uniform weights
            uniform_weights = torch.ones((actual_batch_size, 1), dtype=torch.float32, device=self.device)
            for i in range(len(batches)):
                batches[i].append(uniform_weights[i * actual_batch_size // mini_batches:
                                                  (i + 1) * actual_batch_size // mini_batches])
            return batches

        # Get the total priority and ensure it's valid
        total_priority = self.sum_tree.total()

        # Safety check: if total priority is invalid or too large, reset priorities
        if not np.isfinite(total_priority) or total_priority <= 0:
            print("Warning: SumTree total priority is invalid. Resetting all priorities.")
            # Reset the tree with uniform priorities
            self.sum_tree.tree = np.zeros(2 * self.sum_tree.capacity - 1)

            # Set leaf nodes to a small constant value
            for i in range(self.sum_tree.capacity - 1, 2 * self.sum_tree.capacity - 1):
                self.sum_tree.tree[i] = 1.0

            # Rebuild the tree by propagating values upward
            for i in range(self.sum_tree.capacity - 1, 0, -1):
                self.sum_tree.tree[(i - 1) // 2] = self.sum_tree.tree[i * 2 + 1] + self.sum_tree.tree[i * 2 + 2]

            total_priority = self.sum_tree.total()

        # Calculate segment size for sampling
        segment = total_priority / actual_batch_size

        priorities = []
        tree_indexes = []
        memory_indexes = []

        for i in range(actual_batch_size):
            try:
                # Safe calculations with bounds checking
                a = min(segment * i, total_priority)
                b = min(segment * (i + 1), total_priority)

                # Ensure valid range for sampling
                if a >= b:
                    if i == 0:
                        a = 0
                        b = total_priority / actual_batch_size
                    else:
                        a = (total_priority / actual_batch_size) * i
                        b = (total_priority / actual_batch_size) * (i + 1)

                # Limit to safe float range
                a = max(0, min(a, 1e38))
                b = max(a + 1e-10, min(b, 1e38))

                s = np.random.uniform(a, b)

                tree_idx, priority, memory_idx = self.sum_tree.get(s)

                # Ensure the priority is valid
                if not np.isfinite(priority) or priority <= 0:
                    priority = 1.0

                priorities.append(priority)
                tree_indexes.append(tree_idx)
                memory_indexes.append(memory_idx)

            except (OverflowError, ValueError) as e:
                print(f"Error during sampling: {e}, using fallback sampling")
                # Fallback: sample uniformly for this element
                memory_idx = np.random.randint(0, self.sum_tree.n_entries)
                memory_indexes.append(memory_idx)
                tree_indexes.append(self.capacity - 1 + memory_idx % self.sum_tree.capacity)
                priorities.append(1.0)

        # Calculate importance sampling weights with safety checks
        try:
            sampling_probabilities = np.array(priorities) / total_priority
            # Clip probabilities to avoid extreme values
            sampling_probabilities = np.clip(sampling_probabilities, 1e-10, 1.0)

            # Avoid extreme exponentiation
            beta_adjusted = min(self.beta, 0.9999)  # Limit beta to avoid extreme values
            is_weights = np.power(self.sum_tree.n_entries * sampling_probabilities, -beta_adjusted)

            # Safety normalization
            max_weight = np.max(is_weights)
            if max_weight <= 0 or not np.isfinite(max_weight):
                is_weights = np.ones_like(is_weights)
            else:
                is_weights /= max_weight  # Normalize

        except (OverflowError, ValueError, ZeroDivisionError) as e:
            print(f"Error calculating weights: {e}, using uniform weights")
            is_weights = np.ones(len(priorities))

        # Store the tree indexes for later priority updates
        self.tree_indexes = tree_indexes

        # Convert to torch tensor with safety checks for invalid values
        try:
            memory_indexes = torch.tensor(memory_indexes, dtype=torch.long, device=self.device)
            is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        except Exception as e:
            print(f"Error converting to tensors: {e}, using fallback")
            memory_indexes = torch.randint(0, size, (actual_batch_size,), device=self.device)
            is_weights = torch.ones((actual_batch_size, 1), dtype=torch.float32, device=self.device)

        # Store for later use in update_priorities
        self.sampling_indexes = memory_indexes

        # Get the samples using the standard sampling method with our calculated indexes
        try:
            batches = self.sample_by_index(names=names, indexes=memory_indexes, mini_batches=mini_batches)
        except Exception as e:
            print(f"Error in sample_by_index: {e}, using fallback uniform sampling")
            # Fallback to standard sampling
            return super().sample(names, batch_size, mini_batches, sequence_length)

        # Add importance sampling weights to each mini-batch
        for i in range(len(batches)):
            batches[i].append(
                is_weights[i * actual_batch_size // mini_batches:(i + 1) * actual_batch_size // mini_batches])

        return batches

    def update_priorities(self, errors: torch.Tensor) -> None:
        """Update priorities based on TD errors

        :param errors: TD errors for the sampled transitions
        :type errors: torch.Tensor
        """
        if not hasattr(self, 'sampling_indexes') or not hasattr(self, 'tree_indexes') or \
                self.sampling_indexes is None or self.tree_indexes is None:
            return

        # Clip errors to prevent extreme priority values
        errors = errors.detach().abs().clamp(0.0, 10.0).cpu().numpy()

        for i, error in enumerate(errors):
            try:
                # Calculate new priority
                priority = self._get_priority(error)

                # Ensure priority is within sensible bounds
                priority = max(1e-8, min(priority, 1e8))

                # Update max priority
                self._max_priority = max(self._max_priority, priority)

                # Update sum tree
                self.sum_tree.update(self.tree_indexes[i], priority)

            except (IndexError, ValueError, OverflowError) as e:
                print(f"Error updating priority at index {i}: {e}")
                continue