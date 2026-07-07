from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["transmits", "fbins", "elements"],
    meta_fields=[],
)
@dataclass
class Indices:
    transmits: jnp.ndarray
    fbins: jnp.ndarray
    elements: jnp.ndarray

    @property
    def size(self):
        return self.transmits.size

    @classmethod
    def get_full(cls, n_tx, n_fbins, n_el):
        transmits, fbins, elements = jnp.meshgrid(
            jnp.arange(n_tx),
            jnp.arange(n_fbins),
            jnp.arange(n_el),
            indexing="ij",
        )
        return cls(
            transmits=transmits.ravel(),
            fbins=fbins.ravel(),
            elements=elements.ravel(),
        )

    def __getitem__(self, key):
        return Indices(
            transmits=self.transmits[key],
            fbins=self.fbins[key],
            elements=self.elements[key],
        )

    def sample_all_elements(self, key, n_samples, n_el):
        """Sample (frame, transmit, fbin) combos randomly, including all elements for each.

        Assumes indices are ordered so that groups of n_el consecutive entries
        share the same (frame, transmit, fbin) values (as produced by get_full).

        Args:
            key: JAX PRNG key.
            n_samples: Number of (frame, transmit, fbin) combinations to sample.
            n_el: Number of elements per combination.

        Returns:
            Flat index array of shape (n_samples * n_el,) into self.
        """
        n_groups = self.size // n_el
        group_idx = jax.random.choice(key, n_groups, (n_samples,), replace=False)
        flat_idx = (group_idx[:, None] * n_el + jnp.arange(n_el)[None, :]).ravel()
        return flat_idx

    def sample_all_elements_and_frames(self, key, n_samples, n_el, n_frames):
        """Sample (transmit, fbin) combos randomly, including all elements and frames for each.

        Assumes indices are ordered as (frames, transmits, fbins, elements) with
        elements varying fastest (as produced by get_full).

        Args:
            key: JAX PRNG key.
            n_samples: Number of (transmit, fbin) combinations to sample.
            n_el: Number of elements per combination.
            n_frames: Number of frames.

        Returns:
            Flat index array of shape (n_samples * n_frames * n_el,) into self.
        """
        n_groups = self.size // (n_el * n_frames)  # n_tx * n_fbins
        group_idx = jax.random.choice(key, n_groups, (n_samples,), replace=False)

        frame_stride = n_groups * n_el
        frame_offsets = jnp.arange(n_frames) * frame_stride  # (n_frames,)
        group_offsets = group_idx * n_el  # (n_samples,)
        element_offsets = jnp.arange(n_el)  # (n_el,)

        flat_idx = (
            frame_offsets[:, None, None]
            + group_offsets[None, :, None]
            + element_offsets[None, None, :]
        ).ravel()
        return flat_idx

    def reshape(self, new_shape):
        """Reshapes the sample indices."""
        tx = self.transmits.reshape(new_shape)
        fbins = self.fbins.reshape(new_shape)
        el = self.elements.reshape(new_shape)
        return Indices(transmits=tx, fbins=fbins, elements=el)
