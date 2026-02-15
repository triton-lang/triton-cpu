from __future__ import annotations

from ..backends import backends, DriverBase

from typing import Any, Callable, Generic, TypeVar, Union

import os

def _create_driver() -> DriverBase:
    if os.getenv("TRITON_CPU_BACKEND", "0") == "1":
        if "cpu" not in backends:
            raise RuntimeError("TRITON_CPU_BACKEND is set, but CPU backend is unavailable.")
        return backends["cpu"].driver()
    active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(active_drivers) >= 2 and backends["cpu"].driver.is_active():
        print("Both CPU and GPU backends are available. Using the GPU backend.")
        active_drivers.remove(backends["cpu"].driver)
    if len(active_drivers) != 1:
        raise RuntimeError(f"{len(active_drivers)} active drivers ({active_drivers}). There should only be one.")
    return active_drivers[0]()


T = TypeVar("T")


class LazyProxy(Generic[T]):

    def __init__(self, init_fn: Callable[[], T]) -> None:
        self._init_fn = init_fn
        self._obj: Union[T, None] = None

    def _initialize_obj(self) -> T:
        if self._obj is None:
            self._obj = self._init_fn()
        return self._obj

    def __getattr__(self, name) -> Any:
        return getattr(self._initialize_obj(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["_init_fn", "_obj"]:
            super().__setattr__(name, value)
        else:
            setattr(self._initialize_obj(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._initialize_obj(), name)

    def __repr__(self) -> str:
        if self._obj is None:
            return f"<{self.__class__.__name__} for {self._init_fn} not yet initialized>"
        return repr(self._obj)

    def __str__(self) -> str:
        return str(self._initialize_obj())


class DriverConfig:

    def __init__(self) -> None:
        self.default: LazyProxy[DriverBase] = LazyProxy(_create_driver)
        self.active: Union[LazyProxy[DriverBase], DriverBase] = self.default

    def set_active(self, driver: DriverBase) -> None:
        self.active = driver

    def reset_active(self) -> None:
        self.active = self.default

    def set_active_to_cpu(self):
        if "cpu" not in backends:
            raise RuntimeError("CPU backend is unavailable")
        self.active = backends["cpu"].driver()

    def set_active_to_gpu(self):
        active_gpus = [(name, backend.driver)
                       for name, backend in backends.items()
                       if backend.driver.is_active() and name != "cpu"]
        if len(active_gpus) != 1:
            raise RuntimeError(f"{len(active_gpus)} active GPU drivers ({active_gpus}). There should only be one GPU.")
        self.active = active_gpus[0][1]()
        return active_gpus[0][0]

    def get_active_gpus(self):
        return [name for name, backend in backends.items() if backend.driver.is_active() and name != "cpu"]


driver = DriverConfig()
