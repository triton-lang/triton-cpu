import os

from ..backends import backends
from ..backends import DriverBase


def _create_driver():
    cpu = int(os.getenv("TRITON_CPU_BACKEND", "0"))
    if cpu >= 1:
        if "cpu" not in backends:
            raise RuntimeError("TRITON_CPU_BACKEND is set, but CPU backend is unavailable.")
        if cpu == 1:
            return backends["cpu"].driver()
        elif cpu >= 2:
            return backends["nvidia"].driver(cpu_mode=True)

    actives = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(actives) >= 2 and backends["cpu"].driver.is_active():
        print("Both CPU and GPU backends are available. Using the GPU backend.")
        actives.remove(backends["cpu"].driver)
    if len(actives) != 1:
        raise RuntimeError(f"{len(actives)} active drivers ({actives}). There should only be one.")
    return actives[0]()


class LazyProxy:

    def __init__(self, init_fn):
        self._init_fn = init_fn
        self._obj = None

    def _initialize_obj(self):
        if self._obj is None:
            self._obj = self._init_fn()

    def __getattr__(self, name):
        self._initialize_obj()
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name in ["_init_fn", "_obj"]:
            super().__setattr__(name, value)
        else:
            self._initialize_obj()
            setattr(self._obj, name, value)

    def __delattr__(self, name):
        self._initialize_obj()
        delattr(self._obj, name)

    def __repr__(self):
        if self._obj is None:
            return f"<{self.__class__.__name__} for {self._init_fn} not yet initialized>"
        return repr(self._obj)

    def __str__(self):
        self._initialize_obj()
        return str(self._obj)


class DriverConfig:

    def __init__(self):
        self.default = LazyProxy(_create_driver)
        self.active = self.default

    def set_active(self, driver: DriverBase):
        self.active = driver

    def reset_active(self):
        self.active = self.default

    def set_active_to_cpu(self, experimental=False):
        if "cpu" not in backends:
            raise RuntimeError("CPU backend is unavailable")
        if not experimental:
            self.active = backends["cpu"].driver()
        else:
            if "nvidia" not in backends:
                raise RuntimeError("New CPU mode is implemented on Nvidia backend")
            self.active = backends["nvidia"].driver(cpu_mode=True)

    def set_active_to_gpu(self):
        active_gpus = [(name, backend.driver)
                       for name, backend in backends.items()
                       if backend.driver.is_active() and name != "cpu"]
        if len(active_gpus) != 1:
            raise RuntimeError(f"{len(active_gpus)} active GPU drivers ({active_gpus}). There should only be one GPU.")
        self.active = active_gpus[0][1](cpu_mode=False)
        return active_gpus[0][0]

    def get_active_gpus(self):
        return [name for name, backend in backends.items() if backend.driver.is_active() and name != "cpu"]


driver = DriverConfig()
