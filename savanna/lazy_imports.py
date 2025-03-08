from contextlib import contextmanager
import os

import lazy_import_plus as lazy_import

lazy_import.lazy_module("deepspeed.launcher.runner")
lazy_import.lazy_class("deepspeed.pipe.PipelineModule")
lazy_import.lazy_class("deepspeed.pipe.LayerSpec")
lazy_import.lazy_class("deepspeed.pipe.TiedLayerSpec")
lazy_import.lazy_module("wandb")


@contextmanager
def transformer_engine_on_import():
    # By default, transformer engine 1.10 will run a recursive glob trying
    # to find libnvrtc. This becomes especially expensive on NFS.
    #
    # Disable auto-loading of cudnn / nvrtc / library for transformer_engine 1.10.0
    os.environ["NVTE_PROJECT_BUILDING"] = "1"
    yield
    import transformer_engine

    if not transformer_engine.__version__.startswith("1.10.0"):
        return

    from transformer_engine import common
    from transformer_engine.common import set_conda_path_vars, _load_cudnn, _load_library

    set_conda_path_vars()
    common._CUDNN_LIB_CTYPES = _load_cudnn()
    common._NVRTC_LIB_CTYPES = _load_nvrtc()
    common._TE_LIB_CTYPES = _load_library()


def _load_nvrtc():
    """Load NVRTC shared library."""
    import ctypes
    import glob
    import subprocess
    from transformer_engine.common import _get_sys_extension

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        libs = glob.glob(f"{cuda_home}/lib/libnvrtc.{_get_sys_extension()}*")
        if not libs:
            libs = glob.glob(f"{cuda_home}/**/libnvrtc.{_get_sys_extension()}*", recursive=True)
        libs = list(filter(lambda x: not ("stub" in x or "libnvrtc-builtins" in x), libs))
        libs.sort(reverse=True, key=os.path.basename)
        if libs:
            return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

    libs = subprocess.check_output("ldconfig -p | grep 'libnvrtc'", shell=True)
    libs = libs.decode("utf-8").split("\n")
    sos = []
    for lib in libs:
        if "stub" in lib or "libnvrtc-builtins" in lib:
            continue
        if "libnvrtc" in lib and "=>" in lib:
            sos.append(lib.split(">")[1].strip())
    if sos:
        return ctypes.CDLL(sos[0], mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(f"libnvrtc.{_get_sys_extension()}", mode=ctypes.RTLD_GLOBAL)


# lazy_import.lazy_module('transformer_engine', on_import=transformer_engine_on_import())
