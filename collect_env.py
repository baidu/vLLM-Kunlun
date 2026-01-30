# SPDX-License-Identifier: Apache-2.0
# vLLM-Kunlun Environment Information Collection Tool (Fixed Version)
"""
Environment information collection script for Kunlun XPU
Fixed the following issues:
1. Device name displayed as "GPU" → Now correctly shows "P800 OAM"
2. XRE version command error → Now parsed from xpu-smi output
3. vLLM-Kunlun version hardcoded → Now fetched from pip package metadata
"""

import os
import subprocess
import sys
import re
from collections import namedtuple

# =============================================================================
# Part 1: Basic Utility Functions
# =============================================================================


def run(command):
    """
    Execute shell command and return result
    [Principle Explanation - Web Development Analogy]
    This is like the fetch() function in frontend development, sending a request and getting a response.
    - command: The command to execute (similar to a URL)
    - returns: (return_code, stdout, stderr)
    Args:
        command: Command as string or list
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    shell = True if isinstance(command, str) else False
    try:
        p = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture error output
            shell=shell,
        )
        raw_output, raw_err = p.communicate()
        rc = p.returncode
        # Decode byte stream to string
        output = raw_output.decode("utf-8").strip()
        err = raw_err.decode("utf-8").strip()
        return rc, output, err
    except FileNotFoundError:
        return 127, "", "Command not found"


def run_and_read_all(run_lambda, command):
    """Execute command, return output if successful, None otherwise"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Execute command and extract first regex match"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


# Check if PyTorch is available
try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False


# =============================================================================
# Part 2: General System Information Collection (Reusing vLLM Original Logic)
# =============================================================================


def get_platform():
    """Get operating system platform"""
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win32"):
        return "win32"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    return sys.platform


def get_os(run_lambda):
    """Get detailed operating system information"""
    from platform import machine

    if get_platform() == "linux":
        # Try reading /etc/*-release
        rc, out, _ = run_lambda(
            "cat /etc/*-release 2>/dev/null | grep PRETTY_NAME | head -1"
        )
        if rc == 0 and out:
            match = re.search(r'PRETTY_NAME="(.*)"', out)
            if match:
                return f"{match.group(1)} ({machine()})"
        # Fallback: use lsb_release
        rc, out, _ = run_lambda("lsb_release -d 2>/dev/null")
        if rc == 0 and out:
            match = re.search(r"Description:\s*(.*)", out)
            if match:
                return f"{match.group(1)} ({machine()})"
    return f"{get_platform()} ({machine()})"


def get_gcc_version(run_lambda):
    """Get GCC version"""
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")


def get_clang_version(run_lambda):
    """Get Clang version"""
    return run_and_parse_first_match(
        run_lambda, "clang --version", r"clang version (.*)"
    )


def get_cmake_version(run_lambda):
    """Get CMake version"""
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")


def get_libc_version():
    """Get libc version"""
    import platform

    if get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())


def get_python_platform():
    """Get Python platform information"""
    import platform

    return platform.platform()


def get_cpu_info(run_lambda):
    """Get CPU information"""
    if get_platform() == "linux":
        rc, out, err = run_lambda("lscpu")
        return out if rc == 0 else err
    return "N/A"


def get_pip_packages(run_lambda, patterns=None):
    """Get pip package list"""
    if patterns is None:
        patterns = {
            "torch",
            "numpy",
            "triton",
            "transformers",
            "vllm",
            "kunlun",
            "xpu",
            "bkcl",
            "xmlir",
        }

    cmd = [sys.executable, "-mpip", "list", "--format=freeze"]
    out = run_and_read_all(run_lambda, cmd)
    if out is None:
        return "pip3", ""

    filtered = "\n".join(
        line
        for line in out.splitlines()
        if any(name.lower() in line.lower() for name in patterns)
    )
    pip_version = "pip3" if sys.version[0] == "3" else "pip"
    return pip_version, filtered


def get_conda_packages(run_lambda, patterns=None):
    """Get conda package list"""
    if patterns is None:
        patterns = {
            "torch",
            "numpy",
            "triton",
            "transformers",
            "kunlun",
            "xpu",
            "bkcl",
            "xmlir",
        }

    conda = os.environ.get("CONDA_EXE", "conda")
    out = run_and_read_all(run_lambda, [conda, "list"])
    if out is None:
        return None

    return "\n".join(
        line
        for line in out.splitlines()
        if not line.startswith("#")
        and any(name.lower() in line.lower() for name in patterns)
    )


# =============================================================================
# Part 3: Kunlun-Specific Information Collection (Core Fix)
# =============================================================================


def parse_xpu_smi_output(run_lambda):
    """
    Parse the complete output of xpu-smi command
    [Principle Explanation]
    The xpu-smi output format is similar to nvidia-smi, we need to parse it with regex.
    Example output format:
    +-----------------------------------------------------------------------------+
    | XPU-SMI               Driver Version: 515.58       XPU-RT Version: N/A      |
    |-------------------------------+----------------------+----------------------+
    |   0  P800 OAM           N/A   | 00000000:52:00.0 N/A |                    0 |
    | N/A   43C  N/A     85W / 400W |      4MiB / 98304MiB |      0%      Default |
    Returns:
        dict: Dictionary containing parsing results
    """
    rc, output, _ = run_lambda("xpu-smi")
    if rc != 0 or not output:
        return None

    result = {
        "raw_output": output,
        "driver_version": None,
        "xre_version": None,
        "devices": [],
    }

    # Parse header: Driver Version and XPU-RT Version
    # Format: | XPU-SMI               Driver Version: 515.58       XPU-RT Version: N/A      |
    header_match = re.search(
        r"Driver Version:\s*(\S+)\s+XPU-RT Version:\s*(\S+)", output
    )
    if header_match:
        result["driver_version"] = header_match.group(1)
        xre = header_match.group(2)
        result["xre_version"] = xre if xre != "N/A" else None

    # Parse device information
    # Format: |   0  P800 OAM           N/A   | 00000000:52:00.0 N/A |
    # Following: | N/A   43C  N/A     85W / 400W |      4MiB / 98304MiB |

    # Find all device lines (containing device ID and name)
    device_pattern = re.compile(
        r"\|\s*(\d+)\s+(\S+(?:\s+\S+)?)\s+(?:N/A|On|Off)\s*\|"  # ID and Name
        r"\s*([0-9a-fA-F:\.]+)\s*"  # Bus-Id
    )

    # Find memory information
    memory_pattern = re.compile(
        r"\|\s*N/A\s+\d+C\s+N/A\s+\d+W\s*/\s*\d+W\s*\|"
        r"\s*(\d+)MiB\s*/\s*(\d+)MiB\s*\|"  # Memory-Usage / Total
    )

    lines = output.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        device_match = device_pattern.search(line)
        if device_match:
            device_id = int(device_match.group(1))
            device_name = device_match.group(2).strip()
            bus_id = device_match.group(3)

            # Next line should have memory info
            memory_used = 0
            memory_total = 0
            if i + 1 < len(lines):
                mem_match = memory_pattern.search(lines[i + 1])
                if mem_match:
                    memory_used = int(mem_match.group(1))
                    memory_total = int(mem_match.group(2))

            result["devices"].append(
                {
                    "id": device_id,
                    "name": device_name,  # This will correctly get "P800 OAM"
                    "bus_id": bus_id,
                    "memory_used_mib": memory_used,
                    "memory_total_mib": memory_total,
                }
            )
        i += 1

    return result


def get_kunlun_gpu_info(run_lambda):
    """
    Get Kunlun XPU device information
    [Fix Explanation]
    Previously used torch.cuda.get_device_properties() to get the name,
    but it only returns "GPU" (because Kunlun masquerades as CUDA).
    Now parse xpu-smi output to correctly get "P800 OAM".
    Returns:
        str: Device information string
    """
    parsed = parse_xpu_smi_output(run_lambda)

    if parsed and parsed["devices"]:
        # Get real device name from xpu-smi parsing
        lines = []
        for dev in parsed["devices"]:
            memory_gb = dev["memory_total_mib"] / 1024
            # Correctly display: XPU 0: P800 OAM (96.0GB)
            lines.append(f"XPU {dev['id']}: {dev['name']} ({memory_gb:.1f}GB)")
        return "\n".join(lines)

    # Fallback: Use PyTorch interface (but will display as GPU)
    if TORCH_AVAILABLE:
        try:
            device_count = torch.cuda.device_count()
            lines = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                name = props.name if hasattr(props, "name") else "Kunlun XPU"
                memory_gb = (
                    props.total_memory / (1024**3)
                    if hasattr(props, "total_memory")
                    else 0
                )
                lines.append(f"XPU {i}: {name} ({memory_gb:.1f}GB)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    return None


def get_kunlun_driver_version(run_lambda):
    """
    Get Kunlun driver version
    [Fix Explanation]
    Parse directly from xpu-smi output header instead of calling incorrect commands.
    Returns:
        str: Driver version, e.g., "515.58"
    """
    parsed = parse_xpu_smi_output(run_lambda)
    if parsed and parsed["driver_version"]:
        return parsed["driver_version"]
    return None


def get_kunlun_xre_version(run_lambda):
    """
    Get Kunlun XRE (Runtime) version
    [Fix Explanation]
    Previously used `xpu-smi --version` but that parameter doesn't exist.
    Now parse the "XPU-RT Version" field from xpu-smi standard output header.
    Returns:
        str: XRE version, or None (if showing N/A)
    """
    parsed = parse_xpu_smi_output(run_lambda)
    if parsed and parsed["xre_version"]:
        return parsed["xre_version"]
    return "N/A (not installed or not detected)"


def get_kunlun_topo(run_lambda):
    """
    Get Kunlun XPU topology information
    Returns:
        str: Topology information
    """
    # xpu-smi topo -m command can get topology
    output = run_and_read_all(run_lambda, "xpu-smi topo -m")
    if output:
        return output

    # Fallback: Show device count
    if TORCH_AVAILABLE:
        try:
            count = torch.cuda.device_count()
            return f"Detected {count} Kunlun XPU device(s)"
        except Exception:
            pass

    return None


def get_bkcl_version(run_lambda):
    """
    Get BKCL (communication library) version
    [Principle Explanation]
    BKCL = Baidu Kunlun Communication Library
    Similar to NVIDIA's NCCL, used for multi-card communication.
    Returns:
        str: BKCL version information
    """
    # Method 1: From your logs, BKCL prints version when loading
    # [WARN][BKCL][globals.cpp:268] xccl version: 6ab4ffb [rdma] ...
    # We can try importing related modules
    try:
        # Try getting from torch_xmlir
        import torch_xmlir

        # Find path to libbkcl.so
        bkcl_path = None
        if hasattr(torch_xmlir, "__file__"):
            import os

            base = os.path.dirname(torch_xmlir.__file__)
            candidate = os.path.join(base, "libbkcl.so")
            if os.path.exists(candidate):
                bkcl_path = candidate
        if bkcl_path:
            return f"Found at: {bkcl_path}"
    except ImportError:
        pass

    # Method 2: Search from ldconfig
    rc, out, _ = run_lambda("ldconfig -p 2>/dev/null | grep -i bkcl | head -1")
    if rc == 0 and out:
        return out

    return None


def get_vllm_kunlun_version():
    """
    Get vLLM-Kunlun version
    [Fix Explanation]
    Previously got hardcoded version "0.9.2" from vllm_kunlun.platforms.version,
    but actual pip installed version is "0.1.0".
    Now prioritize using importlib.metadata to get real installed version.
    Returns:
        str: Version number
    """
    # Method 1 (recommended): Use importlib.metadata (Python 3.8+)
    try:
        from importlib.metadata import version

        return version("vllm-kunlun")
    except Exception:
        pass

    # Method 2: Use pkg_resources
    try:
        import pkg_resources

        return pkg_resources.get_distribution("vllm-kunlun").version
    except Exception:
        pass

    # Method 3 (fallback): Get from code (may be inaccurate)
    try:
        from vllm_kunlun.platforms.version import get_xvllm_version

        return get_xvllm_version() + " (from code, may be inaccurate)"
    except ImportError:
        pass

    return "N/A"


def get_vllm_version():
    """Get vLLM main package version"""
    try:
        from importlib.metadata import version

        return version("vllm")
    except Exception:
        pass

    try:
        from vllm import __version__

        return __version__
    except ImportError:
        pass

    return "N/A"


# =============================================================================
# Part 4: Environment Variable Collection
# =============================================================================


def get_kunlun_env_vars():
    """Get Kunlun-related environment variables"""
    env_vars = ""
    kunlun_prefixes = (
        "XPU",
        "KUNLUN",
        "BKCL",
        "XCCL",
        "XRE",
        "TORCH",
        "VLLM",
    )
    secret_terms = ("secret", "token", "api", "access", "password")

    for k, v in sorted(os.environ.items()):
        if any(term in k.lower() for term in secret_terms):
            continue
        if any(k.upper().startswith(prefix) for prefix in kunlun_prefixes):
            env_vars += f"{k}={v}\n"

    return env_vars


# =============================================================================
# Part 5: Define Data Structure and Formatted Output
# =============================================================================

KunlunSystemEnv = namedtuple(
    "KunlunSystemEnv",
    [
        # General system information
        "os",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "libc_version",
        "python_version",
        "python_platform",
        "pip_version",
        "pip_packages",
        "conda_packages",
        "cpu_info",
        # PyTorch information
        "torch_version",
        "is_debug_build",
        # Kunlun-specific information
        "kunlun_xpu_info",
        "kunlun_driver_version",
        "kunlun_xre_version",
        "bkcl_version",
        "kunlun_topo",
        # vLLM related
        "vllm_version",
        "vllm_kunlun_version",
        "env_vars",
    ],
)


def get_kunlun_env_info():
    """Collect all environment information"""
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    # PyTorch information
    if TORCH_AVAILABLE:
        torch_version = torch.__version__
        debug_mode_str = str(torch.version.debug)
    else:
        torch_version = "N/A"
        debug_mode_str = "N/A"

    sys_version = sys.version.replace("\n", " ")

    return KunlunSystemEnv(
        # General system information
        os=get_os(run_lambda),
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        libc_version=get_libc_version(),
        python_version=f"{sys_version} ({sys.maxsize.bit_length() + 1}-bit runtime)",
        python_platform=get_python_platform(),
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=get_conda_packages(run_lambda),
        cpu_info=get_cpu_info(run_lambda),
        # PyTorch information
        torch_version=torch_version,
        is_debug_build=debug_mode_str,
        # Kunlun-specific information
        kunlun_xpu_info=get_kunlun_gpu_info(run_lambda),
        kunlun_driver_version=get_kunlun_driver_version(run_lambda),
        kunlun_xre_version=get_kunlun_xre_version(run_lambda),
        bkcl_version=get_bkcl_version(run_lambda),
        kunlun_topo=get_kunlun_topo(run_lambda),
        # vLLM related
        vllm_version=get_vllm_version(),
        vllm_kunlun_version=get_vllm_kunlun_version(),
        env_vars=get_kunlun_env_vars(),
    )


# Output format template
kunlun_env_info_fmt = """
==============================
        System Info
==============================
OS                           : {os}
GCC version                  : {gcc_version}
Clang version                : {clang_version}
CMake version                : {cmake_version}
Libc version                 : {libc_version}
==============================
       PyTorch Info
==============================
PyTorch version              : {torch_version}
Is debug build               : {is_debug_build}
==============================
      Python Environment
==============================
Python version               : {python_version}
Python platform              : {python_platform}
==============================
    Kunlun / XPU Info
==============================
XPU models and configuration : 
{kunlun_xpu_info}
Kunlun driver version        : {kunlun_driver_version}
XRE (Runtime) version        : {kunlun_xre_version}
BKCL version                 : {bkcl_version}
XPU Topology:
{kunlun_topo}
==============================
          CPU Info
==============================
{cpu_info}
==============================
Versions of relevant libraries
==============================
{pip_packages}
{conda_packages}
==============================
      vLLM-Kunlun Info
==============================
vLLM Version                 : {vllm_version}
vLLM-Kunlun Version          : {vllm_kunlun_version}
==============================
     Environment Variables
==============================
{env_vars}
""".strip()


def pretty_str(envinfo):
    """Format environment information"""
    mutable_dict = envinfo._asdict()

    # Replace None with "Could not collect"
    for key in mutable_dict:
        if mutable_dict[key] is None:
            mutable_dict[key] = "Could not collect"

    # Handle pip package list
    if mutable_dict["pip_packages"]:
        mutable_dict["pip_packages"] = "\n".join(
            f"[{envinfo.pip_version}] {line}"
            for line in mutable_dict["pip_packages"].split("\n")
            if line
        )
    else:
        mutable_dict["pip_packages"] = "No relevant packages"

    # Handle conda package list
    if mutable_dict["conda_packages"]:
        mutable_dict["conda_packages"] = "\n".join(
            f"[conda] {line}"
            for line in mutable_dict["conda_packages"].split("\n")
            if line
        )
    else:
        mutable_dict["conda_packages"] = ""

    return kunlun_env_info_fmt.format(**mutable_dict)


def get_pretty_kunlun_env_info():
    """Get formatted environment information"""
    return pretty_str(get_kunlun_env_info())


def main():
    """Main entry point"""
    print("Collecting Kunlun XPU environment information...")
    output = get_pretty_kunlun_env_info()
    print(output)


if __name__ == "__main__":
    main()
