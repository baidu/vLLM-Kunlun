# Operator performance

## XProfiler

### 1.Download and install

- The download link for the x86_64 platform installation package xre-Linux-x86_64 is:

`https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.26/peermem/xre-Linux-x86_64-5.0.21.26.run`

`https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.26/peermem/xre-Linux-x86_64-5.0.21.26.tar.gz`

- If the client is using bdCentOS, we recommend using the following download link:

`https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.26/xre-bdcentos-x86_64-5.0.21.26.tar.gz`

After downloading and extracting, you can directly execute `xpu-installer` and `install_rt.sh` to install.

### 2.Start using

XProfiler supports three modes: 1) fork mode; 2) time mode; and 3) daemon mode. After execution, XProfiler will generate two types of JSON files:

- xprofiler.settings.json: Records the event configuration for this trace.

- xprofiler.trace.json: Records the results of this trace.

The specific modes will be introduced below.

#### fork mode

The fork pattern is used to track the entire time period from the start to the end of a user program. This pattern is suitable for most inference tasks and is the simplest to use. An example is shown below:

```bash
/xxxx/xxxx/xprofiler -r500 --xpu=0 python test.py
```

- --r: Sets the trace time resolution in nanoseconds (ns). The default is 100. If an "out of space error" occurs, try increasing the -r value to 500.

- --xpu: Specifies the acquisition device ID, supporting multi-card configuration. --xpu=all enables all cards; the default is card 0.

More parameters can be found in the command-line parameters section later.

#### time mode

The time mode is used to track user programs for a period of time. This method is suitable for tasks that need to run for a long time.

Using the -t or --time command-line parameter, XPorfiler will run for the specified time and then exit, in seconds. In this mode, the application needs to be started separately. An example is as follows:

(1) Starting XPorfiler

```bash
/xxxx/xxxx/xprofiler -r 500 --xpu=0 -t600 # Time mode collects events within a specified time period, measured in seconds (s).
```

A temporary .sock file will be generated in the execution directory. The path needs to be configured in the environment variables.

(2) Start the program

```bash
export XPU_ENABLE_PROFILER_TRACING=1
export XPU_TRACING_OUTPUT_NAME=<xprofiler execution directory>/xprofiler.sock
# Start your own program
python xxx.py
```

#### deamon mode

The daemon mode is used to track the event timeline of a specified code segment, eliminating interference from redundant information. The startup command is the same as in fork mode.

(1) Insert start and stop interfaces.

```python
import xtorch_ops
# Only capture events during the generate phase
xtorch_ops.kunlun_profiler_start()
        outputs = llm.generate(
            inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
xtorch_ops.kunlun_profiler_end()
```

(2) Launch X profiler in a terminal

```python
# Specify the output file as the trace_output file in the current path.
/xxxx/xxxx/xprofiler-Linux_x86_64-2.0.2.0/bin/xprofiler -r 500 --xpu=0 -e ./trace_output -d
```

After startup, a .sock file will be generated in the current directory.

```bash
xprofiler.sock
```

(3) Launch your own program on another terminal.

```python
export XPU_ENABLE_PROFILER_TRACING=1
# Here, the path to the .sock file from step 2 is used for assignment.
export XPU_TRACING_OUTPUT_NAME=<xprofiler execution directory>/xprofiler.sock
# Start your own program
python xxx.py
```

Note: If you want to specify a particular card to run on, you must import the XPU_VISIBLE_DEVICES environment variable in the terminal in steps 2 and 3; otherwise, you will not be able to capture the data.

#### More parameters

| parameters                 | Example                                 | default value | describe                                                                                                                                                                                           |
| -------------------------- | --------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -b or --buffer-size        | -b=512                                  | 256           | Specifies the size of the trace buffer in MB. This is generally not required. However, if there are many trace signals, the buffer size can be increased appropriately to avoid OOS (Out of Size). |
| -x or --xpu                | -x=0--xpu=0                             | 0             | Set the card number to be tracked; multiple cards or all cards can be set.                                                                                                                         |
| -t or --time               | -t=10                                   | off           | Enable time mode, in seconds, to capture information over a specified period.                                                                                                                      |
| -d or --deamonize          | -r500                                   | 0             | Enable daemon mode to retrieve events in the background.                                                                                                                                           |
| -r or --export-profile     | -e ./trace_output-e ./output/trace.json | ./            | Record the trace results to a document or folder. If this parameter is not specified, a default xprofiler.trace.json file will be generated in the execution directory.                            |
| -S or --settings           | -S xprofiler.trace.json                 | off           | xprofiler reads a JSON file containing the events that need to be traced. If this parameter is not configured, xprofiler enables `--profile-api-trace` and `--sse-trace` by default.               |
| -A or --profiler-api-trace | -A                                      | on            | Get driver events.                                                                                                                                                                                 |
| -s or --sse-trace          | -s                                      | on            | Get all SSE events.                                                                                                                                                                                |
| -C or --cluster-trace      | -C                                      | off           | Retrieve all cluster events.                                                                                                                                                                       |
| -n or --sdnn-trace         | -n                                      | off           | Get all SDNN events.                                                                                                                                                                               |
| -c or --sdnn-cluster-trace | -c                                      | off           | Retrieve all SDNN cluster events.                                                                                                                                                                  |
| -E or --cache-trace        | -E                                      | off           | Get bandwidth statistics events.                                                                                                                                                                   |
| -u or --debug              | -u44:open log，debug level-u0:close log | 33            | Debug the interface and enable driver event/device event logging.。                                                                                                                                |

### 3.View Results

The generated xprofiler.trace.json file can be viewed and analyzed using a visual interface. Two tools are introduced here.

#### Chrome browser

Enter chrome://tracing/ in your browser (you may need to enable developer tools the first time you access this site), and click "load" in the top left corner to import the file. Interface display.

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=89aef70f112a4394adcac8b03ef994db&docGuid=WFoZOcuqnSXJIE)

#### prefetto ui

Search directly, or visit[Perfetto UI](https://ui.perfetto.dev/#!/viewer?local_cache_key)，The interface is as follows。

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=895a715344e9473c9ee93518c3064b27&docGuid=WFoZOcuqnSXJIE)

### 4.Performance Analysis

With various performance data available, analysis and optimization can then be performed based on the results.

(Further details to be added later)
