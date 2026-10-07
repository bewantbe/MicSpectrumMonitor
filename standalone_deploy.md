# How to install in an embedded environment

In some case, such as need to load win32 dll on an amd64 system, or need lagacy 
packages to run, or need a standalone version, follow steps below. You may
tweak the steps.

1. Download and unzip the embeddable package from
   <https://www.python.org/downloads/windows/>. Search for "Windows embeddable
   package (32-bit)" and unzip it to a directory such as
   `.\python-3.11.9-embed-win32`.

   Test it by running `python.exe`.
   Exit the python by `import os; os._exit(0)`, you don't have usual exit().

2. Enable customizable packages by uncommenting `import site` in
   `python311._pth`.

3. Install pip:

   ```powershell
   cd .\python-3.11.9-embed-win32
   Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
   .\python.exe .\get-pip.py
   ```

   Optionally, install also setuptools and meson-python to facilitate building source-only packages.
   ```powershell
   ./python.exe -m pip install setuptools
   ./python.exe -m pip install --upgrade wheel
   ./python.exe -m pip install --upgrade python
   ./python.exe -m pip install meson-python
   ```

4. Install dependencies. The project requirements use PyQt6; for a PyQt5
   environment, install PyQt5 instead:

   ```powershell
   .\python.exe -m pip install -r ..\requirements.txt
   ```

   If above command fails, try different versions, such as:

   ```powershell
   .\python.exe -m pip install numpy pyaudio pyqtgraph PyQt5 "matplotlib==3.7.5" --only-binary=:all:
   ```

   You might need to run above commands in 'Developer Command Prompt' in order to build and install source-only packages.

5. Add any custom import paths to `python311._pth`.

   ```
   ..\MicSpectrumMonitor
   ..\PyAD7606C
   ```

   Copy the core dll(s) to tssampler directory:

   ```
   USB2DaqsB.dll
   USBInterFace_OSCA02.dll
   ```

6. Run and test

   `./python.exe -m audiospectra`


7. Installation

   copy the context of `.\python-3.11.9-embed-win32` to your "install" location
   setup shortcut for `pyaudiospectra.bat`
