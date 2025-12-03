Steps:

MACOS (Silicon)

1. Follow this files readme set up with conda and pip making sure to install python 3.10
- tried to use venv at first and failed on MACOS

2. Next need to link pycffirmware as explain in examples/cf.py
I took these steps:
    a.  git clone https://github.com/utiasDSL/pycffirmware.git
        cd pycffirmware/
        git submodule update --init --recursive
    b.  pip install numpy
        brew install swig
        brew install gcc 
        brew install make
    c.  KEY STEP: update build_osx.sh to update gcc make path
        for example mine is "make CC=/opt/homebrew/bin/gcc-15"
        I also had to update this to hav lines:
            "export CSW_PYTHON=python" 
            "python -m pip install -e ."
        and change lines:
            "export CSW_PYTHON=python3" 
            "python3 -m pip install -e ."
        because I might have done something irregular in my set up
    d.  in pycffirmware/wrapper/pycffirmware.i I had to:
        add:
            '#include "controller_pid.h"'
            '#include "controller_mellinger.h"'
        below "#define SWIG_FILE_WITH_INIT" with other includes
        I also commented out:
        "
            %init %{
                import_array()
            %}
        "
        But I am unsure if its needed
    e.  Then I ran the shell script to compile
        make clean
        ./wrapper/build_osx.sh
    f.  Then I activated my drones conda environment and copied the python module into it
        conda activate drones && cp pycffirmware.py _pycffirmware.cpython-310-darwin.so /Users/masonstark/miniconda3/envs/drones/lib/python3.10/site-packages/
        YOU WILL NEED TO UPDATE THIS OBVIOUSLY
    g.  I went into gym_pybullet_drones/examples/cf.py and updated:
        DEFAULT_SIMULATION_FREQ_HZ = 1000 (CHANGES ARE ALREADY IN THIS COMMIT)
    h.  within gym_pybullet_drones/envs/CFAviary.py you can toggle the controller between
        'mellinger' and 'pid'. both work.
        Then from conda (drones) run python gym_pybullet_drones/examples/cf.py
        to see the example run
        
    



