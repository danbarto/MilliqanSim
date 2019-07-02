
echo "Setting PYTHONPATH environment variable..."
export PYTHONPATH=${PWD}:${PYTHONPATH}

echo "Checking for required modules..."

echo -e "try:\n    import numpy\n    print \"-Successfully loaded numpy\"\n\
except:\n    print\"ERROR: could not import numpy!\"\n    exit(1)" | python

echo -e "try:\n    import matplotlib.pyplot\n    print \"-Successfully loaded matplotlib\"\n\
except:\n    print\"WARNING: could not import matplotlib!\"\n    exit(1)" | python

echo -e "try:\n    import ROOT\n    print \"-Successfully loaded ROOT\"\n\
except:\n    print\"WARNING: could not import pyROOT!\"\n    exit(1)" | python


