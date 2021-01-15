
echo "Setting PYTHONPATH environment variable..."
export PYTHONPATH=${PWD}:${PYTHONPATH}

echo "Checking required packages: numpy, matplotlib, uproot and yahist"

pip list | grep numpy
pip list | grep matplotlib
pip list | grep uproot
pip list | grep yahist

