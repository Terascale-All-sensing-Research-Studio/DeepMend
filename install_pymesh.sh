# Make libs
if [ ! -d "libs" ] 
then
    mkdir libs
fi
cd libs &&

# This is necessasry to install the latest version of cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get update

# Install CGAL
sudo apt-get install -y \
    libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev \
    python3-dev \
    libcgal-dev \
    cmake

# Clone and install
git clone https://github.com/PyMesh/PyMesh.git && \
cd PyMesh && \
git submodule update --init && \
./setup.py build && \
./setup.py install
