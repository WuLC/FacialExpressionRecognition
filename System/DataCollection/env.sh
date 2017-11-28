# setting environment for mac 
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install wget
brew install cmake
brew install boost-python --with-python3 --without-python
wget https://repo.continuum.io/miniconda/Miniconda3-4.3.30-MacOSX-x86_64.sh
pip install Pillow
pip install opencv-python
pip install dlib