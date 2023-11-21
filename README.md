#  sdfi-powerlines


## Install the two python envoriments using the install files in ./install

Change directory into the folder 
```bash
cd install
```

Install the first environment(pdal)

```bash
bash install_pdal_env.sh 
```

Install the second environment(powerlines). For this env you need to have cuda installed on your PC, and set the correct path in file like this `export CUDA_HOME=/usr/local/cuda` 

```bash
bash install_powerlines_env.sh 
```
