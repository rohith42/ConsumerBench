

### Installing necessary programs

#### Useful commands
```bash
# Shows details about the OS
cat /etc/os-release

# Shows details about the CPU
lscpu
```

#### jq
```bash
# Download the binary
wget https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64

# Make it executable and move it to ~/bin
chmod +x jq-linux-amd64
mv -i -n jq-linux-amd64 ~/bin/jq

# Validate with:
jq --version
```

#### moreutils(/ts)
```bash
# Download the RPM package
wget https://kojipkgs.fedoraproject.org//packages/moreutils/0.68/3.el10_0/x86_64/moreutils-0.68-3.el10_0.x86_64.rpm

# Extract the package and it's binaries
rpm2cpio moreutils-0.68-3.el10_0.x86_64.rpm | cpio -idmv

# Move the binaries into ~/bin
cp -r usr/bin/* ~/bin/

# Validate
ls ~/bin
which ts
echo "HELLO!" | ts
```

#### pcm(-memory)
```bash
# Download the RPM package
wget https://download.opensuse.org/repositories/home:/opcm/RHEL_7/x86_64/pcm-0-395.1.x86_64.rpm

# Extract the package and it's binaries
rpm2cpio pcm-0-395.1.x86_64.rpm | cpio -idmv

# Move the binaries into ~/bin
cp -r usr/sbin/* ~/bin/ && cp usr/bin/pcm-client ~/bin/

### TODO: NON-ROOT USER ACTIONS AND ENV VARS??
# As per: https://github.com/intel/pcm#executing-pcm-tools-under-non-root-user-on-linux

# Validate
ls ~/bin
which pcm-memory
pcm-memory --help
```

#### DCGM
```bash
# Download the RPM packages
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/datacenter-gpu-manager-4-core-4.5.3-1.x86_64.rpm
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/datacenter-gpu-manager-4-cuda13-4.5.3-1.x86_64.rpm

# Extract the package and it's binaries
rpm2cpio datacenter-gpu-manager-4-core-4.5.3-1.x86_64.rpm | cpio -idmv
rpm2cpio datacenter-gpu-manager-4-cuda13-4.5.3-1.x86_64.rpm | cpio -idmv

# Move the necessary files into the home directory
mkdir ~/lib64 ~/lib ~/.local/libexec ~/.local/share/
cp -r usr/bin/* ~/bin/
cp -r usr/lib64/* ~/lib64/
cp -r usr/lib/* ~/lib/
cp -r usr/sbin/* ~/bin/
cp -r usr/libexec/* ~/.local/libexec/
cp -r usr/share/* ~/.local/share/


# Then add the following to your ~/.bashrc:
export LD_LIBRARY_PATH="$HOME/lib64:$LD_LIBRARY_PATH"

# Then restart the shell
source ~/.bashrc

# Validate
echo $LD_LIBRARY_PATH
dcgmi discovery -l
```