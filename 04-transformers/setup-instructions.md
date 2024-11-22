# HPC Setup Instructions
This is a guide to help you setup your environment on the university HPC so that you can easily make full use 
of the HPC resources available to you. To follow this guide you will need to know how to interact with a Linux 
environment through the terminal on your OS, have signed up on our HPC project and be on the univsersity network
or be connected to the university VPN.

- [HPC Setup Instructions](#hpc-setup-instructions)
  - [Login to your Account](#login-to-your-account)
  - [Modules](#modules)
  - [Workspaces](#workspaces)
  - [Code](#code)
  - [Virtual Environments](#virtual-environments)
  - [Slurm Jobs](#slurm-jobs)


The archtictural design of HPC comprises of six homogeneous clusters with their own [Slurm][docs-slurm] instances 
and with **cluster specific login nodes** running on the same CPU. The address of the login node of each cluster
follows the pattern `login2.<cluster>.hpc.tu-dresden.de`.
<img src="https://compendium.hpc.tu-dresden.de/jobs_and_resources/misc/architecture_2024.png">

## Login to your Account
To login we will use SSH. As is standard, you can use `ssh host -l user` or `ssh user@host`.
Here `user` will be your ZIH username and `host` will be the address of the login node. 
For example, to login to Alpha you can use:
```bash
ssh login2.alpha.hpc.tu-dresden.de -l <zih-username>
```
When doing this for the first time, you will get a message asking if you trust the identity of this
server. This happens on each initial key-exchange via SSH, type in "yes". Then you will be asked for a password. 
This is your ZIH password. Type it in, you will not see any characters while typing.

Once your login was successful, you should be in a bash login shell. Great! You've reached the login node of Alpha.

Optionally, you can also [create an SSH key][docs-sshkey] to login to avoid having to type in your password each time.
Also, optionally, you can save this login configuration in your SSH config file at `~/.ssh/config`
```
Host alpha
    HostName login2.alpha.hpc.tu-dresden.de
    User <zih-username>
    IdentityFile ~/.ssh/hpc
```
This enables you to login directly with just 
```bash
ssh alpha
```

## Modules
Modules are a way to make software available in your environment.
According to the [official documentation:][docs-modules]
> A module is a user interface that provides utilities for the dynamic modification of a user's environment [...]
> By using modules, you can smoothly switch between different versions of installed software packages and libraries.

What they really are, are Lua scripts that will append paths of various binaries to your `$PATH` and `$LD_LIBRARY_PATH`
environment variables. These are important for our setup because we need to use a Python package **`virtualenv`** which is not available by
default, and the default Python version is quite outdated, Python 3.6, and we want to use Python 3.10.

1. First, remove all the loaded modules.
```bash
module purge
```
2. Run the following commands to load the two prerequisites for Python 3.10 and then the Python module.
```bash
module load release/24.04
module load GCCcore/11.3.0
module load Python/3.10.4
```
3. Now save the currently loaded modules to be loaded automaticaly at each login.
```bash
module save
```

## Workspaces
A workspace is simply a directory.

Directories inside your home directory `/home/<username>` count towards your home storage budget of 50GB.
Since we want to avoid getting our account suspended, we do our best to avoid exceeding this budget.
To this end, we use workspaces to store large files and data which we may be using.

Workspace directories exist on the storage servers, `horse` or `walrus`, which have a very large capacity. 
Each workspace has an expiry date, after which the directory is scheduled for deletion. The maximum duration
of a workspace is 100 days.

1. We recommend that each project you work on should be kept in a workspace.
   To create a workspace you just need to run the following command
```bash
ws_allocate <workspace-name> <duration>
```
2. This will create a directory `/data/horse/ws/<username>-<workspace-name>`. This is where you should keep 
   your code and data files. It is helpful to create a link to this in your home directory for easy access.
```bash
ln -s /data/horse/ws/<username>-<workspace-name> <workspace-name> 
```

## Code 
Until now you had been using notebooks on [JupyterHub][jupyterhub] to write code. While this is good for experiments and demos
but when you need to run long running tasks like training language models, this is less than ideal. This is because
you cannot install new packages that you may be using and JupyterHub instances have a limited runtime and you might not
know how long a certain task will take.

Instead we recommend that you write Python files that can be executed directly. An example of such a file was uploaded
to OPAL for under the [folder for exercise 4.](https://bildungsportal.sachsen.de/opal/auth/RepositoryEntry/46076788747/CourseNode/1731468838182965005)
To the make your code available on the HPC, commit it to a Github repository and clone it into your workspace directory.

**But I still want to use Jupyter Notebooks!**
If you still want to use Jupyter notebooks, you can write them by creating an instance on [JupyterHub][jupyterhub] as before,
but to execute them for a long running task, use the `execute` Jupyter command. This will become relevant when we [write Slurm jobs.](#slurm-jobs)
```bash
jupyter execute notebook.ipynb
```

## Virtual Environments
To install all the packages you need for a particular project, the standard Python way is to create a [virtual environment][docs-venv].
Virtual environments allow users to install additional Python packages and create an isolated run-time environment. We recommend using virtualenv for this purpose.

1. To create a virtual environment, after loading the Python module, run this command
```bash
virtualenv <venv-name>
```
2. This will create a directory with name `<venv-name>` which will hold all your Python packages. 
   It is recommended that you create this directory inside your workspace.
3. Activate the virtual environment by
```bash
source <venv-name>/bin/activate
```
4. Use `pip` to install the packages that you need.
```
(<venv-name>) login2.alpha$ pip install ...
```

## Slurm Jobs
Handling the scheduling, management and running of jobs on the clusters is done by [Slurm][docs-slurm].
A job is a process, or set of processes that run sequentially, that has compute resources attached to it.

An **interactive Slurm job** can be run with `srun [parameters] [command]`. An interactive job will wait for
allocation of resources and will block your terminal until the end of the job. This can be used for testing 
a script and confirming if everything works as expected.

A Slurm **batch job** is a set of commands that will be executed in order and these are described in an executable Bash file. 
This file contains the parameters of the job, along with the commands to execute. 
These are written in comments starting with `#SBATCH` followed by the parameter name and value.
An Slurm batch job can be run with `sbatch script.sh`. This is the preferred choice when you've got your code
working and want to run it, for example to train a model, because it does not block your terminal.

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=ws_secretllm/logs/gpt2-%j.log
#SBATCH --error=ws_secretllm/logs/gpt2-%j.log
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=email@mailbox.tu-dresden.de
#SBATCH --account=p_scads_llm_secrets

source $HOME/ws_secretllm/venv/bin/activate

python $HOME/ws_secretllm/roberta.py \
	--language hi \
	--output_directory $HOME/ws_secretllm/models \
	--epochs 10
```
Each job has an ID which usually a 6 digit number.
- Check the status of a job by `scontrol show job <ID>`
- See all your jobs by `squeue -u $(whoami)`
- Cancel a job with `scancel <ID>`

Read more about [how to write Slurm jobs][docs-jobs].

<!-- Links -->
[docs]: https://compendium.hpc.tu-dresden.de/
[jupyterhub]: https://jupyterhub.hpc.tu-dresden.de/
[docs-slurm]: https://compendium.hpc.tu-dresden.de/jobs_and_resources/slurm/
[docs-modules]: https://compendium.hpc.tu-dresden.de/software/modules/?h=modules
[docs-venv]: https://compendium.hpc.tu-dresden.de/software/python_virtual_environments/?h=virtual
[docs-sshkey]: https://compendium.hpc.tu-dresden.de/access/ssh_login/?h=#before-your-first-connection
[docs-slurm]: https://compendium.hpc.tu-dresden.de/jobs_and_resources/slurm/
[docs-jobs]: https://compendium.hpc.tu-dresden.de/jobs_and_resources/slurm_examples/
[slurm-gen]: https://compendium.hpc.tu-dresden.de/jobs_and_resources/slurm_generator/