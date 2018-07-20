header = """Executable = /usr/bin/matlab
Universe = vanilla
RequestMemory = 6000
RequestCPUs = 16
getenv = True
logdir = logs
output = $(logdir)/$(Cluster)-$(Process).o
error = $(logdir)/$(Cluster)-$(Process).e
log = $(logdir)/$(Cluster)-$(Process).log

args = -nodisplay -singleCompThread

"""

for subj in [1, 2, 3]:
    for roi in range(1, 15):
        header += f'arguments = $(args) -r cssmodel_vtcdata_facetask({subj},{roi});exit\n'
        header += 'queue\n\n'

submitfn = 'cssmodel_vtcdata_facetask_all.submit'

with open(submitfn, 'w') as f:
    f.writelines(header)

